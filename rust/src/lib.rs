#![forbid(unsafe_code)]

use numpy::{
    dtype, ndarray::Array2, Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods,
    PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyMemoryError, PyTypeError, PyValueError},
    prelude::*,
    types::PyModuleMethods,
};

trait ToF32 {
    fn to_f32(self) -> f32;
}

macro_rules! impl_to_f32 {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ToF32 for $ty {
                #[inline]
                fn to_f32(self) -> f32 {
                    self as f32
                }
            }
        )+
    };
}

impl_to_f32!(u8, u16, i16, i32, f32, f64);

const CHANNEL_TILE: usize = 32;

// Match NumPy's contiguous float64 reduction without a squared-data array.
const PAIRWISE_BLOCK: usize = 128;

fn pairwise_squared_deviations(values: &[u8], mean: f64) -> f64 {
    let squared = |value: u8| {
        if value == 0 {
            0.0
        } else {
            let deviation = f64::from(value) - mean;
            deviation * deviation
        }
    };
    match values.len() {
        0..8 => values.iter().fold(-0.0, |sum, &value| sum + squared(value)),
        8..=PAIRWISE_BLOCK => {
            let mut sums = [
                squared(values[0]),
                squared(values[1]),
                squared(values[2]),
                squared(values[3]),
                squared(values[4]),
                squared(values[5]),
                squared(values[6]),
                squared(values[7]),
            ];
            let full = values.len() - values.len() % 8;
            let mut index = 8;
            while index < full {
                for offset in 0..8 {
                    sums[offset] += squared(values[index + offset]);
                }
                index += 8;
            }
            let mut sum = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
            while index < values.len() {
                sum += squared(values[index]);
                index += 1;
            }
            sum
        }
        _ => {
            let mut middle = values.len() / 2;
            middle -= middle % 8;
            pairwise_squared_deviations(&values[..middle], mean)
                + pairwise_squared_deviations(&values[middle..], mean)
        }
    }
}

type RfiStats<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    u64,
    u64,
    f64,
);

#[pyfunction]
fn rfi_stats<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, u8>,
) -> PyResult<Option<RfiStats<'py>>> {
    if !data.is_c_contiguous() {
        return Ok(None);
    }
    let values = match data.as_slice() {
        Ok(values) => values,
        Err(_) => return Ok(None),
    };
    // The channel sums are returned as float64, so retain exact integer sums.
    if u64::try_from(values.len()).unwrap_or(u64::MAX) > (1_u64 << 53) / 65_025 {
        return Ok(None);
    }

    let channels = data.shape()[1];
    if channels == 0 {
        return Ok(None);
    }
    let mut s1 = Vec::new();
    let mut s2 = Vec::new();
    s1.try_reserve_exact(channels)
        .and_then(|_| s2.try_reserve_exact(channels))
        .map_err(|_| PyMemoryError::new_err("RFI channel statistics are too large"))?;
    s1.resize(channels, 0.0);
    s2.resize(channels, 0.0);

    let mut total = 0_u64;
    let mut count = 0_u64;
    for frame in values.chunks_exact(channels) {
        for (channel, &value) in frame.iter().enumerate() {
            if value != 0 {
                let value = u64::from(value);
                total += value;
                count += 1;
                s1[channel] += value as f64;
                s2[channel] += (value * value) as f64;
            }
        }
    }
    if count == 0 {
        return Ok(None);
    }

    let mean = total as f64 / count as f64;
    let variance = pairwise_squared_deviations(values, mean);
    Ok(Some((
        PyArray1::from_vec(py, s1),
        PyArray1::from_vec(py, s2),
        total,
        count,
        variance,
    )))
}

fn add_channel<T: Element + Copy + ToF32>(
    data: &PyReadonlyArray2<'_, T>,
    channel: usize,
    delay: i64,
    start: usize,
    output: &mut [f32],
) {
    let view = data.as_array();
    let column = view.column(channel);
    let nt = data.shape()[0];

    if delay >= nt as i64 || delay <= -(nt as i64) || delay == 0 {
        for (sum, value) in output.iter_mut().zip(column.iter().skip(start)) {
            *sum += (*value).to_f32();
        }
    } else if delay > 0 {
        let split = nt - delay as usize;
        let values = column.iter().skip(split).chain(column.iter().take(split));
        for (sum, value) in output.iter_mut().zip(values.skip(start)) {
            *sum += (*value).to_f32();
        }
    } else {
        let shift = delay.unsigned_abs() as usize;
        let values = column.iter().skip(shift).chain(column.iter().take(shift));
        for (sum, value) in output.iter_mut().zip(values.skip(start)) {
            *sum += (*value).to_f32();
        }
    }
}

fn fill_dedispersed<T, I>(
    data: &PyReadonlyArray2<'_, T>,
    delays: I,
    start: usize,
    output: &mut [f32],
) where
    T: Element + Copy + ToF32,
    I: Clone + Iterator<Item = i64>,
{
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    let contiguous = if data.is_c_contiguous() {
        data.as_slice().ok()
    } else {
        None
    };
    if let Some(contiguous) = contiguous {
        // Keep a short time span hot while accumulating channels in their original order.
        for (block_index, block) in output.chunks_mut(64).enumerate() {
            let time = start + block_index * 64;
            for (channel, delay) in delays.clone().enumerate() {
                let pivot = if delay >= nt as i64 || delay <= -(nt as i64) {
                    0
                } else if delay > 0 {
                    nt - delay as usize
                } else {
                    delay.unsigned_abs() as usize
                };
                let wrap = nt - pivot;
                let source = if time >= wrap {
                    time - wrap
                } else {
                    time + pivot
                };
                let count = block.len().min(nt - source);
                let (first, second) = block.split_at_mut(count);
                add_values(
                    first,
                    contiguous[source * nf + channel..].iter().step_by(nf),
                );
                if !second.is_empty() {
                    add_values(second, contiguous[channel..].iter().step_by(nf));
                }
            }
        }
    } else {
        for (channel, delay) in delays.enumerate() {
            add_channel(data, channel, delay, start, output);
        }
    }
}

fn add_packed_channel<T: Copy + ToF32>(column: &[T], delay: i64, start: usize, output: &mut [f32]) {
    let nt = column.len();
    let pivot = if delay >= nt as i64 || delay <= -(nt as i64) {
        0
    } else if delay > 0 {
        nt - delay as usize
    } else {
        delay.unsigned_abs() as usize
    };
    let wrap = nt - pivot;
    let source = if start >= wrap {
        start - wrap
    } else {
        start + pivot
    };
    let count = output.len().min(nt - source);
    let (first, second) = output.split_at_mut(count);
    add_values(first, column[source..].iter());
    if !second.is_empty() {
        add_values(second, column[..].iter());
    }
}

fn fill_dmtime_packed<T: Copy + ToF32>(
    data: &[T],
    (nt, nf): (usize, usize),
    dm_values: &PyReadonlyArray1<'_, f64>,
    frequency_term: &PyReadonlyArray1<'_, f64>,
    tsamp: f64,
    start: usize,
    output: &mut [f32],
) -> bool {
    let channel_count = CHANNEL_TILE.min(nf);
    let Some(packed_len) = nt.checked_mul(channel_count) else {
        return false;
    };
    let Some(&first) = data.first() else {
        return false;
    };
    let mut packed = Vec::new();
    if packed.try_reserve_exact(packed_len).is_err() {
        return false;
    }
    packed.resize(packed_len, first);

    let dm_values = dm_values.as_array();
    let terms = frequency_term.as_array();
    let length = output.len() / dm_values.len();
    for channel_start in (0..nf).step_by(channel_count) {
        let width = (nf - channel_start).min(channel_count);
        for (time, frame) in data.chunks_exact(nf).enumerate() {
            for channel in 0..width {
                packed[channel * nt + time] = frame[channel_start + channel];
            }
        }
        for (row_index, &dm) in dm_values.iter().enumerate() {
            let row = &mut output[row_index * length..(row_index + 1) * length];
            for (channel, &term) in terms.iter().skip(channel_start).take(width).enumerate() {
                let delay = numpy_i64(((4148808.0 * dm) * term / 1000.0 / tsamp).round_ties_even());
                add_packed_channel(&packed[channel * nt..(channel + 1) * nt], delay, start, row);
            }
        }
    }
    true
}

fn output_length(nt: usize, start: usize, stop: Option<usize>) -> PyResult<usize> {
    let stop = stop.unwrap_or(nt);
    if start > stop || stop > nt {
        return Err(PyValueError::new_err(
            "expected 0 <= start <= stop <= data length",
        ));
    }
    Ok(stop - start)
}

fn dedisperse_array<'py, T: Element + Copy + ToF32>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, T>,
    delay_bins: &PyReadonlyArray1<'py, i64>,
    start: usize,
    stop: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    let length = output_length(nt, start, stop)?;
    let delays = delay_bins.as_array();
    if delays.len() != nf {
        return Err(PyValueError::new_err(format!(
            "delay_bins has length {}, expected {}",
            delays.len(),
            nf
        )));
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))?;
    output.resize(length, 0.0_f32);
    fill_dedispersed(&data, delays.iter().copied(), start, &mut output);
    Ok(PyArray1::from_vec(py, output))
}

fn fill_dedisperse<T: Element + Copy + ToF32>(
    data: &PyReadonlyArray2<'_, T>,
    delays: &PyReadonlyArray1<'_, i64>,
    start: usize,
    output: &mut [f32],
) -> PyResult<()> {
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    if nt == 0 || nf == 0 || output.is_empty() {
        return Ok(());
    }

    let mut sources = Vec::new();
    sources
        .try_reserve_exact(nf)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))?;
    for (channel, &delay) in delays.as_array().iter().enumerate() {
        let source = if delay >= nt as i64 || delay <= -(nt as i64) {
            start
        } else if delay > 0 {
            let delay = delay as usize;
            if start >= delay {
                start - delay
            } else {
                start + nt - delay
            }
        } else if delay < 0 {
            let delay = delay.unsigned_abs() as usize;
            if start < nt - delay {
                start + delay
            } else {
                start - (nt - delay)
            }
        } else {
            start
        };
        sources.push(source * nf + channel);
    }

    let contiguous = if data.is_c_contiguous() {
        data.as_slice().ok()
    } else {
        None
    };
    if let Some(data) = contiguous {
        let data_len = data.len();
        for row in output.chunks_exact_mut(nf) {
            for (value, source) in row.iter_mut().zip(sources.iter_mut()) {
                *value = data[*source].to_f32();
                *source += nf;
                if *source >= data_len {
                    *source -= data_len;
                }
            }
        }
    } else {
        let data = data.as_array();
        for row in output.chunks_exact_mut(nf) {
            for (channel, (value, source)) in row.iter_mut().zip(sources.iter_mut()).enumerate() {
                *value = data[(*source / nf, channel)].to_f32();
                *source += nf;
                if *source >= nt * nf {
                    *source -= nt * nf;
                }
            }
        }
    }
    Ok(())
}

fn dedisperse_channels_array<'py, T: Element + Copy + ToF32>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, T>,
    delay_bins: &PyReadonlyArray1<'py, i64>,
    start: usize,
    stop: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    let length = output_length(nt, start, stop)?;
    let delays = delay_bins.as_array();
    if delays.len() != nf {
        return Err(PyValueError::new_err(format!(
            "delay_bins has length {}, expected {}",
            delays.len(),
            nf
        )));
    }
    let total = length
        .checked_mul(nf)
        .ok_or_else(|| PyMemoryError::new_err("dedispersed output is too large"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(total)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))?;
    output.resize(total, 0.0_f32);
    fill_dedisperse(&data, delay_bins, start, &mut output)?;
    Array2::from_shape_vec((length, nf), output)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))
        .map(|output| output.into_pyarray(py))
}

fn numpy_i64(value: f64) -> i64 {
    const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_F64: f64 = 9_223_372_036_854_775_808.0;

    if value.is_finite() && (I64_MIN_F64..I64_MAX_F64).contains(&value) {
        value as i64
    } else {
        i64::MIN
    }
}

fn dmtime_array<'py, T: Element + Copy + ToF32>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, T>,
    dm_values: &PyReadonlyArray1<'py, f64>,
    frequency_term: &PyReadonlyArray1<'py, f64>,
    tsamp: f64,
    start: usize,
    stop: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    let length = output_length(nt, start, stop)?;
    let terms = frequency_term.as_array();
    if terms.len() != nf {
        return Err(PyValueError::new_err(format!(
            "frequency_term has length {}, expected {}",
            terms.len(),
            nf
        )));
    }

    let dm_values_array = dm_values.as_array();
    let count = dm_values_array.len();
    let total = count
        .checked_mul(length)
        .ok_or_else(|| PyMemoryError::new_err("dmtime output is too large"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(total)
        .map_err(|_| PyMemoryError::new_err("dmtime output is too large"))?;
    output.resize(total, 0.0_f32);
    // Packing visits the full input; keep sparse requests on the existing path.
    let packed =
        if count > 1 && length > 0 && nt > 0 && nf > 0 && nt <= total && data.is_c_contiguous() {
            match data.as_slice() {
                Ok(contiguous) => fill_dmtime_packed(
                    contiguous,
                    (nt, nf),
                    dm_values,
                    frequency_term,
                    tsamp,
                    start,
                    &mut output,
                ),
                Err(_) => false,
            }
        } else {
            false
        };
    if !packed {
        let mut delays = Vec::new();
        delays
            .try_reserve_exact(nf)
            .map_err(|_| PyMemoryError::new_err("dmtime delay buffer is too large"))?;
        delays.resize(nf, i64::MIN);

        for (row_index, &dm) in dm_values_array.iter().enumerate() {
            for (delay, &term) in delays.iter_mut().zip(terms.iter()) {
                *delay = numpy_i64(((4148808.0 * dm) * term / 1000.0 / tsamp).round_ties_even());
            }
            let row = &mut output[row_index * length..(row_index + 1) * length];
            fill_dedispersed(&data, delays.iter().copied(), start, row);
        }
    }

    Array2::from_shape_vec((count, length), output)
        .map_err(|_| PyMemoryError::new_err("dmtime output is too large"))
        .map(|output| output.into_pyarray(py))
}

#[pyfunction(signature = (data, delay_bins, start=0, stop=None))]
fn dedispersets<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    delay_bins: PyReadonlyArray1<'py, i64>,
    start: usize,
    stop: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let data = data.cast::<PyUntypedArray>()?;
    if data.ndim() != 2 {
        return Err(PyValueError::new_err(
            "data must be a two-dimensional array",
        ));
    }
    if !data.is_aligned() || !delay_bins.is_aligned() {
        return Err(PyValueError::new_err("data and delay_bins must be aligned"));
    }

    let data_dtype = data.dtype();
    let type_num = data_dtype.num();
    macro_rules! dispatch {
        ($ty:ty) => {
            return dedisperse_array(
                py,
                data.cast::<PyArray2<$ty>>()?.readonly(),
                &delay_bins,
                start,
                stop,
            )
        };
    }

    if type_num == dtype::<u8>(py).num() {
        dispatch!(u8);
    } else if type_num == dtype::<u16>(py).num() {
        dispatch!(u16);
    } else if type_num == dtype::<i16>(py).num() {
        dispatch!(i16);
    } else if type_num == dtype::<i32>(py).num() {
        dispatch!(i32);
    } else if type_num == dtype::<f32>(py).num() {
        dispatch!(f32);
    } else if type_num == dtype::<f64>(py).num() {
        dispatch!(f64);
    }

    Err(PyTypeError::new_err(
        "data dtype is not supported by your._rust.dedispersets",
    ))
}

#[pyfunction(signature = (data, delay_bins, start=0, stop=None))]
fn dedisperse<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    delay_bins: PyReadonlyArray1<'py, i64>,
    start: usize,
    stop: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data = data.cast::<PyUntypedArray>()?;
    if data.ndim() != 2 {
        return Err(PyValueError::new_err(
            "data must be a two-dimensional array",
        ));
    }
    if !data.is_aligned() || !delay_bins.is_aligned() {
        return Err(PyValueError::new_err("data and delay_bins must be aligned"));
    }

    let type_num = data.dtype().num();
    macro_rules! dispatch {
        ($ty:ty) => {
            return dedisperse_channels_array(
                py,
                data.cast::<PyArray2<$ty>>()?.readonly(),
                &delay_bins,
                start,
                stop,
            )
        };
    }

    if type_num == dtype::<u8>(py).num() {
        dispatch!(u8);
    } else if type_num == dtype::<u16>(py).num() {
        dispatch!(u16);
    } else if type_num == dtype::<i16>(py).num() {
        dispatch!(i16);
    } else if type_num == dtype::<i32>(py).num() {
        dispatch!(i32);
    } else if type_num == dtype::<f32>(py).num() {
        dispatch!(f32);
    } else if type_num == dtype::<f64>(py).num() {
        dispatch!(f64);
    }

    Err(PyTypeError::new_err(
        "data dtype is not supported by your._rust.dedisperse",
    ))
}

#[pyfunction(signature = (data, dm_values, frequency_term, tsamp, start=0, stop=None))]
fn dmtime<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    dm_values: PyReadonlyArray1<'py, f64>,
    frequency_term: PyReadonlyArray1<'py, f64>,
    tsamp: f64,
    start: usize,
    stop: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let data = data.cast::<PyUntypedArray>()?;
    if data.ndim() != 2 {
        return Err(PyValueError::new_err(
            "data must be a two-dimensional array",
        ));
    }
    if !data.is_aligned() || !dm_values.is_aligned() || !frequency_term.is_aligned() {
        return Err(PyValueError::new_err(
            "data, dm_values, and frequency_term must be aligned",
        ));
    }

    let type_num = data.dtype().num();
    macro_rules! dispatch {
        ($ty:ty) => {
            return dmtime_array(
                py,
                data.cast::<PyArray2<$ty>>()?.readonly(),
                &dm_values,
                &frequency_term,
                tsamp,
                start,
                stop,
            )
        };
    }

    if type_num == dtype::<u8>(py).num() {
        dispatch!(u8);
    } else if type_num == dtype::<u16>(py).num() {
        dispatch!(u16);
    } else if type_num == dtype::<i16>(py).num() {
        dispatch!(i16);
    } else if type_num == dtype::<i32>(py).num() {
        dispatch!(i32);
    } else if type_num == dtype::<f32>(py).num() {
        dispatch!(f32);
    } else if type_num == dtype::<f64>(py).num() {
        dispatch!(f64);
    }

    Err(PyTypeError::new_err(
        "data dtype is not supported by your._rust.dmtime",
    ))
}

#[pymodule]
fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dedisperse, module)?)?;
    module.add_function(wrap_pyfunction!(dedispersets, module)?)?;
    module.add_function(wrap_pyfunction!(dmtime, module)?)?;
    module.add_function(wrap_pyfunction!(rfi_stats, module)?)?;
    Ok(())
}

fn add_values<'a, T, I>(output: &mut [f32], values: I)
where
    T: Copy + ToF32 + 'a,
    I: Iterator<Item = &'a T>,
{
    for (sum, value) in output.iter_mut().zip(values) {
        *sum += (*value).to_f32();
    }
}
