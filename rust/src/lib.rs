#![forbid(unsafe_code)]

mod candidates;
mod kernels;
mod resize;

use kernels::{
    fill_dedisperse, fill_dedispersed, fill_dmtime_packed, fill_rfi_stats, numpy_i64, ToF32,
};

use numpy::{
    dtype, ndarray::Array2, Element, IntoPyArray, PyArray1, PyArray2, PyArrayDescrMethods,
    PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyMemoryError, PyTypeError, PyValueError},
    prelude::*,
    types::PyModuleMethods,
};

type RfiStats<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    u64,
    u64,
    f64,
);

type CandidateProducts<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>);

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

    let (total, count, variance) = fill_rfi_stats(values, channels, &mut s1, &mut s2);
    if count == 0 {
        return Ok(None);
    }
    Ok(Some((
        PyArray1::from_vec(py, s1),
        PyArray1::from_vec(py, s2),
        total,
        count,
        variance,
    )))
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
    fill_dedispersed(data.as_array(), delays.iter().copied(), start, &mut output);
    Ok(PyArray1::from_vec(py, output))
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
    fill_dedisperse(data.as_array(), delays, start, &mut output)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))?;
    Array2::from_shape_vec((length, nf), output)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))
        .map(|output| output.into_pyarray(py))
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
                    dm_values_array,
                    terms,
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
            fill_dedispersed(data.as_array(), delays.iter().copied(), start, row);
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

#[allow(clippy::too_many_arguments)] // Keep validation aligned with the Python entry point.
fn candidate_binned_view<'py, T: candidates::CandidateSample>(
    py: Python<'py>,
    data: numpy::ndarray::ArrayView2<'_, T>,
    dm_delay_bins: numpy::ndarray::ArrayView2<'_, i64>,
    ft_delay_bins: numpy::ndarray::ArrayView1<'_, i64>,
    enabled: Option<numpy::ndarray::ArrayView1<'_, bool>>,
    start: usize,
    time_bin: usize,
    time_size: usize,
    threads: usize,
) -> PyResult<CandidateProducts<'py>> {
    if time_bin == 0 || time_size == 0 {
        return Err(PyValueError::new_err(
            "time_bin and time_size must be positive",
        ));
    }

    let (nt, nf) = (data.shape()[0], data.shape()[1]);
    let dmsteps = dm_delay_bins.shape()[0];
    if dm_delay_bins.shape() != [dmsteps, nf]
        || ft_delay_bins.shape() != [nf]
        || enabled.is_some_and(|mask| mask.shape() != [nf])
    {
        return Err(PyValueError::new_err(
            "delay bins and enabled mask must match the input channels",
        ));
    }
    if nf == 0 || dmsteps == 0 {
        return Err(PyValueError::new_err(
            "candidate products require channels and DM trials",
        ));
    }

    let max_delay = dm_delay_bins.iter().chain(ft_delay_bins.iter()).try_fold(
        0_usize,
        |max_delay, &delay| {
            let delay = usize::try_from(delay)
                .map_err(|_| PyValueError::new_err("delay bins must be nonnegative"))?;
            Ok::<_, PyErr>(max_delay.max(delay))
        },
    )?;
    let native_len = time_bin
        .checked_mul(time_size)
        .ok_or_else(|| PyMemoryError::new_err("candidate output is too large"))?;
    let end = start
        .checked_add(native_len)
        .and_then(|end| end.checked_add(max_delay))
        .ok_or_else(|| PyValueError::new_err("candidate source range overflows"))?;
    if end > nt {
        return Err(PyValueError::new_err(
            "candidate source samples escape the nonwrapping input",
        ));
    }
    nf.checked_mul(time_bin)
        .and_then(|_| nf.checked_mul(time_size))
        .and_then(|_| dmsteps.checked_mul(time_size))
        .and_then(|_| time_size.checked_add(1))
        .and_then(|boundary_width| dmsteps.checked_mul(boundary_width))
        .and_then(|_| nt.checked_add(1))
        .and_then(|_| nt.checked_mul(8))
        .ok_or_else(|| PyMemoryError::new_err("candidate products are too large"))?;

    let (ft, dmt) = py
        .detach(|| {
            candidates::binned_products(
                data,
                dm_delay_bins,
                ft_delay_bins,
                enabled,
                start,
                time_bin,
                time_size,
                threads,
            )
        })
        .map_err(|_| PyMemoryError::new_err("candidate products are too large"))?;
    let ft = Array2::from_shape_vec((time_size, nf), ft)
        .map_err(|_| PyMemoryError::new_err("candidate products are too large"))?
        .into_pyarray(py);
    let dmt = Array2::from_shape_vec((dmsteps, time_size), dmt)
        .map_err(|_| PyMemoryError::new_err("candidate products are too large"))?
        .into_pyarray(py);
    Ok((ft, dmt))
}

#[pyfunction(signature = (data, dm_delay_bins, ft_delay_bins, start, time_bin, time_size, enabled=None, threads=1))]
#[allow(clippy::too_many_arguments)] // Python also supplies the interpreter token.
fn candidate_binned<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    dm_delay_bins: PyReadonlyArray2<'py, i64>,
    ft_delay_bins: PyReadonlyArray1<'py, i64>,
    start: usize,
    time_bin: usize,
    time_size: usize,
    enabled: Option<PyReadonlyArray1<'py, bool>>,
    threads: usize,
) -> PyResult<CandidateProducts<'py>> {
    let data = data.cast::<PyUntypedArray>()?;
    if data.ndim() != 2 {
        return Err(PyValueError::new_err(
            "data must be a two-dimensional array",
        ));
    }
    if !data.is_aligned() || !dm_delay_bins.is_aligned() || !ft_delay_bins.is_aligned() {
        return Err(PyValueError::new_err("data and delay bins must be aligned"));
    }
    if enabled.as_ref().is_some_and(|mask| !mask.is_aligned()) {
        return Err(PyValueError::new_err("enabled mask must be aligned"));
    }

    let dm_delays = dm_delay_bins.as_array();
    let ft_delays = ft_delay_bins.as_array();
    let enabled = enabled.as_ref().map(|mask| mask.as_array());
    let type_num = data.dtype().num();
    macro_rules! dispatch {
        ($ty:ty) => {{
            let typed = data.cast::<PyArray2<$ty>>()?.readonly();
            candidate_binned_view(
                py,
                typed.as_array(),
                dm_delays,
                ft_delays,
                enabled,
                start,
                time_bin,
                time_size,
                threads,
            )
        }};
    }

    if type_num == dtype::<bool>(py).num() {
        dispatch!(bool)
    } else if type_num == dtype::<u8>(py).num() {
        dispatch!(u8)
    } else if type_num == dtype::<u16>(py).num() {
        dispatch!(u16)
    } else if type_num == dtype::<u32>(py).num() {
        dispatch!(u32)
    } else if type_num == dtype::<u64>(py).num() {
        dispatch!(u64)
    } else if type_num == dtype::<i8>(py).num() {
        dispatch!(i8)
    } else if type_num == dtype::<i16>(py).num() {
        dispatch!(i16)
    } else if type_num == dtype::<i32>(py).num() {
        dispatch!(i32)
    } else if type_num == dtype::<i64>(py).num() {
        dispatch!(i64)
    } else if type_num == dtype::<f64>(py).num() {
        dispatch!(f64)
    } else {
        Err(PyTypeError::new_err(
            "data dtype is not supported by your._rust.candidate_binned",
        ))
    }
}

/// skimage's `resize(data, (nt, nf // factor), anti_aliasing=True,
/// mode="constant")` of float32 rows before its clip.
#[pyfunction]
fn gaussian_decimate<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    weights: PyReadonlyArray1<'py, f64>,
    factor: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (nt, nf) = (data.shape()[0], data.shape()[1]);
    if factor == 0 || nf == 0 || nf % factor != 0 {
        return Err(PyValueError::new_err(
            "expected factor >= 1 dividing a non-zero number of columns",
        ));
    }
    let values = data
        .as_slice()
        .map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
    let weights = weights
        .as_slice()
        .map_err(|_| PyValueError::new_err("weights must be contiguous"))?;
    if weights.is_empty() {
        return Err(PyValueError::new_err("weights must not be empty"));
    }
    let size = nf / factor;
    if nt == 0 {
        return Array2::from_shape_vec((nt, size), Vec::<f32>::new())
            .map_err(|_| PyMemoryError::new_err("resized plane is too large"))
            .map(|out| out.into_pyarray(py));
    }
    let output_len = nt
        .checked_mul(size)
        .ok_or_else(|| PyMemoryError::new_err("resized plane is too large"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(output_len)
        .map_err(|_| PyMemoryError::new_err("resized plane is too large"))?;
    out.resize(output_len, 0.0_f32);
    py.detach(|| resize::gaussian_decimate(values, nf, weights, factor, &mut out))
        .map_err(|_| PyMemoryError::new_err("resized plane is too large"))?;
    Array2::from_shape_vec((nt, size), out)
        .map_err(|_| PyMemoryError::new_err("resized plane is too large"))
        .map(|out| out.into_pyarray(py))
}

#[pymodule]
fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dedisperse, module)?)?;
    module.add_function(wrap_pyfunction!(dedispersets, module)?)?;
    module.add_function(wrap_pyfunction!(dmtime, module)?)?;
    module.add_function(wrap_pyfunction!(candidate_binned, module)?)?;
    module.add_function(wrap_pyfunction!(rfi_stats, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_decimate, module)?)?;
    Ok(())
}
