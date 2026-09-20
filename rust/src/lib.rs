#![forbid(unsafe_code)]

use numpy::{
    dtype, Element, PyArray1, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods,
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

fn add_channel<T: Element + Copy + ToF32>(
    data: &PyReadonlyArray2<'_, T>,
    channel: usize,
    delay: i64,
    output: &mut [f32],
) {
    let view = data.as_array();
    let column = view.column(channel);
    let nt = output.len();

    if delay >= nt as i64 || delay <= -(nt as i64) || delay == 0 {
        for (sum, value) in output.iter_mut().zip(column.iter()) {
            *sum += (*value).to_f32();
        }
    } else if delay > 0 {
        let split = nt - delay as usize;
        let values = column.iter().skip(split).chain(column.iter().take(split));
        for (sum, value) in output.iter_mut().zip(values) {
            *sum += (*value).to_f32();
        }
    } else {
        let shift = delay.unsigned_abs() as usize;
        let values = column.iter().skip(shift).chain(column.iter().take(shift));
        for (sum, value) in output.iter_mut().zip(values) {
            *sum += (*value).to_f32();
        }
    }
}

fn dedisperse_array<'py, T: Element + Copy + ToF32>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, T>,
    delay_bins: &PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
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
        .try_reserve_exact(nt)
        .map_err(|_| PyMemoryError::new_err("dedispersed output is too large"))?;
    output.resize(nt, 0.0_f32);
    let contiguous = if data.is_c_contiguous() {
        data.as_slice().ok()
    } else {
        None
    };
    if let Some(contiguous) = contiguous {
        // Keep a short time span hot while accumulating channels in their original order.
        for (block_index, block) in output.chunks_mut(64).enumerate() {
            let time = block_index * 64;
            for (channel, &delay) in delays.iter().enumerate() {
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
        for (channel, &delay) in delays.iter().enumerate() {
            add_channel(&data, channel, delay, &mut output);
        }
    }
    Ok(PyArray1::from_vec(py, output))
}

#[pyfunction]
fn dedispersets<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    delay_bins: PyReadonlyArray1<'py, i64>,
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
            return dedisperse_array(py, data.cast::<PyArray2<$ty>>()?.readonly(), &delay_bins)
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

#[pymodule]
fn _rust(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(dedispersets, module)?)?;
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
