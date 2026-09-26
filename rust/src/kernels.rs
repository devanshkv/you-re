//! Computation on borrowed views and Rust buffers. Python validation stays in lib.rs.

use std::collections::TryReserveError;

use numpy::ndarray::{ArrayView1, ArrayView2};

pub(crate) trait ToF32 {
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

pub(crate) fn pairwise_squared_deviations(values: &[u8], mean: f64) -> f64 {
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

fn add_channel<T: Copy + ToF32>(
    data: ArrayView2<'_, T>,
    channel: usize,
    delay: i64,
    start: usize,
    output: &mut [f32],
) {
    let column = data.column(channel);
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

pub(crate) fn fill_dedispersed<T, I>(
    data: ArrayView2<'_, T>,
    delays: I,
    start: usize,
    output: &mut [f32],
) where
    T: Copy + ToF32,
    I: Clone + Iterator<Item = i64>,
{
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    let contiguous = data.as_slice();
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

pub(crate) fn fill_dmtime_packed<T: Copy + ToF32>(
    data: &[T],
    (nt, nf): (usize, usize),
    dm_values: ArrayView1<'_, f64>,
    frequency_term: ArrayView1<'_, f64>,
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

    let terms = frequency_term;
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

pub(crate) fn fill_dedisperse<T: Copy + ToF32>(
    data: ArrayView2<'_, T>,
    delays: ArrayView1<'_, i64>,
    start: usize,
    output: &mut [f32],
) -> Result<(), TryReserveError> {
    let shape = data.shape();
    let nt = shape[0];
    let nf = shape[1];
    if nt == 0 || nf == 0 || output.is_empty() {
        return Ok(());
    }

    let mut sources = Vec::new();
    sources.try_reserve_exact(nf)?;
    for (channel, &delay) in delays.iter().enumerate() {
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

    let contiguous = data.as_slice();
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

pub(crate) fn numpy_i64(value: f64) -> i64 {
    const I64_MIN_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_F64: f64 = 9_223_372_036_854_775_808.0;

    if value.is_finite() && (I64_MIN_F64..I64_MAX_F64).contains(&value) {
        value as i64
    } else {
        i64::MIN
    }
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

pub(crate) fn fill_rfi_stats(
    values: &[u8],
    channels: usize,
    s1: &mut [f64],
    s2: &mut [f64],
) -> (u64, u64, f64) {
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
    let variance = if count == 0 {
        0.0
    } else {
        pairwise_squared_deviations(values, total as f64 / count as f64)
    };
    (total, count, variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, Axis};

    #[test]
    fn shifts_preserve_channel_order_and_strides() {
        let data = array![[1_u8, 2], [3, 4], [5, 6]];
        let mut sums = [0.0; 3];
        fill_dedispersed(data.view(), [1, -1].into_iter(), 0, &mut sums);
        assert_eq!(sums, [9.0, 7.0, 5.0]);
        sums.fill(0.0);
        let mut reversed = data.view();
        reversed.invert_axis(Axis(0));
        fill_dedispersed(reversed, [1, -1].into_iter(), 0, &mut sums);
        assert_eq!(sums, [5.0, 7.0, 9.0]);
        let mut cropped = [0.0; 2];
        fill_dedispersed(
            data.view(),
            [i64::MIN, i64::MAX].into_iter(),
            1,
            &mut cropped,
        );
        assert_eq!(cropped, [7.0, 11.0]);
    }

    #[test]
    fn conversion_and_masked_statistics() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2_f64.powi(63)] {
            assert_eq!(numpy_i64(value), i64::MIN);
        }
        assert_eq!(numpy_i64(-1.5), -1);
        assert_eq!(numpy_i64(1.5), 1);
        let mut s1 = [0.0; 2];
        let mut s2 = [0.0; 2];
        let (total, count, variance) = fill_rfi_stats(&[0, 2, 4, 0], 2, &mut s1, &mut s2);
        assert_eq!((total, count, variance), (6, 2, 2.0));
        assert_eq!(s1, [4.0, 2.0]);
        assert_eq!(s2, [16.0, 4.0]);
    }
}
