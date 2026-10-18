use numpy::ndarray::{ArrayView1, ArrayView2};
use std::thread;

const BAND: usize = 128;
const TILE: usize = 64;
const U8_TILE: usize = 8;
const FLOAT_BAND: usize = 8;
const CHANNEL_BLOCK: usize = 8;
const TABLE_WORKER_LIMIT: usize = 32 * 1024 * 1024;
const TABLE_TOTAL_LIMIT: usize = 128 * 1024 * 1024;

pub(crate) trait CandidateSample: Copy + Send + Sync {
    const INTEGER_MAX_ABS: Option<u64> = None;

    fn to_f64(self) -> f64;

    fn to_i64(self) -> Option<i64> {
        None
    }

    fn transpose_band(data: ArrayView2<'_, Self>, (c0, c1): (usize, usize), band: &mut [Self]) {
        transpose_band_generic(data, (c0, c1), band);
    }
}

macro_rules! impl_candidate_sample {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CandidateSample for $ty {
                #[inline]
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )+
    };
}

macro_rules! impl_integer_candidate_sample {
    ($($ty:ty => $max_abs:expr),+ $(,)?) => {
        $(
            impl CandidateSample for $ty {
                const INTEGER_MAX_ABS: Option<u64> = Some($max_abs);

                #[inline]
                fn to_f64(self) -> f64 {
                    self as f64
                }

                #[inline]
                fn to_i64(self) -> Option<i64> {
                    Some(self as i64)
                }
            }
        )+
    };
}

impl_candidate_sample!(u32, u64, i8, i64, f64);
impl_integer_candidate_sample!(
    u16 => u16::MAX as u64,
    i16 => i16::MAX as u64 + 1,
    i32 => i32::MAX as u64 + 1,
);

impl CandidateSample for bool {
    #[inline]
    fn to_f64(self) -> f64 {
        if self {
            1.0
        } else {
            0.0
        }
    }
}

impl CandidateSample for u8 {
    const INTEGER_MAX_ABS: Option<u64> = Some(u8::MAX as u64);

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn to_i64(self) -> Option<i64> {
        Some(self as i64)
    }

    fn transpose_band(data: ArrayView2<'_, Self>, (c0, c1): (usize, usize), band: &mut [Self]) {
        let (nt, nf) = (data.shape()[0], data.shape()[1]);
        if data.is_standard_layout() {
            if let Some(values) = data.as_slice() {
                transpose_u8_band(values, nf, nt, (c0, c1), band);
                return;
            }
        }
        transpose_band_generic(data, (c0, c1), band);
    }
}

fn transpose_band_generic<T: Copy>(
    data: ArrayView2<'_, T>,
    (c0, c1): (usize, usize),
    band: &mut [T],
) {
    let nt = data.shape()[0];
    for s0 in (0..nt).step_by(TILE) {
        let s1 = (s0 + TILE).min(nt);
        for ch in c0..c1 {
            let row = &mut band[(ch - c0) * nt..(ch - c0 + 1) * nt];
            for s in s0..s1 {
                row[s] = data[(s, ch)];
            }
        }
    }
}

fn transpose_u8_band(data: &[u8], nf: usize, nt: usize, (c0, c1): (usize, usize), band: &mut [u8]) {
    let width = c1 - c0;
    let (full_s, full_c) = (nt - nt % U8_TILE, width - width % U8_TILE);
    for s0 in (0..full_s).step_by(TILE) {
        let s1 = (s0 + TILE).min(full_s);
        for cb in (0..full_c).step_by(U8_TILE) {
            for s in (s0..s1).step_by(U8_TILE) {
                let mut rows = [0_u64; U8_TILE];
                for (i, row) in rows.iter_mut().enumerate() {
                    let at = (s + i) * nf + c0 + cb;
                    let mut bytes = [0_u8; U8_TILE];
                    bytes.copy_from_slice(&data[at..at + U8_TILE]);
                    *row = u64::from_le_bytes(bytes);
                }
                transpose8(&mut rows);
                for (j, row) in rows.iter().enumerate() {
                    let at = (cb + j) * nt + s;
                    band[at..at + U8_TILE].copy_from_slice(&row.to_le_bytes());
                }
            }
        }
    }
    for ch in 0..width {
        let rest = if ch < full_c { full_s } else { 0 };
        for s in rest..nt {
            band[ch * nt + s] = data[s * nf + c0 + ch];
        }
    }
}

fn transpose8(rows: &mut [u64; U8_TILE]) {
    for (step, mask) in [
        (1, 0x00FF_00FF_00FF_00FF_u64),
        (2, 0x0000_FFFF_0000_FFFF),
        (4, 0x0000_0000_FFFF_FFFF),
    ] {
        let shift = 8 * step as u32;
        for a in (0..U8_TILE).filter(|a| a & step == 0) {
            let b = a + step;
            let t = ((rows[a] >> shift) ^ rows[b]) & mask;
            rows[b] ^= t;
            rows[a] ^= t << shift;
        }
    }
}

fn allocated<T>(len: usize) -> Result<Vec<T>, ()> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| ())?;
    Ok(values)
}

fn filled<T: Copy>(len: usize, value: T) -> Result<Vec<T>, ()> {
    let mut values = allocated(len)?;
    values.resize(len, value);
    Ok(values)
}

fn boundary_table_geometry(
    nt: usize,
    time_bin: usize,
    dmsteps: usize,
    boundary_width: usize,
    threads: usize,
) -> Option<(usize, usize)> {
    if time_bin == 0 {
        return None;
    }
    let row_len = (nt / time_bin).checked_add(1)?;
    let table_len = time_bin.checked_mul(row_len)?;
    let table_bytes = table_len
        .checked_mul(CHANNEL_BLOCK)?
        .checked_mul(std::mem::size_of::<i64>())?;
    if table_bytes > TABLE_WORKER_LIMIT
        || table_bytes.checked_mul(threads)? > TABLE_TOTAL_LIMIT
        || 2 * table_len as u128 > dmsteps as u128 * boundary_width as u128
    {
        return None;
    }
    Some((row_len, table_len))
}

fn boundary_table(prefix: &[i64], time_bin: usize, row_len: usize, table: &mut [i64]) {
    let nt = prefix.len() - 1;
    for (residue, row) in table.chunks_exact_mut(row_len).enumerate() {
        for (multiple, value) in row.iter_mut().enumerate() {
            let Some(at) = multiple
                .checked_mul(time_bin)
                .and_then(|at| at.checked_add(residue))
            else {
                break;
            };
            if at > nt {
                break;
            }
            *value = prefix[at];
        }
    }
}

fn integer_accum_safe<T: CandidateSample>(nt: usize, nf: usize) -> bool {
    let Some(max_abs) = T::INTEGER_MAX_ABS else {
        return false;
    };
    let Some(bound) = (nt as u128)
        .checked_mul(nf as u128)
        .and_then(|bound| bound.checked_mul(max_abs as u128))
    else {
        return false;
    };
    // Boundary subtraction can combine two prefixes of opposite signs.
    bound <= i64::MAX as u128 / 2
}

#[allow(clippy::too_many_arguments)]
/// Build nonwrapping candidate planes from explicit nonnegative delay offsets.
/// Core integer dtypes use exact i64 prefixes; other dtypes retain the f64 path.
pub(crate) fn binned_products<T: CandidateSample>(
    data: ArrayView2<'_, T>,
    dm_delays: ArrayView2<'_, i64>,
    ft_delays: ArrayView1<'_, i64>,
    enabled: Option<ArrayView1<'_, bool>>,
    start: usize,
    time_bin: usize,
    time_size: usize,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>), ()> {
    let (nt, nf) = (data.shape()[0], data.shape()[1]);
    if integer_accum_safe::<T>(nt, nf) {
        binned_products_i64(
            data, dm_delays, ft_delays, enabled, start, time_bin, time_size, threads,
        )
    } else {
        binned_products_f64(
            data, dm_delays, ft_delays, enabled, start, time_bin, time_size,
        )
    }
}

struct ThreadSums {
    dmt_boundaries: Vec<i64>,
    ft_bands: Vec<(usize, Vec<f32>)>,
}

#[allow(clippy::too_many_arguments)]
fn binned_products_i64<T: CandidateSample>(
    data: ArrayView2<'_, T>,
    dm_delays: ArrayView2<'_, i64>,
    ft_delays: ArrayView1<'_, i64>,
    enabled: Option<ArrayView1<'_, bool>>,
    start: usize,
    time_bin: usize,
    time_size: usize,
    threads: usize,
) -> Result<(Vec<f32>, Vec<f32>), ()> {
    let nf = data.shape()[1];
    let dmsteps = dm_delays.shape()[0];
    let boundary_width = time_size.checked_add(1).ok_or(())?;
    let boundary_len = dmsteps.checked_mul(boundary_width).ok_or(())?;
    let band_count = nf.div_ceil(BAND);
    let thread_count = threads.max(1).min(band_count);
    let table_geometry = boundary_table_geometry(
        data.shape()[0],
        time_bin,
        dmsteps,
        boundary_width,
        thread_count,
    );

    let mut parts = if thread_count == 1 {
        let mut parts = allocated(1)?;
        parts.push(binned_products_i64_worker(
            data,
            dm_delays,
            ft_delays,
            enabled,
            start,
            time_bin,
            time_size,
            boundary_width,
            boundary_len,
            table_geometry,
            band_count,
            0,
            1,
        )?);
        parts
    } else {
        thread::scope(|scope| {
            let mut handles = allocated(thread_count)?;
            for worker in 0..thread_count {
                handles.push(scope.spawn(move || {
                    binned_products_i64_worker(
                        data,
                        dm_delays,
                        ft_delays,
                        enabled,
                        start,
                        time_bin,
                        time_size,
                        boundary_width,
                        boundary_len,
                        table_geometry,
                        band_count,
                        worker,
                        thread_count,
                    )
                }));
            }
            let mut parts = allocated(thread_count)?;
            for handle in handles {
                parts.push(handle.join().expect("candidate worker panicked")?);
            }
            Ok::<_, ()>(parts)
        })?
    };

    let first = parts.remove(0);
    let mut dmt_boundaries = first.dmt_boundaries;
    let mut ft = filled(nf.checked_mul(time_size).ok_or(())?, 0.0_f32)?;
    for (c0, band) in first.ft_bands {
        write_ft_band(&mut ft, nf, time_size, c0, &band);
    }
    for part in parts {
        for (sum, value) in dmt_boundaries.iter_mut().zip(part.dmt_boundaries) {
            *sum += value;
        }
        for (c0, band) in part.ft_bands {
            write_ft_band(&mut ft, nf, time_size, c0, &band);
        }
    }
    let divisor = (nf * time_bin) as f64;
    let dmt_len = dmsteps.checked_mul(time_size).ok_or(())?;
    let mut dmt = filled(dmt_len, 0.0_f32)?;
    for dm in 0..dmsteps {
        let boundaries = &dmt_boundaries[dm * boundary_width..(dm + 1) * boundary_width];
        for j in 0..time_size {
            dmt[dm * time_size + j] = ((boundaries[j + 1] - boundaries[j]) as f64 / divisor) as f32;
        }
    }
    Ok((ft, dmt))
}

#[allow(clippy::too_many_arguments)]
fn binned_products_i64_worker<T: CandidateSample>(
    data: ArrayView2<'_, T>,
    dm_delays: ArrayView2<'_, i64>,
    ft_delays: ArrayView1<'_, i64>,
    enabled: Option<ArrayView1<'_, bool>>,
    start: usize,
    time_bin: usize,
    time_size: usize,
    boundary_width: usize,
    boundary_len: usize,
    table_geometry: Option<(usize, usize)>,
    band_count: usize,
    worker: usize,
    worker_count: usize,
) -> Result<ThreadSums, ()> {
    let (nt, nf) = (data.shape()[0], data.shape()[1]);
    let band_channels = BAND.min(nf);
    let band_len = band_channels.checked_mul(nt).ok_or(())?;
    let mut band = filled(band_len, data[(0, 0)])?;
    let mut prefix = filled(nt.checked_add(1).ok_or(())?, 0_i64)?;
    let mut dmt_boundaries = filled(boundary_len, 0_i64)?;
    let owned_bands = (band_count - 1 - worker) / worker_count + 1;
    let mut ft_bands = allocated(owned_bands)?;
    let mut tables = Vec::new();
    let table_len = table_geometry.map_or(0, |(_, len)| len);
    let use_tables = table_geometry.is_some_and(|_| {
        table_len
            .checked_mul(CHANNEL_BLOCK)
            .is_some_and(|len| tables.try_reserve_exact(len).is_ok())
    });
    if use_tables {
        tables.resize(table_len * CHANNEL_BLOCK, 0_i64);
    }
    let channel_block = if use_tables { CHANNEL_BLOCK } else { 1 };

    for band_index in (worker..band_count).step_by(worker_count) {
        let c0 = band_index * BAND;
        let c1 = (c0 + BAND).min(nf);
        let width = c1 - c0;
        T::transpose_band(data, (c0, c1), &mut band[..width * nt]);
        let mut ft_band = filled(width.checked_mul(time_size).ok_or(())?, 0.0_f32)?;

        for group_start in (c0..c1).step_by(channel_block) {
            let group_end = (group_start + channel_block).min(c1);
            for ch in group_start..group_end {
                if enabled.is_some_and(|mask| !mask[ch]) {
                    continue;
                }
                let row = &band[(ch - c0) * nt..(ch - c0 + 1) * nt];
                let mut sum = 0_i64;
                prefix[0] = 0;
                for (out, &value) in prefix[1..].iter_mut().zip(row) {
                    sum += value
                        .to_i64()
                        .expect("integer path requires an integer sample");
                    *out = sum;
                }

                let ft_lo = start + ft_delays[ch] as usize;
                for j in 0..time_size {
                    let lo = ft_lo + j * time_bin;
                    let run = prefix[lo + time_bin] - prefix[lo];
                    ft_band[(ch - c0) * time_size + j] = (run as f64 / time_bin as f64) as f32;
                }

                if use_tables {
                    let (row_len, table_len) = table_geometry.expect("table geometry was enabled");
                    let table_start = (ch - group_start) * table_len;
                    boundary_table(
                        &prefix,
                        time_bin,
                        row_len,
                        &mut tables[table_start..table_start + table_len],
                    );
                } else {
                    for dm in 0..dm_delays.shape()[0] {
                        let lo = start + dm_delays[(dm, ch)] as usize;
                        let boundaries =
                            &mut dmt_boundaries[dm * boundary_width..(dm + 1) * boundary_width];
                        for (j, boundary) in boundaries.iter_mut().enumerate() {
                            *boundary += prefix[lo + j * time_bin];
                        }
                    }
                }
            }

            if use_tables {
                let (row_len, table_len) = table_geometry.expect("table geometry was enabled");
                for dm in 0..dm_delays.shape()[0] {
                    let boundaries =
                        &mut dmt_boundaries[dm * boundary_width..(dm + 1) * boundary_width];
                    for ch in group_start..group_end {
                        if enabled.is_some_and(|mask| !mask[ch]) {
                            continue;
                        }
                        let lo = start + dm_delays[(dm, ch)] as usize;
                        let table_start = (ch - group_start) * table_len;
                        let table = &tables[table_start..table_start + table_len];
                        let first = (lo % time_bin) * row_len + lo / time_bin;
                        let run = &table[first..first + boundary_width];
                        for (boundary, &value) in boundaries.iter_mut().zip(run) {
                            *boundary += value;
                        }
                    }
                }
            }
        }
        ft_bands.push((c0, ft_band));
    }

    Ok(ThreadSums {
        dmt_boundaries,
        ft_bands,
    })
}

fn write_ft_band(ft: &mut [f32], nf: usize, time_size: usize, c0: usize, band: &[f32]) {
    let width = band.len() / time_size;
    for ch in 0..width {
        for j in 0..time_size {
            ft[j * nf + c0 + ch] = band[ch * time_size + j];
        }
    }
}

fn binned_products_f64<T: CandidateSample>(
    data: ArrayView2<'_, T>,
    dm_delays: ArrayView2<'_, i64>,
    ft_delays: ArrayView1<'_, i64>,
    enabled: Option<ArrayView1<'_, bool>>,
    start: usize,
    time_bin: usize,
    time_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), ()> {
    let (nt, nf) = (data.shape()[0], data.shape()[1]);
    let dmsteps = dm_delays.shape()[0];
    let band_channels = FLOAT_BAND.min(nf);
    let band_len = band_channels.checked_mul(nt).ok_or(())?;
    let mut band = filled(band_len, 0.0_f64)?;
    let mut prefix = filled(nt.checked_add(1).ok_or(())?, 0.0_f64)?;
    let dmt_len = dmsteps.checked_mul(time_size).ok_or(())?;
    let mut dmt_sums = filled(dmt_len, 0.0_f64)?;
    let ft_len = nf.checked_mul(time_size).ok_or(())?;
    let mut ft = filled(ft_len, 0.0_f32)?;

    for ch0 in (0..nf).step_by(FLOAT_BAND) {
        let width = (nf - ch0).min(FLOAT_BAND);
        for t in 0..nt {
            for local_ch in 0..width {
                let ch = ch0 + local_ch;
                band[local_ch * nt + t] = if enabled.is_none_or(|mask| mask[ch]) {
                    data[(t, ch)].to_f64()
                } else {
                    0.0
                };
            }
        }

        for local_ch in 0..width {
            let ch = ch0 + local_ch;
            let row = &band[local_ch * nt..(local_ch + 1) * nt];
            prefix[0] = 0.0;
            for t in 0..nt {
                prefix[t + 1] = prefix[t] + row[t];
            }

            let ft_delay = ft_delays[ch] as usize;
            for j in 0..time_size {
                let lo = start + ft_delay + j * time_bin;
                ft[j * nf + ch] = ((prefix[lo + time_bin] - prefix[lo]) / time_bin as f64) as f32;
            }

            for dm in 0..dmsteps {
                let delay = dm_delays[(dm, ch)] as usize;
                let row = &mut dmt_sums[dm * time_size..(dm + 1) * time_size];
                for (j, sum) in row.iter_mut().enumerate() {
                    let lo = start + delay + j * time_bin;
                    *sum += prefix[lo + time_bin] - prefix[lo];
                }
            }
        }
    }

    let divisor = (nf * time_bin) as f64;
    let mut dmt = filled(dmt_len, 0.0_f32)?;
    for (out, sum) in dmt.iter_mut().zip(dmt_sums) {
        *out = (sum / divisor) as f32;
    }
    Ok((ft, dmt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{Array1, Array2};

    #[test]
    fn uses_explicit_right_hand_delays_and_preserves_the_enabled_mask() {
        let data = Array2::from_shape_vec(
            (7, 2),
            vec![0_u8, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61],
        )
        .unwrap();
        let dm_delays = Array2::from_shape_vec((1, 2), vec![0_i64, 1]).unwrap();
        let ft_delays = Array1::from_vec(vec![1_i64, 0]);
        let enabled = Array1::from_vec(vec![true, false]);
        let (ft, dmt) = binned_products(
            data.view(),
            dm_delays.view(),
            ft_delays.view(),
            Some(enabled.view()),
            1,
            2,
            2,
            1,
        )
        .unwrap();
        assert_eq!(ft, [25.0, 0.0, 45.0, 0.0]);
        assert_eq!(dmt, [7.5, 17.5]);
    }

    #[test]
    fn tiled_transpose_handles_sample_and_channel_tails_with_multiple_threads() {
        let (nt, nf) = (73, 131);
        let data = Array2::from_shape_vec(
            (nt, nf),
            (0..nt * nf).map(|i| (i * 17 % 251) as u8).collect(),
        )
        .unwrap();
        let dm_delays = Array2::from_shape_fn((64, nf), |(dm, ch)| ((dm + ch) % 4) as i64);
        let ft_delays = Array1::from_shape_fn(nf, |ch| (ch % 4) as i64);
        assert!(boundary_table_geometry(nt, 3, 64, 5, 2).is_some());
        let one = binned_products(
            data.view(),
            dm_delays.view(),
            ft_delays.view(),
            None,
            2,
            3,
            4,
            1,
        )
        .unwrap();
        let many = binned_products(
            data.view(),
            dm_delays.view(),
            ft_delays.view(),
            None,
            2,
            3,
            4,
            8,
        )
        .unwrap();
        assert_eq!(one, many);
    }

    #[test]
    fn integer_path_rejects_prefix_bounds_that_can_overflow_i64() {
        assert!(!integer_accum_safe::<i32>(usize::MAX, usize::MAX));
        assert!(integer_accum_safe::<i32>(10, 4));
    }
}
