//! Decimated, cropped frequency-time and DM-time planes for the CPU candmaker.
//! Computation only; Python validation stays in lib.rs.
//!
//! Both planes keep `ncols` columns of `tdf` samples from column `col0`, and
//! each channel is rolled so that output sample `t` takes input sample
//! `(t - shift) % nt`. Two ways to sum:
//!
//! - exact: integer sums through each channel's running sum, one difference
//!   per column whatever `tdf`, divided once;
//! - upstream rounding: the float32 per-sample rows the full-array path makes,
//!   averaged over each column in NumPy's float32 order, so the planes match
//!   the dedisperse/dmtime/decimate route bit for bit.

use std::thread;

/// Integer samples summed exactly in `i64`.
pub(crate) trait Sample: Copy + Send + Sync + Into<i64> + Into<f64> {
    fn to_f32(self) -> f32;
}

macro_rules! impl_sample {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Sample for $ty {
                #[inline]
                fn to_f32(self) -> f32 {
                    self as f32
                }
            }
        )+
    };
}

impl_sample!(u8, u16, i16, i32);

/// What to compute; shifts are already reduced into `[0, nt)`.
pub(crate) struct Crop<'a> {
    pub nt: usize,
    pub nf: usize,
    /// `(dmsteps, nf)` DM-time shifts
    pub dmt_shifts: &'a [usize],
    pub dmsteps: usize,
    /// `(nf,)` frequency-time shifts
    pub ft_shifts: &'a [usize],
    pub tdf: usize,
    pub col0: usize,
    pub ncols: usize,
    pub threads: usize,
}

// channels per transposed band, and samples per transposed tile
const BAND: usize = 128;
const TILE: usize = 64;
// samples by DMs of float32 rows summed at a time with upstream rounding
const BLOCK_SAMPLES: usize = 4096;
const DM_TILE: usize = 8;

/// `(nf, nt)` channel-major copy of `(nt, nf)` data, `threads` bands at a time.
fn channel_major<T: Sample>(data: &[T], nt: usize, nf: usize, threads: usize) -> Vec<T> {
    let mut rows = vec![data.first().copied().unwrap_or_else(|| unreachable!()); nt * nf];
    let bands: Vec<(usize, &mut [T])> = rows
        .chunks_mut(BAND * nt)
        .enumerate()
        .map(|(i, rows)| (i * BAND, rows))
        .collect();
    run_split(bands, threads, |(c0, band)| {
        let c1 = (c0 + BAND).min(nf);
        for s0 in (0..nt).step_by(TILE) {
            let s1 = (s0 + TILE).min(nt);
            for ch in c0..c1 {
                let row = &mut band[(ch - c0) * nt..(ch - c0 + 1) * nt];
                for s in s0..s1 {
                    row[s] = data[s * nf + ch];
                }
            }
        }
    });
    rows
}

/// Run `work` on every item, the items split between up to `threads` threads.
fn run_split<I: Send, F: Fn(I) + Sync>(items: Vec<I>, threads: usize, work: F) {
    let threads = threads.max(1).min(items.len().max(1));
    if threads == 1 {
        items.into_iter().for_each(&work);
        return;
    }
    let mut groups: Vec<Vec<I>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, item) in items.into_iter().enumerate() {
        groups[i % threads].push(item);
    }
    thread::scope(|scope| {
        for group in groups {
            let work = &work;
            scope.spawn(move || group.into_iter().for_each(work));
        }
    });
}

// ---------------------------------------------------------------- exact sums

/// One thread's sums: its DM-time boundary plane, and each band's
/// frequency-time columns with the band's first channel.
type ThreadSums = (Vec<i64>, Vec<(usize, Vec<i64>)>);

/// `prefix[x]` = the sum of the first `x` samples of `row`.
fn running_sum<T: Sample>(row: &[T], prefix: &mut [i64]) {
    let mut acc = 0_i64;
    prefix[0] = 0;
    for (p, &v) in prefix[1..].iter_mut().zip(row) {
        acc += Into::<i64>::into(v);
        *p = acc;
    }
}

/// Add to `dest[j]` a channel's running sum at `lo + j * tdf`, counting on
/// past the chunk end as though it repeated: `dest[j + 1] - dest[j]` is then
/// its sum over the `tdf` samples from `lo + j * tdf`, wrapped. Summed over
/// channels the differences telescope, one read per channel and column.
fn add_boundaries(prefix: &[i64], lo: usize, tdf: usize, dest: &mut [i64]) {
    let nt = prefix.len() - 1;
    let total = prefix[nt];
    let before = dest.len().min((nt - lo) / tdf + 1);
    for (j, d) in dest[..before].iter_mut().enumerate() {
        *d += prefix[lo + j * tdf];
    }
    for (j, d) in dest.iter_mut().enumerate().skip(before) {
        *d += prefix[lo + j * tdf - nt] + total;
    }
}

/// Add to `dest[j]` a channel's sum over its `tdf` samples from `lo + j * tdf`.
fn add_runs(prefix: &[i64], mut lo: usize, tdf: usize, dest: &mut [i64]) {
    let nt = prefix.len() - 1;
    let total = prefix[nt];
    let mut p = prefix[lo];
    for d in dest.iter_mut() {
        let mut hi = lo + tdf;
        let q = if hi <= nt {
            prefix[hi]
        } else {
            hi -= nt;
            prefix[hi] + total
        };
        *d += q - p;
        p = prefix[hi];
        lo = hi;
    }
}

/// First sample of column `col0` for a channel shifted by `shift`.
#[inline]
fn start(crop: &Crop<'_>, shift: usize) -> usize {
    (crop.col0 * crop.tdf + crop.nt - shift) % crop.nt
}

/// Exact planes: `(dmt, ft)`, `(dmsteps, ncols)` and `(ncols, nf)` float32.
pub(crate) fn exact<T: Sample>(data: &[T], crop: &Crop<'_>) -> (Vec<f32>, Vec<f32>) {
    let (nt, nf, ncols, dmsteps) = (crop.nt, crop.nf, crop.ncols, crop.dmsteps);
    // each thread takes bands of channels, transposes one band at a time into
    // its own buffer, and keeps one channel's running sum in cache while every
    // DM step and the frequency-time column read it
    let bands: Vec<usize> = (0..nf).step_by(BAND).collect();
    let threads = crop.threads.max(1).min(bands.len());
    let mut groups: Vec<Vec<usize>> = (0..threads).map(|_| Vec::new()).collect();
    for (i, c0) in bands.into_iter().enumerate() {
        groups[i % threads].push(c0);
    }
    let per_thread: Vec<ThreadSums> = thread::scope(|scope| {
        let handles: Vec<_> = groups
            .into_iter()
            .map(|group| {
                scope.spawn(move || {
                    let mut dmt = vec![0_i64; dmsteps * (ncols + 1)];
                    let mut ft = Vec::with_capacity(group.len());
                    let mut band = vec![data[0]; BAND * nt];
                    let mut prefix = vec![0_i64; nt + 1];
                    for c0 in group {
                        let c1 = (c0 + BAND).min(nf);
                        for s0 in (0..nt).step_by(TILE) {
                            let s1 = (s0 + TILE).min(nt);
                            for ch in c0..c1 {
                                let row = &mut band[(ch - c0) * nt..(ch - c0 + 1) * nt];
                                for s in s0..s1 {
                                    row[s] = data[s * nf + ch];
                                }
                            }
                        }
                        let mut ft_band = vec![0_i64; (c1 - c0) * ncols];
                        for ch in c0..c1 {
                            running_sum(&band[(ch - c0) * nt..(ch - c0 + 1) * nt], &mut prefix);
                            for kk in 0..dmsteps {
                                let lo = start(crop, crop.dmt_shifts[kk * nf + ch]);
                                let dest = &mut dmt[kk * (ncols + 1)..(kk + 1) * (ncols + 1)];
                                add_boundaries(&prefix, lo, crop.tdf, dest);
                            }
                            let lo = start(crop, crop.ft_shifts[ch]);
                            let dest = &mut ft_band[(ch - c0) * ncols..(ch - c0 + 1) * ncols];
                            add_runs(&prefix, lo, crop.tdf, dest);
                        }
                        ft.push((c0, ft_band));
                    }
                    (dmt, ft)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let scale = crop.tdf as f64;
    let mut dmt = vec![0_f32; dmsteps * ncols];
    for kk in 0..dmsteps {
        for j in 0..ncols {
            let mut sum = 0_i64;
            for (part, _) in &per_thread {
                let row = &part[kk * (ncols + 1)..];
                sum += row[j + 1] - row[j];
            }
            dmt[kk * ncols + j] = (sum as f64 / scale) as f32;
        }
    }
    let mut ft = vec![0_f32; ncols * nf];
    for (_, bands) in &per_thread {
        for (c0, sums) in bands {
            for (g, channel) in sums.chunks(ncols).enumerate() {
                for (j, &sum) in channel.iter().enumerate() {
                    ft[j * nf + c0 + g] = (sum as f64 / scale) as f32;
                }
            }
        }
    }
    (dmt, ft)
}

// ---------------------------------------------------------- upstream rounding

/// NumPy's float32 pairwise sum (`pairwise_sum` in its loops): sequential
/// below 8 values, eight accumulators up to 128, halves (rounded down to a
/// multiple of 8) above.
fn numpy_pairwise(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 8 {
        let mut sum = -0.0_f32;
        for &v in values {
            sum += v;
        }
        sum
    } else if n <= 128 {
        let mut r = [0_f32; 8];
        r.copy_from_slice(&values[..8]);
        let full = n - n % 8;
        let mut i = 8;
        while i < full {
            for k in 0..8 {
                r[k] += values[i + k];
            }
            i += 8;
        }
        let mut sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &values[full..] {
            sum += v;
        }
        sum
    } else {
        let mut half = n / 2;
        half -= half % 8;
        numpy_pairwise(&values[..half]) + numpy_pairwise(&values[half..])
    }
}

/// `values.mean()` for a contiguous float32 run, as NumPy reduces it: the
/// pairwise sum of all of it, divided in float32.
fn numpy_mean_contiguous(values: &[f32]) -> f32 {
    numpy_pairwise(values) / values.len() as f32
}

/// The mean over a strided axis, as NumPy reduces it: in order from the first.
fn numpy_mean_strided(values: impl Iterator<Item = f32>, n: usize) -> f32 {
    let mut values = values;
    let mut sum = values.next().unwrap_or(0.0);
    for v in values {
        sum += v;
    }
    sum / n as f32
}

/// Planes rounded as the full-array path rounds them: `(dmt, ft)`.
///
/// DM-time: each output sample is a float32 sum over channels in order, then
/// each column the float32 mean of its `tdf` samples, reduced pairwise along
/// the contiguous axis. Frequency-time: each channel's shifted samples
/// averaged in float32 in order, the reduction NumPy makes along a strided
/// axis.
pub(crate) fn upstream<T: Sample>(data: &[T], crop: &Crop<'_>) -> (Vec<f32>, Vec<f32>) {
    let (nt, nf, ncols, dmsteps, tdf) = (crop.nt, crop.nf, crop.ncols, crop.dmsteps, crop.tdf);
    let rows = channel_major(data, nt, nf, crop.threads);
    let rows = &rows;

    // DM-time, in blocks of columns and tiles of DMs
    // blocks of about 4096 samples by 8 DMs keep a block's float32 rows in a
    // core's cache while every channel adds into them
    let block_cols = (BLOCK_SAMPLES / tdf).max(1);
    let dm_tile = DM_TILE.min(dmsteps.max(1));
    let mut dmt = vec![0_f32; dmsteps * ncols];
    let mut items = Vec::new();
    for j0 in (0..ncols).step_by(block_cols) {
        for k0 in (0..dmsteps).step_by(dm_tile) {
            items.push((j0, k0));
        }
    }
    let results = std::sync::Mutex::new(Vec::with_capacity(items.len()));
    run_split(items, crop.threads, |(j0, k0)| {
        let j1 = (j0 + block_cols).min(ncols);
        let k1 = (k0 + dm_tile).min(dmsteps);
        let len = (j1 - j0) * tdf;
        let mut acc = vec![0_f32; (k1 - k0) * len];
        for ch in 0..nf {
            let row = &rows[ch * nt..(ch + 1) * nt];
            for kk in k0..k1 {
                let from = (start(crop, crop.dmt_shifts[kk * nf + ch]) + j0 * tdf) % nt;
                let out = &mut acc[(kk - k0) * len..(kk - k0 + 1) * len];
                let first = len.min(nt - from);
                for (o, &v) in out[..first].iter_mut().zip(&row[from..]) {
                    *o += v.to_f32();
                }
                let mut done = first;
                while done < len {
                    let take = (len - done).min(nt);
                    for (o, &v) in out[done..done + take].iter_mut().zip(row) {
                        *o += v.to_f32();
                    }
                    done += take;
                }
            }
        }
        let mut means = Vec::with_capacity((k1 - k0) * (j1 - j0));
        for kk in 0..k1 - k0 {
            for j in 0..j1 - j0 {
                let samples = &acc[kk * len + j * tdf..kk * len + (j + 1) * tdf];
                means.push(if tdf == 1 {
                    samples[0]
                } else {
                    numpy_mean_contiguous(samples)
                });
            }
        }
        results.lock().unwrap().push((j0, j1, k0, k1, means));
    });
    for (j0, j1, k0, k1, means) in results.into_inner().unwrap() {
        for kk in k0..k1 {
            let m = &means[(kk - k0) * (j1 - j0)..(kk - k0 + 1) * (j1 - j0)];
            dmt[kk * ncols + j0..kk * ncols + j1].copy_from_slice(m);
        }
    }

    // frequency-time
    let mut ft = vec![0_f32; ncols * nf];
    let channels: Vec<usize> = (0..nf).collect();
    let columns = std::sync::Mutex::new(Vec::with_capacity(nf));
    run_split(channels, crop.threads, |ch| {
        let row = &rows[ch * nt..(ch + 1) * nt];
        let first = start(crop, crop.ft_shifts[ch]);
        let column: Vec<f32> = (0..ncols)
            .map(|j| {
                let base = first + j * tdf;
                let samples = (0..tdf).map(|s| row[(base + s) % nt].to_f32());
                if tdf == 1 {
                    row[base % nt].to_f32()
                } else {
                    numpy_mean_strided(samples, tdf)
                }
            })
            .collect();
        columns.lock().unwrap().push((ch, column));
    });
    for (ch, column) in columns.into_inner().unwrap() {
        for (j, v) in column.into_iter().enumerate() {
            ft[j * nf + ch] = v;
        }
    }
    (dmt, ft)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(data: &[u8], crop: &Crop<'_>) -> (Vec<f64>, Vec<f64>) {
        let (nt, nf, tdf) = (crop.nt, crop.nf, crop.tdf);
        let value = |ch: usize, t: usize, shift: usize| {
            f64::from(data[((t + nt * 4 - shift) % nt) * nf + ch])
        };
        let mut dmt = vec![0.0; crop.dmsteps * crop.ncols];
        let mut ft = vec![0.0; crop.ncols * nf];
        for j in 0..crop.ncols {
            for s in 0..tdf {
                let t = (crop.col0 + j) * tdf + s;
                for ch in 0..nf {
                    for kk in 0..crop.dmsteps {
                        dmt[kk * crop.ncols + j] += value(ch, t, crop.dmt_shifts[kk * nf + ch]);
                    }
                    ft[j * nf + ch] += value(ch, t, crop.ft_shifts[ch]);
                }
            }
        }
        (
            dmt.into_iter().map(|v| v / tdf as f64).collect(),
            ft.into_iter().map(|v| v / tdf as f64).collect(),
        )
    }

    #[test]
    fn exact_and_upstream_match_small_sums() {
        let (nt, nf) = (37, 5);
        let data: Vec<u8> = (0..nt * nf).map(|i| (i * 37 % 251) as u8).collect();
        let dmt_shifts: Vec<usize> = (0..3 * nf).map(|i| (i * 11) % nt).collect();
        let ft_shifts: Vec<usize> = (0..nf).map(|i| (i * 7 + 30) % nt).collect();
        for (tdf, col0, ncols, threads) in [(1, 0, 37, 1), (4, 2, 7, 2), (5, 1, 7, 3), (9, 0, 4, 2)]
        {
            let crop = Crop {
                nt,
                nf,
                dmt_shifts: &dmt_shifts,
                dmsteps: 3,
                ft_shifts: &ft_shifts,
                tdf,
                col0,
                ncols,
                threads,
            };
            let (dmt, ft) = reference(&data, &crop);
            let exact = exact(&data, &crop);
            let upstream = upstream(&data, &crop);
            for (got, want) in [
                (&exact.0, &dmt),
                (&exact.1, &ft),
                (&upstream.0, &dmt),
                (&upstream.1, &ft),
            ] {
                for (g, w) in got.iter().zip(want) {
                    assert_eq!(*g, *w as f32, "tdf {tdf} col0 {col0} ncols {ncols}");
                }
            }
        }
    }

    #[test]
    fn numpy_pairwise_splits_like_numpy() {
        // an 8-accumulator block and a split, on values whose sum rounds
        let values: Vec<f32> = (0..300).map(|i| 16_777_216.0 + (i % 3) as f32).collect();
        let half = 144;
        assert_eq!(
            numpy_pairwise(&values),
            numpy_pairwise(&values[..half]) + numpy_pairwise(&values[half..])
        );
        assert_eq!(numpy_pairwise(&[1.0, 2.0, 3.0]), 6.0);
    }
}
