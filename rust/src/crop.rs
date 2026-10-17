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

use crate::kernels::numpy_i64;

/// Integer samples summed exactly in `i64`.
pub(crate) trait Sample: Copy + Send + Sync + Into<i64> + Into<f64> {
    fn to_f32(self) -> f32;

    /// Channels `c0..c1` of `(nt, nf)` data, channel-major into `band`
    /// (`band[(ch - c0) * nt + s]`), a tile of samples at a time.
    fn transpose_band(
        data: &[Self],
        nf: usize,
        nt: usize,
        (c0, c1): (usize, usize),
        band: &mut [Self],
    ) {
        for s0 in (0..nt).step_by(TILE) {
            let s1 = (s0 + TILE).min(nt);
            for ch in c0..c1 {
                let row = &mut band[(ch - c0) * nt..(ch - c0 + 1) * nt];
                for s in s0..s1 {
                    row[s] = data[s * nf + ch];
                }
            }
        }
    }
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

impl_sample!(u16, i16, i32);

impl Sample for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    /// As the default, but 8 samples by 8 channels at a time: a `u64` load
    /// from each of 8 samples, transposed in registers, a `u64` store to each
    /// of 8 channels, rather than one byte at a time.
    fn transpose_band(
        data: &[u8],
        nf: usize,
        nt: usize,
        (c0, c1): (usize, usize),
        band: &mut [u8],
    ) {
        let width = c1 - c0;
        let (full_s, full_c) = (nt - nt % 8, width - width % 8);
        for s0 in (0..full_s).step_by(TILE) {
            let s1 = (s0 + TILE).min(full_s);
            // a block of channels at a time, so each channel's cache line of
            // the tile is written whole before the next
            for cb in (0..full_c).step_by(8) {
                for s in (s0..s1).step_by(8) {
                    let mut rows = [0_u64; 8];
                    for (i, row) in rows.iter_mut().enumerate() {
                        let at = (s + i) * nf + c0 + cb;
                        let mut bytes = [0_u8; 8];
                        bytes.copy_from_slice(&data[at..at + 8]);
                        *row = u64::from_le_bytes(bytes);
                    }
                    transpose8(&mut rows);
                    for (j, row) in rows.iter().enumerate() {
                        let at = (cb + j) * nt + s;
                        band[at..at + 8].copy_from_slice(&row.to_le_bytes());
                    }
                }
            }
        }
        // the channels and samples past the last whole block
        for ch in 0..width {
            let rest = if ch < full_c { full_s } else { 0 };
            for s in rest..nt {
                band[ch * nt + s] = data[s * nf + c0 + ch];
            }
        }
    }
}

/// Transpose 8 rows of 8 bytes in place, byte `j` of row `i` to byte `i` of
/// row `j`: swap single bytes between neighbouring rows, then pairs of bytes
/// two rows apart, then quads four rows apart.
fn transpose8(rows: &mut [u64; 8]) {
    // rows `step` apart swap units of `step` bytes
    for (step, mask) in [
        (1, 0x00FF_00FF_00FF_00FF_u64),
        (2, 0x0000_FFFF_0000_FFFF),
        (4, 0x0000_0000_FFFF_FFFF),
    ] {
        let shift = 8 * step as u32;
        for a in (0..8).filter(|a| a & step == 0) {
            let b = a + step;
            let t = ((rows[a] >> shift) ^ rows[b]) & mask;
            rows[b] ^= t;
            rows[a] ^= t << shift;
        }
    }
}

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
// channels whose boundary tables are added to a DM row before the next row
const CHANNEL_BLOCK: usize = 8;

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

/// A channel's shift for one DM, as NumPy makes it: `np.where(np.abs(d) <
/// nt, d % nt, 0)` of `d = np.round(kdm * term / 1000.0 / tsamp)` cast to
/// int64, where `kdm` is `4148808.0 * dm` and `term` the channel's
/// `1 / f0**2 - 1 / f**2`, every step the same float64 operation.
pub(crate) fn shift(kdm: f64, term: f64, tsamp: f64, nt: usize) -> usize {
    let delay = numpy_i64((kdm * term / 1000.0 / tsamp).round_ties_even());
    let nt = nt as i64;
    // within the chunk the wrap needs no division
    if (0..nt).contains(&delay) {
        delay as usize
    } else if (1 - nt..0).contains(&delay) {
        (delay + nt) as usize
    } else if delay == i64::MIN {
        // NumPy's abs of the int64 minimum stays negative, so it is kept
        delay.rem_euclid(nt) as usize
    } else {
        0
    }
}

/// `(kdms.len(), terms.len())` shifts, one row per DM, `threads` rows at a time.
pub(crate) fn dm_shifts(
    kdms: &[f64],
    terms: &[f64],
    tsamp: f64,
    nt: usize,
    threads: usize,
) -> Vec<usize> {
    let mut shifts = vec![0; kdms.len() * terms.len()];
    let rows: Vec<(f64, &mut [usize])> = kdms
        .iter()
        .copied()
        .zip(shifts.chunks_mut(terms.len().max(1)))
        .collect();
    run_split(rows, threads, |(kdm, row)| {
        for (s, &term) in row.iter_mut().zip(terms) {
            *s = shift(kdm, term, tsamp, nt);
        }
    });
    shifts
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

/// A channel's running sum at every `r + m * tdf`, counting on past the chunk
/// end as though it repeated, laid out by residue: `table[r * row_len + m]`.
/// The boundaries `lo + j * tdf` of the columns from any `lo` are then the
/// contiguous run from `(lo % tdf) * row_len + lo / tdf`, so adding them to
/// a DM row is a plain vector add however far apart they are in time.
fn boundary_table(prefix: &[i64], tdf: usize, row_len: usize, table: &mut [i64]) {
    let nt = prefix.len() - 1;
    let total = prefix[nt];
    for (r, row) in table.chunks_exact_mut(row_len).enumerate() {
        let (mut x, mut base) = (r, 0_i64);
        for value in row.iter_mut() {
            *value = base + prefix[x];
            x += tdf;
            while x > nt {
                x -= nt;
                base += total;
            }
        }
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

/// Whether residue tables pay for themselves. Building a channel's table
/// costs about its length; it saves strided reads of the `dmsteps * (ncols +
/// 1)` boundaries. Up to about half that, a block of tables stays in cache
/// and wins (by up to 1.4x with many channels and short chunks); on long
/// chunks the tables are tens of MB per thread, and the direct reads are up
/// to 4x faster and far less sensitive to memory contention.
fn tables_pay(table_len: usize, dmsteps: usize, ncols: usize) -> bool {
    2 * table_len <= dmsteps * (ncols + 1)
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
    let row_len = (crop.nt - 1) / crop.tdf + crop.ncols + 1;
    let use_tables = tables_pay(crop.tdf * row_len, crop.dmsteps, crop.ncols);
    exact_with(data, crop, use_tables)
}

/// [`exact`], with residue tables or with direct reads of each channel's
/// running sum; both give the same sums.
fn exact_with<T: Sample>(data: &[T], crop: &Crop<'_>, use_tables: bool) -> (Vec<f32>, Vec<f32>) {
    let (nt, nf, ncols, dmsteps, tdf) = (crop.nt, crop.nf, crop.ncols, crop.dmsteps, crop.tdf);
    // every `lo < nt` and its `ncols + 1` boundaries fall in a residue's row
    let row_len = (nt - 1) / tdf + ncols + 1;
    let table_len = tdf * row_len;
    // each thread takes bands of channels and transposes one band at a time
    // into its own buffer. With tables, a block of channels' tables is then
    // added to each DM row in turn, so a row stays in L1 for the whole block
    // rather than the thread's DM-time plane passing through cache once per
    // channel; without, each channel's running sum is read where each DM row
    // needs it.
    let block = if use_tables { CHANNEL_BLOCK } else { 1 };
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
                    let mut tables = vec![0_i64; if use_tables { block * table_len } else { 0 }];
                    for c0 in group {
                        let c1 = (c0 + BAND).min(nf);
                        T::transpose_band(data, nf, nt, (c0, c1), &mut band);
                        let mut ft_band = vec![0_i64; (c1 - c0) * ncols];
                        for g0 in (c0..c1).step_by(block) {
                            let g1 = (g0 + block).min(c1);
                            for ch in g0..g1 {
                                running_sum(&band[(ch - c0) * nt..(ch - c0 + 1) * nt], &mut prefix);
                                if use_tables {
                                    let table = &mut tables[(ch - g0) * table_len..][..table_len];
                                    boundary_table(&prefix, tdf, row_len, table);
                                } else {
                                    for (kk, dest) in dmt.chunks_exact_mut(ncols + 1).enumerate() {
                                        let lo = start(crop, crop.dmt_shifts[kk * nf + ch]);
                                        add_boundaries(&prefix, lo, tdf, dest);
                                    }
                                }
                                let lo = start(crop, crop.ft_shifts[ch]);
                                let dest = &mut ft_band[(ch - c0) * ncols..(ch - c0 + 1) * ncols];
                                add_runs(&prefix, lo, tdf, dest);
                            }
                            if !use_tables {
                                continue;
                            }
                            for (kk, dest) in dmt.chunks_exact_mut(ncols + 1).enumerate() {
                                let shifts = &crop.dmt_shifts[kk * nf..(kk + 1) * nf];
                                for (ch, table) in (g0..g1).zip(tables.chunks_exact(table_len)) {
                                    let lo = start(crop, shifts[ch]);
                                    let run =
                                        &table[(lo % tdf) * row_len + lo / tdf..][..ncols + 1];
                                    for (d, &v) in dest.iter_mut().zip(run) {
                                        *d += v;
                                    }
                                }
                            }
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
    fn exact_matches_across_channel_blocks_and_bands() {
        // two full bands and a partial one, whose last channel block is partial
        let (nt, nf, dmsteps) = (41, 2 * BAND + CHANNEL_BLOCK + 3, 4);
        let data: Vec<u8> = (0..nt * nf).map(|i| (i * 89 % 253) as u8).collect();
        let dmt_shifts: Vec<usize> = (0..dmsteps * nf).map(|i| (i * 13 + i / nf) % nt).collect();
        let ft_shifts: Vec<usize> = (0..nf).map(|i| (i * 5 + 17) % nt).collect();
        for (tdf, col0, ncols, threads) in
            [(1, 3, 38, 2), (3, 1, 12, 3), (8, 0, 5, 1), (10, 1, 3, 4)]
        {
            let crop = Crop {
                nt,
                nf,
                dmt_shifts: &dmt_shifts,
                dmsteps,
                ft_shifts: &ft_shifts,
                tdf,
                col0,
                ncols,
                threads,
            };
            let (dmt, ft) = reference(&data, &crop);
            for use_tables in [false, true] {
                let (got_dmt, got_ft) = exact_with(&data, &crop, use_tables);
                for (got, want) in [(&got_dmt, &dmt), (&got_ft, &ft)] {
                    for (g, w) in got.iter().zip(want) {
                        assert_eq!(
                            *g, *w as f32,
                            "tdf {tdf} col0 {col0} ncols {ncols} tables {use_tables}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn shifts_wrap_as_numpy_does() {
        // delays of 3, -3, 10 (the chunk length) and -12 samples
        let nt = 10;
        let (kdm, tsamp) = (1000.0, 1.0);
        assert_eq!(shift(kdm, 3.0, tsamp, nt), 3);
        assert_eq!(shift(kdm, -3.0, tsamp, nt), 7);
        assert_eq!(shift(kdm, 10.0, tsamp, nt), 0);
        assert_eq!(shift(kdm, -12.0, tsamp, nt), 0);
        // halves round to even
        assert_eq!(shift(kdm, 2.5, tsamp, nt), 2);
        assert_eq!(shift(kdm, -2.5, tsamp, nt), 8);
        // NaN casts to the int64 minimum, whose NumPy abs stays negative
        assert_eq!(
            shift(kdm, f64::NAN, tsamp, nt),
            (i64::MIN.rem_euclid(10)) as usize
        );
        let table = dm_shifts(&[1000.0, 2000.0], &[1.0, -1.0, 4.0], 1.0, nt, 3);
        assert_eq!(table, [1, 9, 4, 2, 8, 8]);
    }

    #[test]
    fn block_transpose_matches_the_scalar_one() {
        let mut rows: [u64; 8] = [0; 8];
        for (i, row) in rows.iter_mut().enumerate() {
            *row = u64::from_le_bytes(std::array::from_fn(|j| (i * 8 + j) as u8));
        }
        transpose8(&mut rows);
        for (j, row) in rows.iter().enumerate() {
            assert_eq!(
                row.to_le_bytes(),
                std::array::from_fn(|i| (i * 8 + j) as u8)
            );
        }
        // whole and partial blocks of samples and channels, and a band
        // starting part way along
        for (nt, nf, c0, c1) in [
            (64, 16, 0, 16),
            (77, 37, 3, 30),
            (130, 140, 5, 133),
            (7, 9, 0, 9),
        ] {
            let data: Vec<u8> = (0..nt * nf).map(|i| (i * 131 % 251) as u8).collect();
            let mut fast = vec![0_u8; (c1 - c0) * nt];
            let mut slow = vec![1_u8; (c1 - c0) * nt];
            <u8 as Sample>::transpose_band(&data, nf, nt, (c0, c1), &mut fast);
            for ch in c0..c1 {
                for s in 0..nt {
                    slow[(ch - c0) * nt + s] = data[s * nf + ch];
                }
            }
            assert_eq!(fast, slow, "nt {nt} nf {nf} channels {c0}..{c1}");
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
