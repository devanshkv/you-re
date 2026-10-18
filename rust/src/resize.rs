//! The candmaker's frequency resize: skimage's anti-aliased `resize` of each
//! row by an integer factor, a Gaussian filter then linear interpolation,
//! computed only at the filtered columns the interpolation reads.
//!
//! SciPy filters a line in float64 with a symmetric kernel as
//! `x[c] * w[0] + (x[c - r] + x[c + r]) * w[r] + ... + (x[c - 1] + x[c + 1]) * w[1]`,
//! zero past either end (`grid-constant`), and rounds it to float32. With
//! `grid_mode`, an even factor `f` puts output `o` halfway between filtered
//! columns `f * o + f / 2 - 1` and `f * o + f / 2`, and an odd one on column
//! `f * o + (f - 1) / 2`. The same operations in the same order give the same
//! bits; clipping to the input's range stays in Python.

// rows filtered together
const ROWS: usize = 8;

/// `out` (`nt` rows of `nf / factor`) from `data` (`nt` rows of `nf`), with
/// `weights` the half Gaussian kernel, centre first.
pub(crate) fn gaussian_decimate(
    data: &[f32],
    nf: usize,
    weights: &[f64],
    factor: usize,
    out: &mut [f32],
) -> Result<(), ()> {
    if factor == 0 || nf == 0 || nf % factor != 0 || weights.is_empty() {
        return Err(());
    }
    let size = nf / factor;
    let r = weights.len() - 1;
    let padded_columns = r
        .checked_mul(2)
        .and_then(|padding| nf.checked_add(padding))
        .ok_or(())?;
    let lines_len = padded_columns.checked_mul(ROWS).ok_or(())?;
    let reads = if factor % 2 == 0 { 2 } else { 1 };
    let column_count = size.checked_mul(reads).ok_or(())?;
    let sums_len = column_count.checked_mul(ROWS).ok_or(())?;
    let input_block_len = nf.checked_mul(ROWS).ok_or(())?;
    let output_block_len = size.checked_mul(ROWS).ok_or(())?;
    let first = (factor - 1) / 2;
    let mut columns = Vec::new();
    columns.try_reserve_exact(column_count).map_err(|_| ())?;
    for o in 0..size {
        for s in 0..reads {
            columns.push(r + factor * o + first + s);
        }
    }
    // ROWS rows at a time, interleaved: `lines[c * ROWS + i]` is row `i` at
    // padded column `c`, so each term of the filter is a contiguous vector
    // op across the rows, in the order SciPy takes for every one of them.
    let mut lines = Vec::new();
    lines.try_reserve_exact(lines_len).map_err(|_| ())?;
    lines.resize(lines_len, 0.0_f64);
    let mut sums = Vec::new();
    sums.try_reserve_exact(sums_len).map_err(|_| ())?;
    sums.resize(sums_len, 0.0_f64);

    for (block, dest) in data
        .chunks(input_block_len)
        .zip(out.chunks_mut(output_block_len))
    {
        let rows = block.len() / nf;
        let padded = &mut lines[r * ROWS..(r + nf) * ROWS];
        for (c, column) in padded.chunks_exact_mut(ROWS).enumerate() {
            for (x, row) in column.iter_mut().zip(block.chunks_exact(nf)) {
                *x = f64::from(row[c]);
            }
        }
        for (&c, sum) in columns.iter().zip(sums.chunks_exact_mut(ROWS)) {
            let mut acc = lane(&lines, c).map(|x| x * weights[0]);
            for j in (1..=r).rev() {
                let w = weights[j];
                let (before, after) = (lane(&lines, c - j), lane(&lines, c + j));
                for ((s, a), b) in acc.iter_mut().zip(before).zip(after) {
                    *s += (a + b) * w;
                }
            }
            sum.copy_from_slice(&acc);
        }
        for (i, dest) in dest.chunks_exact_mut(size).enumerate().take(rows) {
            for (o, d) in dest.iter_mut().enumerate() {
                let filtered = |s: usize| f64::from(sums[(o * reads + s) * ROWS + i] as f32);
                let mut t = 0.0_f64;
                if reads == 2 {
                    t += filtered(0) * 0.5;
                    t += filtered(1) * 0.5;
                } else {
                    t += filtered(0);
                }
                *d = t as f32;
            }
        }
    }
    Ok(())
}

/// Padded column `c` of the interleaved rows.
#[inline]
fn lane(lines: &[f64], c: usize) -> [f64; ROWS] {
    let mut values = [0.0; ROWS];
    values.copy_from_slice(&lines[c * ROWS..][..ROWS]);
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_factor_averages_the_two_middle_columns() {
        // a one-tap kernel leaves the filter as the identity
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut out = vec![0.0; 4];
        gaussian_decimate(&data, 8, &[1.0], 4, &mut out).unwrap();
        assert_eq!(out, [1.5, 5.5, 9.5, 13.5]);
    }

    #[test]
    fn odd_factor_takes_the_middle_column_and_pads_with_zero() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = vec![0.0; 2];
        // centre 0.5, a quarter either side
        gaussian_decimate(&data, 6, &[0.5, 0.25], 3, &mut out).unwrap();
        assert_eq!(out, [2.0, 5.0]);
        let mut edge = vec![0.0; 6];
        gaussian_decimate(&data, 6, &[0.5, 0.25], 1, &mut edge).unwrap();
        assert_eq!(edge[0], 1.0); // 0.5 * 1 + 0.25 * (0 + 2)
        assert_eq!(edge[5], 4.25); // 0.5 * 6 + 0.25 * (5 + 0)
    }

    #[test]
    fn zero_comes_out_positive() {
        let data = [-0.0_f32; 4];
        let mut out = vec![1.0; 2];
        gaussian_decimate(&data, 4, &[1.0], 2, &mut out).unwrap();
        assert!(out.iter().all(|v| v.to_bits() == 0));
    }
}
