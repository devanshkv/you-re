"""Installed-package GPU checks: python checks/gpu_parity.py [--require-gpu].

Compares the GPU candmaker planes (your.utils.gpu.gpu_dedisp_and_dmt_crop)
with independent NumPy references. Run from outside the source tree (or use
python -I) to verify the installed package. No fixtures or optional test
framework are required. Without a CUDA device it reports SKIP, or fails with
--require-gpu.

Contract checked, for every output cell:

- the frequency-time plane sums, over each run of `nchans // 256` channels and
  `tdf` samples, the channel rolled by its delay at the candidate's DM;
- the DM-time plane sums every channel, rolled by its delay at each of 256 DMs
  from 0 to twice the candidate's DM, over `tdf` samples;
- `tdf` is 1 below width 3, else width // 2; the kept columns are the central
  `width` of `nsamples // tdf`, wrapping when there are fewer;
- sums are exact for integer data, and float64 for float data, then float32;
- frequency-time delays round float32-squared frequencies in float64, as the
  atomic kernels did; DM-time delays use float64 frequencies throughout;
- the result does not depend on the summing path, the GPU, the run or the
  input's memory layout, and the input is not modified.
"""

import argparse
import sys
from types import SimpleNamespace

import numpy as np

DTYPES = (np.uint8, np.uint16, np.int16, np.float32)


def same(actual, expected, what):
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (what, actual.dtype, expected.dtype)
    if not np.array_equal(actual, expected, equal_nan=True):
        bad = np.argwhere(actual != expected)
        raise AssertionError(
            f"{what}: {len(bad)} cells differ, first {tuple(bad[0])}: "
            f"{actual[tuple(bad[0])]} != {expected[tuple(bad[0])]}"
        )


def ft_delays(freqs, dm, tsamp):
    """Frequency-time delays, rounded as the atomic kernels rounded them."""
    f = np.asarray(freqs, dtype=np.float32)
    squares = (f * f).astype(np.float64)
    shift = -4148808.0 * dm * (1 / squares[0] - 1 / squares) / 1000 / tsamp
    return np.round(shift).astype(np.int64)


def dmt_delays(freqs, dms, tsamp):
    """DM-time delays: (nchans, ndms), float64 throughout."""
    f = np.asarray(freqs, dtype=np.float64)
    shift = -4148808.0 * dms[None, :] * (1 / f[0] ** 2 - 1 / f[:, None] ** 2)
    return np.round(shift / 1000 / tsamp).astype(np.int64)


def span_sums(prefix, starts, tdf):
    """Sum of each channel over `tdf` samples from each start, wrapping."""
    nsamples = prefix.shape[1] - 1
    rows = np.arange(prefix.shape[0])[:, None]
    lo = starts % nsamples
    hi = lo + tdf
    inside = np.minimum(hi, nsamples)
    total = prefix[rows, inside] - prefix[rows, lo]
    wrapped = hi > nsamples
    return total + np.where(wrapped, prefix[rows, np.maximum(hi - nsamples, 0)], 0)


def reference(data, freqs, dm, tsamp, width_samples, width=256):
    """Both planes from their definitions, in NumPy."""
    nsamples, nchans = data.shape
    tdf = 1 if width_samples < 3 else width_samples // 2
    fdf = nchans // 256
    rows, ncols = nchans // fdf, nsamples // tdf
    cols = (ncols // 2 - width // 2 + np.arange(width)) % ncols
    if not np.issubdtype(data.dtype, np.integer):
        return float_reference(data, freqs, dm, tsamp, tdf, fdf, cols)
    values = data.T.astype(np.int64)
    prefix = np.zeros((nchans, nsamples + 1), dtype=np.int64)
    np.cumsum(values, axis=1, out=prefix[:, 1:])

    def spans(delays):
        return span_sums(prefix, cols[None, :] * tdf + delays[:, None], tdf)

    per_channel = spans(ft_delays(freqs, dm, tsamp))
    ft = per_channel[: rows * fdf].reshape(rows, fdf, width).sum(1)
    delays = dmt_delays(freqs, np.linspace(0, 2 * dm, 256), tsamp)
    dmt = np.stack([spans(delays[:, kk]).sum(0) for kk in range(256)])
    return ft.T.astype(np.float32), dmt.astype(np.float32)


def float_reference(data, freqs, dm, tsamp, tdf, fdf, cols):
    """
    Float data sums in float64, which is not associative, so add the samples
    in the kernels' order: channel by channel, each channel's samples in turn.
    """
    nsamples, nchans = data.shape
    rows = nchans // fdf
    values = data.T.astype(np.float64)
    starts = cols * tdf
    ft = np.zeros((rows, len(cols)))
    delays = ft_delays(freqs, dm, tsamp) % nsamples
    channel = np.arange(rows)[:, None] * fdf
    for k in range(fdf):
        for s in range(tdf):
            t = (starts[None, :] + s + delays[channel + k]) % nsamples
            ft += values[channel + k, t]
    dmt = np.zeros((256, len(cols)))
    delays = dmt_delays(freqs, np.linspace(0, 2 * dm, 256), tsamp) % nsamples
    for ch in range(nchans):
        for s in range(tdf):
            dmt += values[ch, (starts[None, :] + s + delays[ch][:, None]) % nsamples]
    return ft.T.astype(np.float32), dmt.astype(np.float32)


def candidate(data, freqs, dm, tsamp, width_samples, dtype):
    return SimpleNamespace(
        data=data,
        chan_freqs=freqs,
        dm=dm,
        width=width_samples,
        your_header=SimpleNamespace(dtype=dtype, tsamp=tsamp),
    )


def planes(gpu, data, freqs, dm, tsamp, width_samples, **kwargs):
    cand = candidate(data, freqs, dm, tsamp, width_samples, data.dtype)
    gpu.gpu_dedisp_and_dmt_crop(cand, **kwargs)
    return cand.dedispersed, cand.dmt


def sample(rng, dtype, shape):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, int(info.max) + 1, shape).astype(dtype)
    return rng.standard_normal(shape).astype(dtype) * 100


def direct_checks(gpu, cuda):
    rng = np.random.default_rng(20260928)
    tsamp = 1.2665e-3
    # (nsamples, nchans, width in samples, DM, band): narrow and wide
    # decimation, fewer columns than kept (wrapping), fdf > 1 with leftover
    # channels, delays several chunks long, and both band orders
    cases = [
        (300, 256, 1, 0.0, (1500.0, -0.5)),
        (300, 336, 2, 475.284, (1500.0, -0.5)),
        (100, 256, 1, 50.0, (1500.0, -0.5)),
        (4096 + 7, 512, 32, 157.7, (1400.0, -0.25)),
        (4096, 700, 16, 3000.0, (800.0, 0.2)),
        (32768 + 13, 256, 256, 698.6, (1500.0, -0.5)),
        (65536, 256, 512, 9000.0, (1500.0, -0.5)),
    ]
    for nsamples, nchans, width_samples, dm, (f0, df) in cases:
        freqs = f0 + df * np.arange(nchans)
        tdf = 1 if width_samples < 3 else width_samples // 2
        for dtype in DTYPES:
            if not np.issubdtype(dtype, np.integer) and nchans * tdf > 16384:
                continue  # the sequential float reference would take minutes
            base = sample(rng, dtype, (nsamples, nchans))
            ft, dmt = reference(base, freqs, dm, tsamp, width_samples)
            what = (
                f"{nsamples}x{nchans} width {width_samples} DM {dm} {np.dtype(dtype)}"
            )
            for layout, data in (
                ("C", base),
                ("F", np.asfortranarray(base)),
                ("strided", np.repeat(base, 2, axis=1)[:, ::2]),
            ):
                before = data.copy()
                # the block path serves integer data only; float data must
                # give the same planes whatever block_from asks for
                for block_from in (1, 128, 10**9):
                    got = planes(
                        gpu,
                        data,
                        freqs,
                        dm,
                        tsamp,
                        width_samples,
                        block_from=block_from,
                    )
                    same(got[0], ft, f"{what} {layout} block_from {block_from} FT")
                    same(got[1], dmt, f"{what} {layout} block_from {block_from} DMT")
                same(data, before, f"{what} {layout} input")
            print(f"  ok {what}", flush=True)

    # accumulators past int32: every sample at the uint16 maximum
    nsamples, nchans, width_samples = 65536, 256, 512
    data = np.full((nsamples, nchans), 65535, dtype=np.uint16)
    freqs = 1500.0 - 0.5 * np.arange(nchans)
    ft, dmt = reference(data, freqs, 300.0, tsamp, width_samples)
    assert dmt.max() > np.iinfo(np.int32).max
    for block_from in (1, 10**9):
        got = planes(
            gpu, data, freqs, 300.0, tsamp, width_samples, block_from=block_from
        )
        same(got[0], ft, f"uint16 maximum block_from {block_from} FT")
        same(got[1], dmt, f"uint16 maximum block_from {block_from} DMT")
    print("  ok int64 accumulators", flush=True)

    # the same planes on every run and every device
    data = sample(rng, np.uint8, (32768, 512))
    freqs = 1500.0 - 0.5 * np.arange(512)
    first = planes(gpu, data, freqs, 2000.0, tsamp, 2048)
    for device in list(range(len(cuda.gpus))) * 2:
        got = planes(gpu, data, freqs, 2000.0, tsamp, 2048, device=device)
        same(got[0], first[0], f"device {device} FT")
        same(got[1], first[1], f"device {device} DMT")
    print(f"  ok repeatable on {len(cuda.gpus)} device(s)", flush=True)

    # fewer than 256 channels is refused, not silently wrong
    try:
        planes(gpu, data[:, :255], freqs[:255], 10.0, tsamp, 2)
    except IndexError:
        pass
    else:
        raise AssertionError("fewer than 256 channels did not raise IndexError")


def public_checks(gpu, cuda):
    """A read through the page-locked buffer gives the planes a plain read does."""
    import os
    import tempfile

    from your.candidate import Candidate
    from your.formats.filwriter import make_sigproc_object

    rng = np.random.default_rng(7)
    nchans, nsamples = 336, 4096
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "parity.fil")
        sigproc = make_sigproc_object(
            rawdatafile=path,
            source_name="parity",
            nchans=nchans,
            foff=-0.5,
            fch1=1500.0,
            tsamp=1.2665e-3,
            tstart=60000.0,
            src_raj=0.0,
            src_dej=0.0,
            machine_id=0,
            nbeams=1,
            ibeam=0,
            nbits=8,
            nifs=1,
            barycentric=0,
            pulsarcentric=0,
            telescope_id=6,
            data_type=0,
            az_start=-1,
            za_start=-1,
        )
        sigproc.write_header(path)
        sigproc.append_spectra(
            rng.integers(0, 256, (nsamples, nchans), dtype=np.uint8), path
        )
        buffer = gpu.PinnedReadBuffer(0, granule=2**20)
        results = []
        for read_buffer in (None, buffer, buffer):
            cand = Candidate(path, dm=300.0, tcand=2.0, width=64, snr=10.0)
            if read_buffer is not None:
                cand.read_buffer = read_buffer
            cand.get_chunk(for_preprocessing=True)
            reference_planes = reference(
                cand.data,
                cand.chan_freqs,
                cand.dm,
                cand.your_header.tsamp,
                cand.width,
            )
            gpu.gpu_dedisp_and_dmt_crop(cand)
            same(cand.dedispersed, reference_planes[0], "Candidate FT")
            same(cand.dmt, reference_planes[1], "Candidate DMT")
            results.append((cand.dedispersed, cand.dmt))
            cand.close()
        for got in results[1:]:
            same(got[0], results[0][0], "page-locked read FT")
            same(got[1], results[0][1], "page-locked read DMT")
    print("  ok Candidate reads, plain and page-locked", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu", action="store_true", help="fail rather than skip"
    )
    args = parser.parse_args()
    from numba import cuda

    if not cuda.is_available():
        if args.require_gpu:
            raise SystemExit("FAIL: no CUDA device")
        print("SKIP: no CUDA device")
        return
    from your.utils import gpu

    direct_checks(gpu, cuda)
    public_checks(gpu, cuda)
    print("PASS: GPU candmaker planes match their NumPy references")


if __name__ == "__main__":
    sys.exit(main())
