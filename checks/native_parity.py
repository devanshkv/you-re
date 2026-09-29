"""Installed-package checks: python checks/native_parity.py [--missing-rust].

Run from outside the source tree (or use python -I) to verify wheel contents.
No fixtures, archived binaries, or optional test framework are required.
"""

import argparse
import importlib.abc
import sys
import warnings
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

DTYPES = (np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64)


def same(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.ma.getmaskarray(actual), np.ma.getmaskarray(expected)
    )
    np.testing.assert_array_equal(np.ma.filled(actual, 0), np.ma.filled(expected, 0))
    if actual.dtype.kind == "f":
        zeros = (np.asarray(actual) == 0) & (np.asarray(expected) == 0)
        np.testing.assert_array_equal(
            np.signbit(np.asarray(actual)[zeros]),
            np.signbit(np.asarray(expected)[zeros]),
        )


def reference(data, delays, start, stop):
    # Preserve the historical out-of-range identity shift and channel sum order.
    shifted = np.empty((stop - start, data.shape[1]), dtype=np.float32)
    total = np.zeros(stop - start, dtype=np.float32)
    for channel, delay in enumerate(delays):
        column = data[:, channel]
        if -len(column) < int(delay) < len(column):
            column = np.roll(column, int(delay))
        column = column[start:stop]
        shifted[:, channel] = column
        total += column
    return shifted, total


def candidate(module, data):
    obj = module.Candidate.__new__(module.Candidate)
    obj.data = data
    obj.dm = 7.0
    obj.nchans = data.shape[1]
    obj.fch1 = 1500.0
    obj.foff = -0.25
    obj.formatclass = SimpleNamespace(native_tsamp=lambda _: 0.001)
    return obj


def outputs(obj):
    obj.dedisperse()
    obj.dmtime(dmsteps=5)
    return obj.dedispersed, obj.dedispersets(), obj.dmt


def direct_checks(native):
    rng = np.random.default_rng(20260926)
    for dtype in DTYPES:
        base = rng.integers(0, 100, (129, 37)).astype(dtype)
        for data in (base, np.asfortranarray(base), base[::2, ::2], base[::-1, ::-1]):
            nt, nf = data.shape
            delays = np.resize(
                np.array([0, 1, -1, nt, -nt, 2**63 - 1, -(2**63)], dtype=np.int64), nf
            )
            before = data.copy()
            for start, stop in ((0, nt), (3, nt - 1), (nt, nt)):
                shifted, total = reference(data, delays, start, stop)
                same(native.dedisperse(data, delays, start, stop), shifted)
                same(native.dedispersets(data, delays, start, stop), total)
                terms = np.linspace(-1e-6, 1e-6, nf)
                dms = np.array([-7.0, 0.0, 7.0])
                expected = []
                for dm in dms:
                    bins = np.round(4148808.0 * dm * terms / 1000.0 / 0.001).astype(
                        np.int64
                    )
                    expected.append(reference(data, bins, start, stop)[1])
                same(
                    native.dmtime(data, dms, terms, 0.001, start, stop),
                    np.array(expected),
                )
            same(data, before)
    # Native accumulation converts each channel value to f32 before adding.
    # Keep its cancellation/rounding contract explicit, independently of NumPy f64 +=.
    for dtype in (np.float32, np.float64):
        for values in (
            [1e20, 1.0, -1e20, 3.0],
            [0.1, -0.2, 1.0 / 3, -0.0],
            [np.nan, np.inf, -np.inf, 0.0],
        ):
            data = np.tile(np.array(values, dtype=dtype), (67, 1))
            delays = np.zeros(4, np.int64)
            with np.errstate(invalid="ignore"):
                shifted, total = reference(data.astype(np.float32), delays, 0, 67)
            same(native.dedisperse(data, delays), shifted)
            same(native.dedispersets(data, delays), total)
            same(
                native.dmtime(data, np.zeros(2), np.zeros(4), 0.001),
                np.stack([total, total]),
            )
    for nt, nf in ((0, 3), (3, 0), (0, 0)):
        data = np.empty((nt, nf), dtype=np.float32)
        delays = np.zeros(nf, dtype=np.int64)
        same(native.dedispersets(data, delays), np.zeros(nt, dtype=np.float32))
        same(native.dedisperse(data, delays), np.zeros((nt, nf), dtype=np.float32))
        same(
            native.dmtime(data, np.zeros(2), np.zeros(nf), 0.001),
            np.zeros((2, nt), dtype=np.float32),
        )
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    half = np.array([-1.5, -0.5, 0.5, 1.5, np.nan, np.inf, -np.inf, 2**63])
    terms = half / 4148808.0
    with np.errstate(invalid="ignore"):
        bins = np.round(4148808.0 * terms / 1000.0 / 0.001).astype(np.int64)
    expected = reference(data, bins, 0, 8)[1]
    same(native.dmtime(data, np.ones(2), terms, 0.001), np.stack([expected, expected]))
    same(native.dmtime(data, np.empty(0), terms, 0.001), np.empty((0, 8), np.float32))
    for function in (native.dedisperse, native.dedispersets):
        for args in (
            (data, np.zeros(7, np.int64)),
            (data, np.zeros(8, np.int64), 4, 3),
            (data, np.zeros(8, np.int64), 0, 9),
        ):
            try:
                function(*args)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid input accepted")
        try:
            function(data.astype(np.int64), np.zeros(8, np.int64))
        except TypeError:
            pass
        else:
            raise AssertionError("unsupported dtype accepted")


def public_checks(module, rfi):
    data = (np.arange(129 * 37) % 251).astype(np.uint8).reshape(129, 37)
    obj = candidate(module, data)
    names = ("_rust_dedisperse", "_rust_dedispersets", "_rust_dmtime")
    with ExitStack() as stack:
        calls = [
            stack.enter_context(patch.object(module, name, wraps=getattr(module, name)))
            for name in names
        ]
        actual = outputs(obj)
        assert all(call.call_count > 0 for call in calls)
    # int64 takes the NumPy route while retaining these small integer values.
    expected = outputs(candidate(module, data.astype(np.int64)))
    for a, b in zip(actual, expected):
        same(a, b)
    for sample in (
        data,
        data[:, ::2],
        np.asfortranarray(data),
        np.zeros_like(data),
        np.empty((0, 3), np.uint8),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            actual = rfi.spectral_kurtosis(sample, N=3, d=1.75)
            expected = rfi.spectral_kurtosis(sample.astype(np.uint16), N=3, d=1.75)
        same(actual, expected)
    with patch.object(rfi, "_rust_rfi_stats", wraps=rfi._rust_rfi_stats) as call:
        rfi.spectral_kurtosis(data, N=3)
        assert call.call_count == 1
    # Float64 overflow must retain NumPy's exception policy via the fallback.
    obj = candidate(module, np.full((4, 3), 1e300))
    with patch.object(
        module, "_rust_dedisperse", side_effect=AssertionError("native float64 route")
    ):
        with np.errstate(over="raise"):
            try:
                obj.dedisperse()
            except FloatingPointError:
                pass
            else:
                raise AssertionError("float64 conversion did not raise")


def crop_reference(obj, data, tdf, time_size, dmsteps):
    """Exact DM-time and frequency-time crops from int64 sums, divided once."""
    nt, nf = data.shape
    freqs = obj.chan_freqs
    term = 1 / freqs[0] ** 2 - 1 / freqs**2

    def rolled(delays):
        delays = np.round(delays).astype(np.int64)
        return np.stack(
            [
                np.roll(data[:, ch].astype(np.int64), int(d))
                if -nt < d < nt
                else data[:, ch].astype(np.int64)
                for ch, d in enumerate(delays)
            ],
            axis=1,
        )

    ncols = -(-nt // tdf)
    col0 = ncols // 2 - time_size // 2
    cols = slice(col0 * tdf, (col0 + time_size) * tdf)

    def columns(x):
        x = x[cols]
        return x.reshape(time_size, tdf, *x.shape[1:]).sum(1) / tdf

    dms = obj.dm + np.linspace(-obj.dm, obj.dm, dmsteps)
    dmt = np.stack(
        [columns(rolled((4148808.0 * dm) * term / 1000.0 / 0.001).sum(1)) for dm in dms]
    )
    ft = columns(rolled(4148808.0 * obj.dm * term / 1000 / 0.001))
    return dmt.astype(np.float32), ft.astype(np.float32)


def crop_checks(module):
    """
    crop_planes: with upstream rounding, bit for bit the dmtime/dedisperse
    route decimated and cropped as the candmaker does; exact, the int64 sums.
    """
    from your.utils.misc import _decimate, crop

    rng = np.random.default_rng(20260929)
    # (dtype, nt, nf, dm, tdf, time_size): narrow and wide columns, delays
    # past the chunk, and uint16 sums large enough for float32 to round
    cases = [
        (np.uint8, 512, 37, 7.0, 1, 256),
        (np.uint8, 2000, 37, 30.0, 4, 256),
        (np.uint16, 4096, 64, 7.0, 16, 128),
        (np.uint16, 8192, 64, 2.0, 64, 64),
        (np.int16, 3000, 29, 40.0, 8, 256),
        (np.int32, 1500, 16, 7.0, 2, 256),
    ]
    rounded = False
    for dtype, nt, nf, dm, tdf, time_size in cases:
        info = np.iinfo(dtype)
        low, high = (0, 256) if dtype == np.uint8 else (info.min // 2, info.max // 2)
        if dtype == np.uint16:
            low, high = info.max - 4096, info.max
        data = rng.integers(low, int(high) + 1, (nt, nf)).astype(dtype)
        obj = candidate(module, data)
        obj.dm = dm
        dmsteps = 16

        full = candidate(module, data)
        full.dm = dm
        full.dmtime(dmsteps=dmsteps)
        full.dedisperse()
        dmt, ft = full.dmt, full.dedispersed
        if tdf != 1:
            dmt = _decimate(dmt, tdf, 1, pad=True, mode="median")
            ft = _decimate(ft, tdf, 0, pad=True, mode="median")
        dmt = crop(dmt, dmt.shape[1] // 2 - time_size // 2, time_size, 1)
        ft = crop(ft, ft.shape[0] // 2 - time_size // 2, time_size, 0)
        exact = crop_reference(obj, data, tdf, time_size, dmsteps)

        before = data.copy()
        for threads in (1, 3):
            with patch.object(
                module, "_rust_crop_planes", wraps=module._rust_crop_planes
            ) as call:
                got = obj.crop_planes(
                    tdf, time_size, dmsteps, upstream_rounding=True, threads=threads
                )
                assert call.call_count == 1
            same(got[0], dmt)
            same(got[1], ft)
            got = obj.crop_planes(
                tdf, time_size, dmsteps, upstream_rounding=False, threads=threads
            )
            same(got[0], exact[0])
            same(got[1], exact[1])
        same(data, before)
        rounded |= not np.array_equal(dmt, exact[0])
    assert rounded, "no case exercised float32 rounding"

    # the full-array route stays for padded crops and float data
    data = (np.arange(1001 * 37) % 251).astype(np.uint8).reshape(1001, 37)
    assert candidate(module, data).crop_planes(4, 251) is None  # takes in the pad
    assert candidate(module, data.astype(np.float32)).crop_planes(4, 64) is None

    native = module._rust_crop_planes
    shifts = np.zeros((2, 37), np.int64)
    for bad in (
        dict(dmt_shifts=shifts + 1001),
        dict(tdf=0),
        dict(col0=250),
    ):
        kwargs = dict(
            dmt_shifts=shifts,
            ft_shifts=np.zeros(37, np.int64),
            tdf=4,
            col0=0,
            ncols=250,
        )
        kwargs.update(bad)
        try:
            native(data, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"crop_planes accepted {bad}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing-rust", action="store_true")
    args = parser.parse_args()
    if args.missing_rust:

        class NoRust(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "your._rust":
                    raise ModuleNotFoundError(
                        "Rust deliberately unavailable", name=fullname
                    )
                return None

        sys.meta_path.insert(0, NoRust())
        # Candidate import also imports RFI through package initialization.
        try:
            importlib.import_module("your.candidate")
        except ModuleNotFoundError as error:
            assert error.name == "your._rust", error
        else:
            raise AssertionError("Candidate imported without required Rust extension")
        print("PASS: missing Rust fails imports")
        return

    from your import _rust

    for name in ("crop_planes", "dedisperse", "dedispersets", "dmtime", "rfi_stats"):
        assert callable(getattr(_rust, name))
    direct_checks(_rust)
    import your.candidate as module
    from your.utils import rfi

    public_checks(module, rfi)
    crop_checks(module)
    print("PASS: native kernels and NumPy compatibility paths")


if __name__ == "__main__":
    main()
