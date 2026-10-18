"""Installed-package checks: python checks/native_parity.py [--missing-rust].

Run from outside the source tree (or use python -I) to verify wheel contents.
No fixtures, archived binaries, or optional test framework are required.
"""

import argparse
import importlib.abc
import sys
import warnings
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


def direct_candidate_reference(data, frequencies, tsamp, dm_values, start, stop):
    """Independent absolute-index reference for Candidate's non-circular API."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    dm_values = np.asarray(dm_values, dtype=np.float64)
    delays = np.rint(
        4148808.0
        * dm_values[:, None]
        * (1.0 / frequencies[None, :] ** 2 - 1.0 / frequencies.max() ** 2)
        / 1000.0
        / tsamp
    ).astype(np.int64)
    nt, nchans = stop - start, data.shape[1]
    result = np.zeros((len(dm_values), nt), dtype=np.float32)
    if not len(dm_values) or nt == 0:
        return result, delays

    source_start = max(0, start + int(delays.min()))
    source_stop = min(len(data), stop + int(delays.max()))
    coverage = data[source_start:source_stop]
    fill = (
        np.median(coverage, axis=0).astype(np.float32)
        if len(coverage)
        else np.zeros(nchans, dtype=np.float32)
    )
    output_times = np.arange(start, stop, dtype=np.int64)
    for row, offsets in enumerate(delays):
        total = np.zeros(nt, dtype=np.float32)
        for channel, offset in enumerate(offsets):
            source_times = output_times + offset
            valid = (source_times >= 0) & (source_times < len(data))
            values = np.full(nt, fill[channel], dtype=np.float32)
            values[valid] = data[source_times[valid], channel]
            total += values
        result[row] = total / nchans
    return result, delays


def _direct_candidate_channels(data, frequencies, tsamp, dm, start, stop):
    """Independent direct-index reference for Candidate.dedisperse()."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    delays = np.rint(
        4148808.0
        * float(dm)
        * (1.0 / frequencies**2 - 1.0 / frequencies.max() ** 2)
        / 1000.0
        / tsamp
    ).astype(np.int64)
    output = np.empty((stop - start, len(frequencies)), dtype=np.float32)
    if stop == start:
        return output
    source_start = max(0, start + int(delays.min()))
    source_stop = min(len(data), stop + int(delays.max()))
    coverage = data[source_start:source_stop]
    fill = (
        np.median(coverage, axis=0).astype(np.float32)
        if len(coverage)
        else np.zeros(len(frequencies), dtype=np.float32)
    )
    output_times = np.arange(start, stop, dtype=np.int64)
    for channel, offset in enumerate(delays):
        source_times = output_times + offset
        valid = (source_times >= 0) & (source_times < len(data))
        output[:, channel] = fill[channel]
        output[valid, channel] = data[source_times[valid], channel]
    return output


def public_checks(module, rfi):
    data = (np.arange(129 * 37) % 251).astype(np.uint8).reshape(129, 37)
    # Keep coverage of the public API over layouts and dtypes, but validate it
    # against absolute source indexing rather than another Candidate instance.
    for sample in (
        data,
        data[:, ::2],
        np.asfortranarray(data),
        np.zeros_like(data),
        np.empty((0, 3), np.uint8),
    ):
        obj = candidate(module, sample)
        frequencies = obj.chan_freqs
        obj.dedisperse()
        same(
            obj.dedispersed,
            _direct_candidate_channels(
                sample, frequencies, obj.native_tsamp, obj.dm, 0, len(sample)
            ),
        )
        same(obj.dedispersets(), obj.dedispersed.sum(axis=1, dtype=np.float32))

        dms = np.linspace(0, 2 * obj.dm, 5)
        dmt_expected, _ = direct_candidate_reference(
            sample, frequencies, obj.native_tsamp, dms, 0, len(sample)
        )
        obj.dmtime(dmsteps=5)
        same(obj.dmt, dmt_expected)

    # Exercise the highest-frequency reference for both channel orderings and
    # explicit dmsteps/time_range overrides on the public Candidate methods.
    rng = np.random.default_rng(20261017)
    for fch1, foff in ((1600.0, -400.0), (400.0, 400.0)):
        sample = rng.integers(0, 100, (48, 4), dtype=np.uint16)
        obj = candidate(module, sample)
        obj.dm = 0.4
        obj.fch1 = fch1
        obj.foff = foff
        start, stop = 9, 37
        obj.dedisperse(time_range=(start, stop))
        expected_ft = _direct_candidate_channels(
            sample, obj.chan_freqs, obj.native_tsamp, obj.dm, start, stop
        )
        same(obj.dedispersed, expected_ft)
        same(
            obj.dedispersets(time_range=(start, stop)),
            expected_ft.sum(axis=1, dtype=np.float32),
        )
        dms = np.linspace(0, 2 * obj.dm, 1)
        expected_dmt, _ = direct_candidate_reference(
            sample, obj.chan_freqs, obj.native_tsamp, dms, start, stop
        )
        obj.dmtime(dmsteps=1, time_range=(start, stop))
        same(obj.dmt, expected_dmt)
        assert obj.dmt.shape == (1, stop - start)

        obj.dmtime(dmsteps=4, time_range=(start, stop))
        expected_dmt, _ = direct_candidate_reference(
            sample,
            obj.chan_freqs,
            obj.native_tsamp,
            np.linspace(0, 2 * obj.dm, 4),
            start,
            stop,
        )
        same(obj.dmt, expected_dmt)

    # Missing end samples use the median of real coverage; they never wrap to
    # the first samples in the in-memory chunk.
    edge_data = np.arange(32 * 4, dtype=np.float32).reshape(32, 4)
    edge = candidate(module, edge_data)
    edge.dm = 0.8
    edge.fch1, edge.foff = 1600.0, -400.0
    start, stop = 27, 32
    edge.dedisperse(time_range=(start, stop))
    same(
        edge.dedispersed,
        _direct_candidate_channels(
            edge_data, edge.chan_freqs, edge.native_tsamp, edge.dm, start, stop
        ),
    )
    assert not np.array_equal(edge.dedispersed[-1], edge_data[0])

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
    # Extreme float64 inputs must retain NumPy's overflow exception policy by
    # taking Candidate's safe conversion path instead of the Rust fast path.
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


def candidate_binned_checks(native):
    # Cross a Rust band boundary and channel tail, use nonzero right-hand
    # delays, and leave masked channels in the output denominator.
    rng = np.random.default_rng(20261018)
    data = rng.integers(0, 251, size=(73, 131), dtype=np.uint8)
    dm_delays = np.fromfunction(
        lambda dm, channel: (3 * dm + channel) % 5, (64, 131), dtype=int
    ).astype(np.int64)
    ft_delays = (np.arange(data.shape[1]) % 5).astype(np.int64)
    enabled = np.ones(data.shape[1], dtype=bool)
    enabled[1::7] = False
    start, time_bin, time_size = 2, 3, 4

    expected_ft = np.zeros((time_size, data.shape[1]), dtype=np.float32)
    for out_time in range(time_size):
        for channel in range(data.shape[1]):
            if enabled[channel]:
                lo = start + int(ft_delays[channel]) + out_time * time_bin
                total = sum(int(value) for value in data[lo : lo + time_bin, channel])
                expected_ft[out_time, channel] = total / time_bin

    expected_dmt = np.zeros((len(dm_delays), time_size), dtype=np.float32)
    for dm, delays in enumerate(dm_delays):
        for out_time in range(time_size):
            total = 0
            for channel, delay in enumerate(delays):
                if enabled[channel]:
                    lo = start + int(delay) + out_time * time_bin
                    total += sum(
                        int(value) for value in data[lo : lo + time_bin, channel]
                    )
            expected_dmt[dm, out_time] = total / (data.shape[1] * time_bin)

    for threads in (1, 4):
        ft, dmt = native.candidate_binned(
            data,
            dm_delays,
            ft_delays,
            start,
            time_bin,
            time_size,
            enabled=enabled,
            threads=threads,
        )
        same(ft, expected_ft)
        same(dmt, expected_dmt)


def resize_checks():
    from skimage.transform import resize

    from your.utils.misc import _anti_aliased_shrink

    rng = np.random.default_rng(20261018)
    data = rng.standard_normal((13, 96)).astype(np.float32)
    expected = resize(data, (13, 24), anti_aliasing=True, mode="constant")
    actual = _anti_aliased_shrink(data, 24)
    assert actual is not None
    same(actual, expected)


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

    for name in (
        "dedisperse",
        "dedispersets",
        "dmtime",
        "candidate_binned",
        "rfi_stats",
        "gaussian_decimate",
    ):
        assert callable(getattr(_rust, name))
    direct_checks(_rust)
    candidate_binned_checks(_rust)
    resize_checks()
    import your.candidate as module
    from your.utils import rfi

    public_checks(module, rfi)
    print("PASS: native kernels and NumPy compatibility paths")


if __name__ == "__main__":
    main()
