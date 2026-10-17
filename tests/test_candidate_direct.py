from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from your.candidate import Candidate


def _candidate(data, frequencies, *, dm=0.4, tsamp=0.001):
    candidate = Candidate.__new__(Candidate)
    candidate.data = np.array(data, copy=True)
    candidate.dm = dm
    candidate.tcand = 0.0
    candidate.width = 2
    candidate.nchans = candidate.data.shape[1]
    candidate.fch1 = float(frequencies[0])
    candidate.foff = float(frequencies[1] - frequencies[0])
    candidate.formatclass = SimpleNamespace(native_tsamp=lambda _: tsamp)
    candidate.kill_mask = np.zeros(candidate.nchans, dtype=bool)
    candidate.rfi_mask = np.zeros(candidate.nchans, dtype=bool)
    candidate.flag_rfi = False
    candidate._chunk_start = None
    candidate._chunk_shape = None
    candidate.dedispersed = None
    candidate.dmt = None
    return candidate


def _delay_bins(frequencies, dm_values, tsamp):
    frequencies = np.asarray(frequencies, dtype=np.float64)
    dm_values = np.asarray(dm_values, dtype=np.float64)
    return np.rint(
        4148808.0
        * dm_values[:, None]
        * (1 / frequencies[None, :] ** 2 - 1 / frequencies.max() ** 2)
        / 1000.0
        / tsamp
    ).astype(np.int64)


def _reference_channels(data, frequencies, tsamp, dm_values, start, stop):
    """Direct absolute-index reference with clipped median edge fill."""
    delays = _delay_bins(frequencies, dm_values, tsamp)
    nchans = data.shape[1]
    output = np.zeros((len(dm_values), stop - start, nchans), dtype=np.float32)
    if not len(dm_values) or start == stop:
        return output

    source_start = max(0, start + int(delays.min()))
    source_stop = min(len(data), stop + int(delays.max()))
    coverage = data[source_start:source_stop]
    medians = (
        np.median(coverage, axis=0).astype(np.float32)
        if len(coverage)
        else np.zeros(nchans, dtype=np.float32)
    )
    times = np.arange(start, stop, dtype=np.int64)
    for row, offsets in enumerate(delays):
        for channel, delay in enumerate(offsets):
            source_times = times + delay
            valid = (source_times >= 0) & (source_times < len(data))
            output[row, :, channel] = medians[channel]
            output[row, valid, channel] = data[source_times[valid], channel]
    return output


@pytest.mark.parametrize(
    "frequencies",
    ([400.0, 800.0, 1200.0, 1600.0], [1600.0, 1200.0, 800.0, 400.0]),
)
def test_direct_products_reference_highest_frequency_and_reductions(frequencies):
    rng = np.random.default_rng(731)
    data = rng.integers(5, 20, size=(64, 4)).astype(np.float32)
    tsamp = 0.001
    dm = 0.4
    target_time = 24
    offsets = _delay_bins(frequencies, [dm], tsamp)[0]
    data[target_time + offsets, np.arange(4)] += 100
    candidate = _candidate(data, frequencies, dm=dm, tsamp=tsamp)

    start, stop = 8, 48
    expected_ft = _reference_channels(data, frequencies, tsamp, [dm], start, stop)[0]
    candidate.dedisperse(time_range=(start, stop))
    np.testing.assert_array_equal(candidate.dedispersed, expected_ft)
    np.testing.assert_array_equal(
        candidate.dedispersets(time_range=(start, stop)),
        expected_ft.sum(axis=1, dtype=np.float32),
    )

    dm_values = np.linspace(0, 2 * dm, 5)
    expected_dm_channels = _reference_channels(
        data, frequencies, tsamp, dm_values, start, stop
    )
    candidate.dmtime(dmsteps=5, time_range=(start, stop))
    expected_dmt = expected_dm_channels.sum(axis=2, dtype=np.float32) / len(frequencies)
    np.testing.assert_array_equal(candidate.dmt, expected_dmt)
    assert candidate.dedispersed[target_time - start].max() >= 100
    assert candidate.dmt.shape == (5, stop - start)


def test_direct_edge_samples_use_real_coverage_median_without_wraparound():
    frequencies = [400.0, 800.0, 1200.0, 1600.0]
    data = np.arange(32 * 4, dtype=np.float32).reshape(32, 4) + 100
    candidate = _candidate(data, frequencies, dm=0.8)
    start, stop = 27, 32
    dm_values = np.linspace(0, 2 * candidate.dm, 4)

    expected_channels = _reference_channels(
        data, frequencies, candidate.native_tsamp, dm_values, start, stop
    )
    candidate.dmtime(dmsteps=4, time_range=(start, stop))
    expected_dmt = expected_channels.sum(axis=2, dtype=np.float32) / len(frequencies)
    np.testing.assert_array_equal(candidate.dmt, expected_dmt)

    ft_dm = _reference_channels(
        data, frequencies, candidate.native_tsamp, [candidate.dm], start, stop
    )[0]
    candidate.dedisperse(time_range=(start, stop))
    np.testing.assert_array_equal(candidate.dedispersed, ft_dm)
    largest_delay_channel = int(np.argmin(frequencies))
    assert (
        _delay_bins(frequencies, [candidate.dm], candidate.native_tsamp)[
            0, largest_delay_channel
        ]
        > len(data) - start
    )
    assert (
        candidate.dedispersed[-1, largest_delay_channel]
        != data[0, largest_delay_channel]
    )


def test_dmtime_explicit_single_trial_and_time_range_override():
    frequencies = [400.0, 800.0, 1200.0, 1600.0]
    data = np.arange(48 * 4, dtype=np.float32).reshape(48, 4)
    candidate = _candidate(data, frequencies, dm=0.4)
    start, stop = 11, 29

    candidate.dmtime(dmsteps=1, time_range=(start, stop))
    expected = _reference_channels(
        data, frequencies, candidate.native_tsamp, [0.0], start, stop
    )
    np.testing.assert_array_equal(
        candidate.dmt, expected.sum(axis=2, dtype=np.float32) / len(frequencies)
    )
    assert candidate.dmt.shape == (1, stop - start)

    candidate.dmtime(dmsteps=3, time_range=(start, stop))
    expected = _reference_channels(
        data,
        frequencies,
        candidate.native_tsamp,
        np.linspace(0, 2 * candidate.dm, 3),
        start,
        stop,
    )
    np.testing.assert_array_equal(
        candidate.dmt, expected.sum(axis=2, dtype=np.float32) / len(frequencies)
    )
    assert candidate.dmt.shape == (3, stop - start)


def test_loaded_file_reads_real_halo_instead_of_wrapping_chunk():
    filename = Path(__file__).parent / "data" / "28.fil"
    candidate = Candidate(fp=str(filename), dm=100.0, tcand=2.0, width=2)
    try:
        base, chunk_size = 1800, 24
        candidate.get_chunk(
            tstart=base * candidate.native_tsamp,
            tstop=(base + chunk_size) * candidate.native_tsamp,
            for_preprocessing=False,
        )
        assert candidate._chunk_start == base
        assert candidate.data.shape[0] == chunk_size

        frequencies = candidate.chan_freqs
        dm_values = np.array([candidate.dm], dtype=np.float64)
        delays = _delay_bins(frequencies, dm_values, candidate.native_tsamp)[0]
        halo_size = chunk_size + int(delays.max())
        halo = candidate.get_data(nstart=base, nsamp=halo_size)
        expected_ft = _reference_channels(
            halo,
            frequencies,
            candidate.native_tsamp,
            dm_values,
            0,
            chunk_size,
        )[0]

        candidate.dedisperse(time_range=(0, chunk_size))
        np.testing.assert_array_equal(candidate.dedispersed, expected_ft)
        low_frequency_channel = int(np.argmin(frequencies))
        assert delays[low_frequency_channel] > chunk_size
        assert (
            candidate.dedispersed[0, low_frequency_channel]
            == halo[delays[low_frequency_channel], low_frequency_channel]
        )
        assert (
            candidate.dedispersed[0, low_frequency_channel]
            != candidate.data[0, low_frequency_channel]
        )
    finally:
        candidate.close()


@pytest.mark.parametrize("nonfinite", [False, True])
def test_integer_fractional_and_nonfinite_median_padding_matches_gpu(nonfinite):
    from your.utils.gpu import gpu_available

    if not gpu_available():
        pytest.skip("requires a CUDA GPU")
    data = np.array([[0, 0], [2, 1], [4, 2], [6, 3]], dtype=np.uint8)
    if nonfinite:
        data = data.astype(np.float32)
        data[0, 1] = np.nan
    cand = _candidate(data, [1500.0, 1000.0], dm=1000.0, tsamp=1.0)
    cand.device = 0
    cand.dedisperse(target="CPU")
    expected = cand.dedispersed.copy()
    assert expected[-1, 1] == (2.0 if nonfinite else 1.5)
    assert np.isfinite(expected).all()
    cand.dedisperse(target="GPU")
    np.testing.assert_array_equal(cand.dedispersed, expected)
    cand.dmtime(dmsteps=3, target="CPU")
    expected = cand.dmt.copy()
    cand.dmtime(dmsteps=3, target="GPU")
    np.testing.assert_allclose(cand.dmt, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mask", [np.array([], dtype=bool), np.array([1]), False])
def test_public_dedispersion_uses_shared_mask_rules(mask):
    from your.utils.gpu import gpu_available

    data = np.arange(16, dtype=np.uint8).reshape(8, 2)
    cand = _candidate(data, [1500.0, 1000.0], dm=0.0)
    cand.kill_mask = mask
    cand.device = 0
    cand.dedisperse(target="CPU")
    expected = data.astype(np.float32)
    if np.asarray(mask).dtype.kind in "iu":
        expected[:, 1] = 0
    np.testing.assert_array_equal(cand.dedispersed, expected)
    if gpu_available():
        cand.dedisperse(target="GPU")
        np.testing.assert_array_equal(cand.dedispersed, expected)
