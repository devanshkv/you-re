from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from your.candidate import Candidate
from your.utils.candidate_pipeline import make_plan, process_cpu
from your.utils.gpu import (
    _dmt_shared_capacity,
    estimate_process_gpu_bytes,
    gpu_available,
    gpu_dedisperse,
    gpu_dmt,
    process_gpu,
)

pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="requires CuPy and a CUDA GPU"
)


def _candidate(data):
    tsamp = 0.001
    return SimpleNamespace(
        data=data,
        dm=100.0,
        tcand=tsamp * 2,
        width=5,
        native_tsamp=tsamp,
        chan_freqs=np.array([500.0, 800.0, 300.0, 700.0, 400.0]),
        your_header=SimpleNamespace(
            nspectra=data.shape[0], dtype=data.dtype, tsamp=tsamp, foff=-1.0
        ),
        kill_mask=np.array([False]),
        rfi_mask=np.zeros(data.shape[1], dtype=bool),
        flag_rfi=False,
    )


def _data(dtype):
    values = np.arange(50, dtype=np.float64).reshape(10, 5) * 0.5 + 3
    if np.dtype(dtype).kind == "f":
        values[4, 3] = np.nan
        values[5, 2] = np.inf
    return values.astype(dtype)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.int32, np.float16, np.float32, np.float64],
)
def test_process_gpu_matches_direct_cpu_for_supported_input_types(dtype):
    data = _data(dtype)
    cand = _candidate(data)
    plan = make_plan(
        cand, time_size=4, frequency_size=3, dmsteps=5, output_dtype=np.float32
    )

    expected = process_cpu(cand, plan, data=data)
    profile = {}
    actual = process_gpu(cand, plan, data=data, device=0, profile=profile)

    assert actual[0].shape == (4, 3)
    assert actual[1].shape == (5, 4)
    assert actual[0].dtype == np.float32
    assert actual[1].dtype == np.float32
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=2e-4)
    assert {
        "upload_s",
        "median_s",
        "scale_s",
        "dedisp_s",
        "bin_s",
        "download_s",
        "finish_s",
    } <= profile.keys()


def test_process_gpu_accepts_channel_major_resident_without_mutating_it():
    import cupy as cp

    data = _data(np.float32)
    cand = _candidate(data)
    plan = make_plan(cand, time_size=4, frequency_size=3, dmsteps=5)
    resident = cp.asarray(data.T.copy())
    before = resident.copy()

    expected = process_cpu(cand, plan, data=data)
    actual = process_gpu(
        cand,
        plan,
        resident=resident,
        resident_start=plan.read_start,
        device=0,
    )

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(
        cp.asnumpy(resident), cp.asnumpy(before), rtol=0, atol=0, equal_nan=True
    )


def test_public_gpu_wrappers_use_direct_bounds_and_explicit_dm_trials():
    data = np.arange(50, dtype=np.float32).reshape(10, 5) * 0.5 + 3
    cand = _candidate(data)
    freqs = cand.chan_freqs
    tsamp = cand.native_tsamp

    def direct_input(self, dm_values, start, stop):
        dm_values = np.asarray(dm_values, dtype=np.float64)
        delays = np.rint(
            4148808.0
            * dm_values[:, None]
            * (1 / freqs[None, :] ** 2 - 1 / freqs.max() ** 2)
            / 1000
            / tsamp
        ).astype(np.int64)
        lo = max(0, start + int(delays.min()))
        hi = min(len(data), stop + int(delays.max()))
        return data[lo:hi].copy(), delays, lo

    cand._direct_input = MethodType(direct_input, cand)

    gpu_dedisperse(cand, dms=100.0, time_range=(2, 8))
    delays = np.rint(
        4148808.0 * 100.0 * (1 / freqs**2 - 1 / freqs.max() ** 2) / 1000 / tsamp
    ).astype(np.int64)
    lo = max(0, 2 + int(delays.min()))
    hi = min(len(data), 8 + int(delays.max()))
    medians = np.median(data[lo:hi], axis=0)
    expected_ft = np.empty((6, len(freqs)), dtype=np.float32)
    for out, sample in enumerate(range(2, 8)):
        for channel, delay in enumerate(delays):
            src = sample + int(delay)
            expected_ft[out, channel] = (
                data[src, channel] if 0 <= src < len(data) else medians[channel]
            )
    np.testing.assert_allclose(cand.dedispersed, expected_ft, rtol=1e-6, atol=1e-6)

    dm_values = np.linspace(0, 200, 5)
    gpu_dmt(cand, dmsteps=5, time_range=(2, 8))
    expected_dmt = np.empty((5, 6), dtype=np.float32)
    for row, dm in enumerate(dm_values):
        dm_delays = np.rint(
            4148808.0 * dm * (1 / freqs**2 - 1 / freqs.max() ** 2) / 1000 / tsamp
        ).astype(np.int64)
        for out, sample in enumerate(range(2, 8)):
            values = [
                data[sample + int(delay), channel]
                if 0 <= sample + int(delay) < len(data)
                else medians[channel]
                for channel, delay in enumerate(dm_delays)
            ]
            expected_dmt[row, out] = np.mean(values)
    np.testing.assert_allclose(cand.dmt, expected_dmt, rtol=1e-6, atol=1e-6)


def test_public_wrappers_use_candidate_file_halo_at_chunk_edge():
    observation = np.arange(100 * 5, dtype=np.float32).reshape(100, 5)
    chunk_start = 10
    data = observation[chunk_start : chunk_start + 8].copy()
    cand = _candidate(data)
    cand.native_tsamp = 0.1
    cand.your_header.tsamp = 0.1
    cand.your_header.nspectra = len(observation)
    cand._chunk_start = chunk_start
    cand._chunk_shape = data.shape
    cand.get_data = lambda nstart, nsamp: observation[nstart : nstart + nsamp]
    cand._direct_input = MethodType(Candidate._direct_input, cand)
    cand._direct_cpu = MethodType(Candidate._direct_cpu, cand)

    time_range = (6, 8)
    expected_ft = cand._direct_cpu([cand.dm], *time_range, keep_channels=True)
    gpu_dedisperse(cand, dms=cand.dm, time_range=time_range)
    np.testing.assert_allclose(cand.dedispersed, expected_ft, rtol=1e-6, atol=1e-6)

    dm_values = np.linspace(0, 2 * cand.dm, 5)
    expected_dmt = cand._direct_cpu(dm_values, *time_range)
    gpu_dmt(cand, dmsteps=5, time_range=time_range)
    np.testing.assert_allclose(cand.dmt, expected_dmt, rtol=1e-6, atol=1e-6)


def test_tiled_dmt_falls_back_to_direct_reads_for_wide_delay_spread():
    data = np.arange(50, dtype=np.float32).reshape(10, 5) * 0.5 + 3
    cand = _candidate(data)
    plan = make_plan(cand, time_size=1, frequency_size=5, dmsteps=8)
    delays = np.broadcast_to(
        (np.arange(8, dtype=np.int64) * 2000)[:, None], (8, 5)
    ).copy()
    plan = replace(
        plan,
        time_size=1,
        frequency_size=5,
        dmsteps=8,
        time_bin=1,
        output_start=0,
        native_count=1,
        dm_values=np.arange(8, dtype=np.float64),
        delays=delays,
        ft_delays=np.zeros(5, dtype=np.int64),
        required_start=0,
        required_stop=14001,
        read_start=0,
        read_stop=len(data),
        pad_before=0,
        pad_after=13991,
    )

    expected = process_cpu(cand, plan, data=data)
    actual = process_gpu(cand, plan, data=data)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-6)


def test_shared_memory_capacity_uses_each_dm_tile_spread():
    delays = np.repeat((np.arange(256, dtype=np.int64) * 1000)[:, None], 4, axis=1)

    capacity = _dmt_shared_capacity(delays, 0, delays.shape[1], 48 * 1024)

    assert capacity == 128 + 7000


def test_gpu_workspace_estimate_is_pure_and_positive():
    cand = _candidate(_data(np.uint8))
    plan = make_plan(cand, time_size=4, frequency_size=3, dmsteps=5)
    estimate = estimate_process_gpu_bytes(plan, cand.data.dtype)
    assert estimate > 0
    assert isinstance(estimate, int)


def _two_channel_candidate(data, kill_mask=None):
    return SimpleNamespace(
        data=data,
        dm=0.0,
        tcand=0.004,
        width=2,
        native_tsamp=0.001,
        chan_freqs=np.array([1000.0, 1400.0]),
        your_header=SimpleNamespace(
            nspectra=data.shape[0], dtype=data.dtype, tsamp=0.001, foff=-1.0
        ),
        kill_mask=np.array([False, False]) if kill_mask is None else kill_mask,
        flag_rfi=False,
    )


def test_process_gpu_scales_large_finite_float64_before_fp32_accumulation():
    import cupy as cp

    data = 1e40 * np.array(
        [
            [1, -1],
            [2, -2],
            [-1, 1],
            [-2, 2],
            [1, 2],
            [2, 1],
            [-2, -1],
            [-1, -2],
        ],
        dtype=np.float64,
    )
    cand = _two_channel_candidate(data)
    plan = make_plan(cand, time_size=8, frequency_size=2, dmsteps=3)
    profile = {}

    expected = process_cpu(cand, plan, data=data)
    actual = process_gpu(cand, plan, data=data, profile=profile)

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-5)
    assert np.any(np.abs(actual[0]) > 0.1)
    assert np.any(np.abs(actual[1]) > 0.1)
    assert profile["scale_s"] >= 0

    resident = cp.asarray(data.T.copy())
    resident_actual = process_gpu(
        cand,
        plan,
        resident=resident,
        resident_start=plan.read_start,
    )
    np.testing.assert_allclose(resident_actual[0], expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(resident_actual[1], expected[1], rtol=2e-5, atol=2e-5)


def test_process_gpu_constant_nonzero_bandpass_is_zero_with_empty_kill_mask():
    data = np.tile(np.array([100.0, 250.0], dtype=np.float32), (8, 1))
    cand = _two_channel_candidate(data, kill_mask=np.array([], dtype=bool))
    plan = make_plan(cand, time_size=8, frequency_size=2, dmsteps=3)

    expected = process_cpu(cand, plan, data=data)
    actual = process_gpu(cand, plan, data=data)

    for product in (*expected, *actual):
        np.testing.assert_array_equal(product, np.zeros_like(product))


def test_process_gpu_scale_ignores_killed_extreme_channel():
    data = np.column_stack(
        (
            np.array([1, 2, -1, -2, 1, 2, -2, -1], dtype=np.float64) * 1e40,
            np.array([1, -1, 2, -2, 3, -3, 4, -4], dtype=np.float64) * 1e-3,
        )
    )
    cand = _two_channel_candidate(data, kill_mask=np.array([True, False]))
    plan = make_plan(cand, time_size=8, frequency_size=2, dmsteps=3)

    expected = process_cpu(cand, plan, data=data)
    actual = process_gpu(cand, plan, data=data)

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=2e-5)
    assert np.any(np.abs(actual[0]) > 0.1)
    assert np.any(np.abs(actual[1]) > 0.1)
