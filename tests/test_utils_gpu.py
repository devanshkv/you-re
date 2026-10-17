from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from your.candidate import Candidate
from your.utils.candidate_pipeline import make_plan, process_cpu
from your.utils.gpu import (
    _corner_turn,
    _dmt_shared_capacities,
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


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.int32, np.float16, np.float32, np.float64],
)
def test_corner_turn_preserves_raw_bits_in_partial_tiles_and_pitched_window(dtype):
    import cupy as cp

    nsamples, nchans = 385, 37
    dtype = np.dtype(dtype)
    source = np.frombuffer(
        np.random.default_rng(138).bytes(nsamples * nchans * dtype.itemsize),
        dtype=dtype,
    ).reshape(nsamples, nchans)
    expected = np.zeros((nchans * 2, nsamples + 19), dtype=dtype)
    expected[::2, 10 : 10 + nsamples] = source.T
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_source = cp.asarray(source)
        backing = cp.zeros(expected.shape, dtype=dtype)
        destination = backing[::2, 3:]
        _corner_turn(cp, device_source, destination, nsamples, 7, stream)
    stream.synchronize()

    actual = cp.asnumpy(backing)
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def test_resident_cache_gpu_growth_only_reads_new_edges():
    import cupy as cp

    from your.utils.candidate_scheduler import _ResidentCache

    source = np.random.default_rng(187).integers(
        0, 65536, size=(510, 37), dtype=np.uint16
    )
    reads = []

    def get_data(start, count):
        reads.append((start, count))
        return source[start : start + count]

    cache = _ResidentCache(cp, SimpleNamespace(get_data=get_data), 0, 37, source.dtype)
    first = cache.ensure(100, 485)
    grown = cache.ensure(84, 510)
    contained = cache.ensure(90, 500)
    np.testing.assert_array_equal(cp.asnumpy(cache.array), source[84:510].T)
    assert reads == [(100, 385), (84, 16), (485, 25)]
    assert first["upload_bytes"] == 385 * 37 * source.dtype.itemsize
    assert grown["upload_bytes"] == 41 * 37 * source.dtype.itemsize
    assert contained["upload_bytes"] == 0
    cache.close()


def test_resident_cache_gpu_staging_respects_budget_and_leaves_no_pool_blocks():
    import cupy as cp

    from your.utils.candidate_scheduler import _ResidentCache

    source = np.arange(17 * 37, dtype=np.int32).reshape(17, 37)
    reads = []

    def get_data(start, count):
        reads.append(count)
        return source[start : start + count]

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        cache = _ResidentCache(
            cp,
            SimpleNamespace(get_data=get_data),
            0,
            37,
            source.dtype,
            memory_budget_bytes=source.nbytes + 2 * source.strides[0],
        )
        stats = cache.ensure(0, len(source))
        np.testing.assert_array_equal(cp.asnumpy(cache.array), source.T)
        assert max(reads) == 2
        assert stats["upload_bytes"] == source.nbytes
        assert pool.total_bytes() == pool.used_bytes()
        cache.close()
    pool.free_all_blocks()


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


def _loop_shared_capacity(
    delays, channel_start, channel_stop, max_shared, dm_tile=8, time_tile=128
):
    max_spread = 0
    for dm_start in range(0, delays.shape[0], dm_tile):
        tile = delays[dm_start : dm_start + dm_tile, channel_start:channel_stop]
        if tile.size:
            max_spread = max(max_spread, int(np.max(np.ptp(tile, axis=0))))
    return max(time_tile, min(time_tile + max_spread, int(max_shared) // 4))


@pytest.mark.parametrize("dmsteps", [0, 1, 7, 8, 9, 17, 255, 256, 257])
@pytest.mark.parametrize("monotonic", [False, True])
def test_shared_memory_capacity_matches_loop_for_tails_ranges_and_limits(
    dmsteps, monotonic
):
    rng = np.random.default_rng(1938)
    delays = rng.integers(0, 8000, size=(dmsteps, 65), dtype=np.int64)
    if monotonic:
        delays.sort(axis=0)
    delays += (np.arange(dmsteps, dtype=np.int64) // 8 * 2**40)[:, None]
    delays += (np.arange(65, dtype=np.int64) * 2**36)[None, :]

    for start, stop in ((0, 65), (0, 32), (32, 64), (64, 65), (4, 31), (5, 5)):
        for limit in (-1, 0, 508, 512, 48 * 1024, 48 * 1024 + 3):
            expected = _loop_shared_capacity(delays, start, stop, limit)
            assert _dmt_shared_capacity(delays, start, stop, limit) == expected
    for limit in (-1, 0, 508, 512, 48 * 1024, 48 * 1024 + 3):
        expected = [
            _loop_shared_capacity(delays, start, start + 32, limit)
            for start in range(0, delays.shape[1], 32)
        ]
        np.testing.assert_array_equal(_dmt_shared_capacities(delays, limit), expected)


def test_shared_memory_capacity_includes_widest_partial_dm_and_channel_tiles():
    delays = np.zeros((11, 35), dtype=np.int64)
    delays[9, 34] = 913

    assert _dmt_shared_capacity(delays, 32, 35, 48 * 1024) == 128 + 913
    assert _dmt_shared_capacity(delays, 0, 32, 48 * 1024) == 128
    delays[9, 34] = 2**40
    assert _dmt_shared_capacity(delays, 32, 35, 48 * 1024) == 48 * 1024 // 4


@pytest.mark.parametrize("shape", [(0, 0), (0, 65), (1, 1), (11, 35), (17, 129)])
def test_batched_shared_capacities_honor_dynamic_tiles_and_empty_shapes(
    shape, monkeypatch
):
    from your.utils import gpu

    delays = np.random.default_rng(1938).integers(0, 8000, size=shape, dtype=np.int64)
    original = delays.copy()
    for channel_tile, dm_tile, time_tile in (
        (17, 3, 64),
        (128, 4, 512),
        (256, 8, 1024),
    ):
        monkeypatch.setattr(gpu, "_CHANNEL_TILE", channel_tile)
        monkeypatch.setattr(gpu, "_DM_TILE", dm_tile)
        monkeypatch.setattr(gpu, "_TIME_TILE", time_tile)
        for limit in (-1, 0, 508, 512, 48 * 1024, 48 * 1024 + 3):
            expected = [
                _loop_shared_capacity(
                    delays, start, start + channel_tile, limit, dm_tile, time_tile
                )
                for start in range(0, shape[1], channel_tile)
            ]
            capacities = _dmt_shared_capacities(delays, limit)
            assert capacities.dtype == np.int64
            np.testing.assert_array_equal(capacities, expected)
            assert _dmt_shared_capacity(delays, 0, shape[1], limit) == max(
                expected, default=time_tile
            )
    np.testing.assert_array_equal(delays, original)


def test_batched_shared_capacities_keep_scalar_floor_for_signed_spread_overflow():
    value = 2**62 + 1
    delays = np.tile(np.array([-value, value] * 4, dtype=np.int64)[:, None], (1, 65))
    expected = [
        _loop_shared_capacity(delays, start, min(start + 32, 65), 48 * 1024)
        for start in range(0, 65, 32)
    ]
    np.testing.assert_array_equal(_dmt_shared_capacities(delays, 48 * 1024), expected)
    assert expected == [128, 128, 128]


def test_batched_shared_capacities_clamp_int64_spreads_before_adding_time_tile():
    delays = np.zeros((11, 35), dtype=np.int64)
    delays[7, 0] = np.iinfo(np.int64).max
    delays[9, 34] = np.iinfo(np.int64).max
    delays[:, 10] = np.iinfo(np.int64).max - 1

    for limit in (-1, 508, 48 * 1024):
        expected = [
            _loop_shared_capacity(delays, start, start + 32, limit) for start in (0, 32)
        ]
        np.testing.assert_array_equal(_dmt_shared_capacities(delays, limit), expected)
        assert _dmt_shared_capacity(delays, 0, 35, limit) == max(expected)


def test_process_gpu_sizes_all_channel_groups_once(monkeypatch):
    from your.utils import gpu

    data = np.arange(650, dtype=np.uint16).reshape(10, 65)
    cand = _candidate(data)
    cand.chan_freqs = np.linspace(1000.0, 1400.0, data.shape[1])
    plan = make_plan(cand, time_size=4, frequency_size=3, dmsteps=11)
    counts = {"limit": 0, "capacity": 0}
    original_limit = gpu._shared_limit
    original_capacities = gpu._dmt_shared_capacities

    def shared_limit(cp):
        counts["limit"] += 1
        return original_limit(cp)

    def shared_capacities(delays, max_shared):
        counts["capacity"] += 1
        return original_capacities(delays, max_shared)

    monkeypatch.setattr(gpu, "_shared_limit", shared_limit)
    monkeypatch.setattr(gpu, "_dmt_shared_capacities", shared_capacities)
    actual = process_gpu(cand, plan, data=data)
    expected = process_cpu(cand, plan, data=data)
    assert counts == {"limit": 1, "capacity": 1}
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-4, atol=2e-4)


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
