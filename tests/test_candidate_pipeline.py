from types import SimpleNamespace

import numpy as np
import pytest

from your.utils.candidate_pipeline import (
    _channel_medians,
    finish_products,
    make_plan,
    process_cpu,
    read_plan_data,
)
from your.utils.gpu import _candidate_input_scale, estimate_process_gpu_bytes


def _candidate(
    frequencies=(1000.0, 1200.0, 1400.0),
    *,
    nspectra=2000,
    tsamp=0.001,
    dm=20.0,
    tcand=0.5,
    width=2,
    kill_mask=None,
    flag_rfi=False,
):
    return SimpleNamespace(
        dm=dm,
        tcand=tcand,
        width=width,
        native_tsamp=tsamp,
        chan_freqs=np.asarray(frequencies),
        your_header=SimpleNamespace(
            nspectra=nspectra,
            dtype=np.float32,
            foff=frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1.0,
        ),
        kill_mask=np.array([False]) if kill_mask is None else kill_mask,
        flag_rfi=flag_rfi,
    )


def test_make_plan_uses_ties_to_even_and_fixed_output_overrides():
    delay_at_dm_three = 4148808.0 * 3.0 * (1.0 / 1000.0**2 - 1.0 / 2000.0**2) / 1000.0
    cand = _candidate(
        frequencies=(1000.0, 2000.0),
        dm=3.0,
        tcand=0.0,
        tsamp=delay_at_dm_three / 1.5,
        width=5,
    )

    plan = make_plan(
        cand, time_size=4, frequency_size=7, dmsteps=3, output_dtype="float32"
    )

    assert plan.output_dtype == np.dtype(np.float32)
    assert plan.time_bin == 2
    assert plan.native_count == 8
    assert plan.output_start == -4
    assert plan.dm_values.dtype == np.float64
    np.testing.assert_array_equal(plan.dm_values, [0.0, 3.0, 6.0])
    # Halfway values round to the nearest even integer under NumPy's ties-to-even rule.
    np.testing.assert_array_equal(plan.delays[:, 0], [0, 2, 3])
    np.testing.assert_array_equal(plan.ft_delays, [2, 0])
    assert plan.pad_before == 4
    assert plan.read_start == 0

    one_trial = make_plan(cand, time_size=3, frequency_size=1, dmsteps=1)
    np.testing.assert_array_equal(one_trial.dm_values, [0.0])
    assert one_trial.delays.shape == (1, 2)
    assert one_trial.output_dtype == np.dtype(np.float16)


@pytest.mark.parametrize(
    "frequencies",
    [(1000.0, 1200.0, 1400.0), (1400.0, 1200.0, 1000.0)],
)
def test_cpu_products_align_a_pulse_for_both_frequency_orders(frequencies):
    cand = _candidate(frequencies)
    plan = make_plan(cand, time_size=9, frequency_size=4, dmsteps=3)
    data = np.full((plan.read_stop - plan.read_start, plan.nchans), 5, dtype=np.float32)
    pulse_time = int(np.floor(cand.tcand / cand.native_tsamp))
    for channel, delay in enumerate(plan.ft_delays):
        source_sample = pulse_time + int(delay)
        data[source_sample - plan.read_start, channel] += 20
    original = data.copy()

    ft, dmt = process_cpu(cand, plan, data)

    assert ft.shape == (9, 4)
    assert dmt.shape == (3, 9)
    assert ft.dtype == dmt.dtype == np.float16
    assert int(np.argmax(ft.astype(np.float32).mean(axis=1))) == 4
    assert np.unravel_index(np.argmax(dmt), dmt.shape) == (1, 4)
    np.testing.assert_array_equal(data, original)


def _direct_numpy_products(data, plan):
    real = np.asarray(data, dtype=np.float64)
    medians = np.median(real, axis=0) if len(real) else np.zeros(plan.nchans)
    logical = np.empty((plan.required_stop - plan.required_start, plan.nchans))
    logical[:] = medians
    overlap_start = max(plan.required_start, plan.read_start)
    overlap_stop = min(plan.required_stop, plan.read_stop)
    if overlap_stop > overlap_start:
        logical[
            overlap_start - plan.required_start : overlap_stop - plan.required_start
        ] = real[overlap_start - plan.read_start : overlap_stop - plan.read_start]

    offset = plan.output_start - plan.required_start
    times = np.arange(plan.native_count, dtype=np.int64) + offset
    all_delays = np.concatenate((plan.delays.ravel(), plan.ft_delays))
    for delay in all_delays:
        assert offset + int(delay) >= 0
        assert offset + plan.native_count + int(delay) <= len(logical)

    ft_native = np.empty((plan.native_count, plan.nchans), dtype=np.float32)
    for channel, delay in enumerate(plan.ft_delays):
        ft_native[:, channel] = logical[times + int(delay), channel]
    ft = ft_native.reshape(plan.time_size, plan.time_bin, plan.nchans).mean(
        axis=1, dtype=np.float32
    )

    dmt_native = np.zeros((plan.dmsteps, plan.native_count), dtype=np.float32)
    for dm_index, delays in enumerate(plan.delays):
        for channel, delay in enumerate(delays):
            dmt_native[dm_index] += logical[times + int(delay), channel]
    dmt_native /= np.float32(plan.nchans)
    dmt = dmt_native.reshape(plan.dmsteps, plan.time_size, plan.time_bin).mean(
        axis=2, dtype=np.float32
    )
    return finish_products(ft, dmt, plan)


def test_read_plan_data_returns_only_clipped_real_samples():
    cand = _candidate(frequencies=(1400.0, 1000.0), nspectra=10, dm=0.0, tcand=0.001)
    plan = make_plan(cand, time_size=4, frequency_size=2, dmsteps=1)
    source = np.arange(20, dtype=np.int16).reshape(10, 2)

    class Reader:
        your_header = cand.your_header

        def get_data(self, *, nstart, nsamp):
            assert (nstart, nsamp) == (0, 3)
            return source[nstart : nstart + nsamp]

    clipped = read_plan_data(Reader(), plan)

    assert (plan.read_start, plan.read_stop) == (0, 3)
    assert clipped.shape == (3, 2)
    assert clipped.dtype == np.int16
    np.testing.assert_array_equal(clipped, source[:3])


def test_edge_padding_uses_real_channel_medians_and_empty_reads_are_valid():
    edge = _candidate(frequencies=(1400.0,), nspectra=10, dm=0.0, tcand=0.001)
    edge_plan = make_plan(edge, time_size=4, frequency_size=1, dmsteps=2)
    assert edge_plan.pad_before == 1
    edge_data = np.full((edge_plan.read_stop, 1), 7, dtype=np.uint8)

    edge_ft, _ = process_cpu(edge, edge_plan, edge_data)

    np.testing.assert_array_equal(edge_ft, np.zeros((4, 1), dtype=np.float16))

    outside = _candidate(
        frequencies=(1400.0, 1000.0),
        nspectra=10,
        dm=0.0,
        tcand=-0.020,
    )
    outside_plan = make_plan(outside, time_size=4, frequency_size=3, dmsteps=1)
    empty = read_plan_data(outside, outside_plan)
    assert empty.shape == (0, 2)
    assert outside_plan.pad_before == 4

    ft, dmt = process_cpu(outside, outside_plan)

    np.testing.assert_array_equal(ft, np.zeros((4, 3), dtype=np.float16))
    np.testing.assert_array_equal(dmt, np.zeros((1, 4), dtype=np.float16))


def test_constant_nonzero_bandpass_finishes_as_zeros():
    cand = _candidate(frequencies=(1000.0, 1400.0), dm=0.0)
    plan = make_plan(cand, time_size=8, frequency_size=3, dmsteps=2)
    data = np.tile(
        np.array([100.0, 250.0], dtype=np.float32),
        (plan.read_stop - plan.read_start, 1),
    )

    ft, dmt = process_cpu(cand, plan, data)

    np.testing.assert_array_equal(ft, np.zeros_like(ft))
    np.testing.assert_array_equal(dmt, np.zeros_like(dmt))


def test_channel_medians_promote_float32_before_reduction():
    values = np.array([[3.0e38], [3.4e38]], dtype=np.float32)

    median = _channel_medians(values, 1)

    assert np.isfinite(median[0])
    np.testing.assert_allclose(median, [3.2e38], rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32, np.float16])
def test_gpu_input_scale_skips_peak_scan_for_bounded_dtypes(dtype):
    class NoCuPyUse:
        def __getattr__(self, name):
            raise AssertionError(f"bounded input scale unexpectedly used CuPy.{name}")

    profile = {}
    scale = _candidate_input_scale(
        cp=NoCuPyUse(),
        data=np.ones((4, 2), dtype=dtype),
        resident=None,
        resident_start=None,
        read_start=0,
        read_stop=4,
        enabled=np.ones(2, dtype=bool),
        device=0,
        stream=None,
        profile=profile,
    )

    assert scale == 1.0
    assert profile["scale_s"] >= 0


def test_gpu_workspace_estimate_drops_scale_scan_buffer_for_bounded_dtype():
    cand = _candidate(frequencies=(1000.0, 1400.0), dm=0.0)
    plan = make_plan(cand, time_size=8, frequency_size=2, dmsteps=3)

    bounded = estimate_process_gpu_bytes(plan, np.uint16)
    floating = estimate_process_gpu_bytes(plan, np.float32)

    assert bounded < floating


def test_samples_past_file_end_are_padded_instead_of_wrapped():
    cand = _candidate(
        frequencies=(1000.0, 1400.0),
        nspectra=1000,
        tsamp=0.001,
        dm=2.5,
        tcand=0.995,
    )
    plan = make_plan(cand, time_size=4, frequency_size=2, dmsteps=2)
    assert plan.pad_after > 0
    data = np.zeros((plan.read_stop - plan.read_start, 2), dtype=np.float32)
    # A circular shift of the clipped read would wrap this into the low-frequency
    # center sample. Its true source time is beyond the file and receives median 0.
    data[0, 0] = 100

    ft, _ = process_cpu(cand, plan, data)

    np.testing.assert_array_equal(ft, np.zeros((4, 2), dtype=np.float16))


@pytest.mark.parametrize(
    "input_dtype",
    [
        np.bool_,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("output_dtype", ["float16", "float32"])
def test_cpu_products_accept_finite_numeric_dtypes(input_dtype, output_dtype):
    cand = _candidate(frequencies=(1000.0, 1400.0), dm=0.0)
    plan = make_plan(
        cand, time_size=5, frequency_size=3, dmsteps=1, output_dtype=output_dtype
    )
    data = np.zeros((plan.read_stop - plan.read_start, 2), dtype=input_dtype)
    data[2, :] = 1

    ft, dmt = process_cpu(cand, plan, data)

    dtype = np.dtype(output_dtype)
    assert ft.shape == (5, 3)
    assert dmt.shape == (1, 5)
    assert ft.dtype == dmt.dtype == dtype
    assert np.all(np.isfinite(ft))
    assert np.all(np.isfinite(dmt))


def test_cpu_scales_finite_float64_products_before_fp32_accumulation():
    cand = _candidate(
        frequencies=(1000.0, 1400.0),
        nspectra=8,
        tsamp=0.001,
        dm=0.0,
        tcand=0.004,
    )
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
    plan = make_plan(cand, time_size=8, frequency_size=2, dmsteps=3)

    ft, dmt = process_cpu(cand, plan, data)

    assert np.all(np.isfinite(ft))
    assert np.all(np.isfinite(dmt))
    assert np.any(np.abs(ft) > 0.1)
    assert np.any(np.abs(dmt) > 0.1)


def test_kill_mask_precedes_rfi_mask_and_does_not_mutate_input(monkeypatch):
    cand = _candidate(
        frequencies=(1000.0, 1400.0),
        dm=0.0,
        kill_mask=np.array([True, False]),
        flag_rfi=True,
    )
    plan = make_plan(cand, time_size=5, frequency_size=2, dmsteps=1)
    data = np.ones((plan.read_stop - plan.read_start, 2), dtype=np.float32)
    data[2, 0] = 20
    data[3, 1] = 30
    original = data.copy()

    def filter_after_kill(*, data, **kwargs):
        assert np.all(data[:, 0] == 0)
        assert np.any(data[:, 1] != 0)
        return np.array([False, True])

    monkeypatch.setattr("your.utils.candidate_pipeline.sk_sg_filter", filter_after_kill)

    ft, dmt = process_cpu(cand, plan, data)

    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(ft, np.zeros((5, 2), dtype=np.float16))
    np.testing.assert_array_equal(dmt, np.zeros((1, 5), dtype=np.float16))


def test_finish_products_profiles_resize_detrend_and_normalization():
    cand = _candidate(frequencies=(1000.0, 1400.0), dm=0.0)
    plan = make_plan(cand, time_size=5, frequency_size=3, dmsteps=2)
    ft = np.arange(10, dtype=np.float32).reshape(5, 2)
    dmt = np.arange(10, dtype=np.float32).reshape(2, 5)
    profile = {}

    finished_ft, finished_dmt = finish_products(ft, dmt, plan, profile=profile)

    assert finished_ft.shape == (5, 3)
    assert finished_dmt.shape == (2, 5)
    assert set(profile) == {"resize_seconds", "detrend_seconds", "normalize_seconds"}
    assert all(np.isfinite(value) and value >= 0 for value in profile.values())


def test_cpu_native_kernels_match_independent_nonwrapping_numpy_reference():
    cand = _candidate((1400.0, 1200.0, 1000.0), width=5)
    plan = make_plan(cand, time_size=9, frequency_size=4, dmsteps=5)
    rng = np.random.default_rng(42)
    data = rng.normal(size=(plan.read_stop - plan.read_start, plan.nchans))

    actual = process_cpu(cand, plan, data)
    expected = _direct_numpy_products(data, plan)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_process_cpu_profiles_read_prepare_kernels_binning_and_host_finish():
    cand = _candidate(frequencies=(1000.0, 1400.0), dm=0.0)
    plan = make_plan(cand, time_size=5, frequency_size=3, dmsteps=2)
    data = np.zeros((plan.read_stop - plan.read_start, plan.nchans), dtype=np.float32)
    profile = {}

    process_cpu(cand, plan, data, profile=profile)

    assert set(profile) == {
        "read_seconds",
        "prepare_seconds",
        "scale_seconds",
        "dedisp_seconds",
        "bin_seconds",
        "resize_seconds",
        "detrend_seconds",
        "normalize_seconds",
    }
    assert all(np.isfinite(value) and value >= 0 for value in profile.values())


def test_float64_values_beyond_float32_range_stay_finite():
    cand = _candidate(frequencies=(1000.0, 1400.0), dm=0.0)
    plan = make_plan(cand, time_size=5, frequency_size=3, dmsteps=1)
    data = np.full((plan.read_stop - plan.read_start, 2), 1e300, dtype=np.float64)
    data[2, 0] = 0

    ft, dmt = process_cpu(cand, plan, data)

    assert np.all(np.isfinite(ft))
    assert np.all(np.isfinite(dmt))


@pytest.mark.parametrize("level", [0, 100])
def test_all_rejected_rfi_channels_produce_fixed_zero_images(level):
    from types import SimpleNamespace

    from your.utils.gpu import gpu_available, process_gpu
    from your.utils.gpu_streaming import process_gpu_streaming

    data = np.full((50, 16), level, dtype=np.uint8)
    cand = SimpleNamespace(
        data=data,
        chan_freqs=np.linspace(1000.0, 900.0, 16),
        native_tsamp=0.001,
        tcand=0.025,
        width=2,
        dm=0.0,
        flag_rfi=True,
        kill_mask=np.array([False]),
        spectral_kurtosis_sigma=4,
        savgol_sigma=4,
        savgol_frequency_window=15,
        your_header=SimpleNamespace(nspectra=len(data), dtype=data.dtype, foff=-1.0),
        get_data=lambda nstart, nsamp: data[nstart : nstart + nsamp].copy(),
    )
    plan = make_plan(cand, time_size=32, frequency_size=16, dmsteps=3)
    reference = process_cpu(cand, plan)
    assert cand.rfi_mask.all()
    for image in reference:
        assert image.dtype == np.float16
        assert not np.any(image)
    if gpu_available():
        for result in (
            process_gpu(cand, plan, data=data[plan.read_start : plan.read_stop]),
            process_gpu_streaming(cand, plan, memory_budget_bytes=65536),
        ):
            assert cand.rfi_mask.all()
            for actual, expected in zip(result, reference):
                np.testing.assert_array_equal(actual, expected)
