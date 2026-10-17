from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from your.utils.candidate_pipeline import make_plan, process_cpu
from your.utils.gpu import gpu_available
from your.utils.gpu_streaming import process_gpu_streaming

pytestmark = pytest.mark.skipif(not gpu_available(), reason="requires a CUDA GPU")


def candidate(data, *, tcand=2.0, width=4, dm=10.0, flag_rfi=False):
    return SimpleNamespace(
        data=data,
        chan_freqs=np.linspace(1500, 1100, data.shape[1]),
        native_tsamp=0.02,
        tcand=tcand,
        width=width,
        dm=dm,
        kill_mask=np.array([False]),
        flag_rfi=flag_rfi,
        spectral_kurtosis_sigma=4,
        savgol_sigma=4,
        savgol_frequency_window=15,
        your_header=SimpleNamespace(nspectra=len(data), dtype=data.dtype, foff=-1.0),
        get_data=lambda nstart, nsamp: data[nstart : nstart + nsamp].copy(),
    )


@pytest.mark.parametrize("edge", [0.0, 2.0, 4.99, 9.0])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_streaming_matches_reference_at_file_edges(edge, dtype):
    rng = np.random.default_rng(45)
    data = rng.integers(90, 140, (250, 5)).astype(dtype)
    cand = candidate(data, tcand=edge)
    cand.kill_mask = np.array([False, True, False, False, False])
    plan = make_plan(cand, time_size=9, frequency_size=4, dmsteps=7)
    stats = {}
    actual = process_gpu_streaming(cand, plan, memory_budget_bytes=65536, profile=stats)
    expected = process_cpu(cand, plan)
    for got, want in zip(actual, expected):
        assert got.dtype == np.float16
        np.testing.assert_allclose(got, want, rtol=0.003, atol=0.003)
    assert stats["streamed"]
    assert (
        stats["read_bytes"]
        == (plan.read_stop - plan.read_start) * data.shape[1] * data.itemsize
    )


def test_streaming_splits_a_single_width_bin():
    rng = np.random.default_rng(23)
    data = rng.normal(100, 4, (6000, 3)).astype(np.float32)
    cand = candidate(data, tcand=60.0, width=2048)
    plan = make_plan(cand, time_size=3, frequency_size=3, dmsteps=3)
    actual = process_gpu_streaming(cand, plan, memory_budget_bytes=65536)
    expected = process_cpu(cand, plan)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=0.004, atol=0.004)


def test_streaming_preserves_optional_rfi_processing():
    rng = np.random.default_rng(52)
    data = rng.integers(70, 170, (250, 48)).astype(np.uint8)
    data[:, 20] = 250
    cand = candidate(data, flag_rfi=True)
    plan = make_plan(cand, time_size=7, frequency_size=6, dmsteps=3)
    actual = process_gpu_streaming(cand, plan, memory_budget_bytes=65536)
    expected = process_cpu(cand, plan)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=0.004, atol=0.004)


def test_streaming_bounds_long_output_accumulators(monkeypatch):
    import cupy as cp

    cand = candidate(np.zeros((1, 1), dtype=np.uint8), tcand=0, dm=0)
    cand.kill_mask = np.array([True])
    plan = make_plan(cand, time_size=65536, frequency_size=1, dmsteps=8)
    allocations = []
    original = cp.zeros

    def record_zeros(shape, *args, **kwargs):
        result = original(shape, *args, **kwargs)
        allocations.append(result.nbytes)
        return result

    monkeypatch.setattr(cp, "zeros", record_zeros)
    stats = {}
    result = process_gpu_streaming(
        cand, plan, memory_budget_bytes=1024**2, profile=stats
    )
    assert max(allocations) <= 1024**2 // 8
    assert result[0].shape == (65536, 1)
    assert result[1].shape == (8, 65536)
    assert not np.any(result[0])
    assert not np.any(result[1])


def test_streaming_preserves_large_finite_float64_structure():
    values = (
        np.array(
            [[1, -1], [2, -2], [-1, 1], [-2, 2], [1, 2], [2, 1], [-2, -1], [-1, -2]],
            dtype=np.float64,
        )
        * 1e40
    )
    cand = candidate(values, tcand=0.08, width=1, dm=0)
    plan = make_plan(cand, time_size=8, frequency_size=2, dmsteps=3)
    actual = process_gpu_streaming(cand, plan, memory_budget_bytes=65536)
    expected = process_cpu(cand, plan)
    for got, want in zip(actual, expected):
        assert np.max(np.abs(got)) > 0
        np.testing.assert_allclose(got, want, rtol=0.003, atol=0.003)


def test_streaming_preserves_cpu_rfi_masked_array_semantics_for_zero_channels():
    data = np.random.default_rng(18).integers(90, 140, (18, 48)).astype(np.float64)
    data[:, ::3] = 0

    def rfi_candidate():
        cand = candidate(data, tcand=0.01, width=1, dm=0, flag_rfi=True)
        cand.native_tsamp = 0.001
        cand.chan_freqs = np.linspace(1000, 1047, 48)
        cand.spectral_kurtosis_sigma = 4
        cand.savgol_sigma = 0
        cand.your_header.foff = 1.0
        cand.your_header.tsamp = 0.001
        return cand

    expected_cand = rfi_candidate()
    plan = make_plan(expected_cand, time_size=32, frequency_size=48, dmsteps=3)
    expected = process_cpu(expected_cand, plan)
    expected_mask = expected_cand.rfi_mask.copy()
    assert np.all(expected_mask[::3])
    assert expected_mask[10]

    actual_cand = rfi_candidate()
    actual = process_gpu_streaming(actual_cand, plan, memory_budget_bytes=65536)
    np.testing.assert_array_equal(actual_cand.rfi_mask, expected_mask)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=0.003, atol=0.003)


@pytest.mark.parametrize(
    "kill_mask",
    [np.array([], dtype=bool), False, np.array([False])],
)
def test_streaming_accepts_cpu_empty_false_kill_masks(kill_mask):
    data = np.arange(40, dtype=np.float32).reshape(10, 4) + 1
    expected_cand = candidate(data, tcand=0.1, width=2, dm=0)
    expected_cand.kill_mask = kill_mask
    plan = make_plan(expected_cand, time_size=4, frequency_size=4, dmsteps=3)
    expected = process_cpu(expected_cand, plan)

    actual_cand = candidate(data, tcand=0.1, width=2, dm=0)
    actual_cand.kill_mask = kill_mask
    actual = process_gpu_streaming(actual_cand, plan, memory_budget_bytes=65536)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=0.003, atol=0.003)


def test_streaming_uses_decoded_float32_dtype_for_calibrated_int32_fits(tmp_path):
    from astropy.io import fits

    from your.candidate import Candidate
    from your.utils.candidate_pipeline import read_plan_data

    source_path = Path(__file__).parent / "data" / "small.fits"
    fits_path = tmp_path / "scaled32.fits"
    with fits.open(source_path) as source:
        table = source["SUBINT"]
        values = np.random.default_rng(217).integers(
            50, 220, table.data["DATA"].shape, dtype=np.int32
        )
        values[:, 310:315, :, 20:100] += 100
        data_column = fits.Column(
            name="DATA",
            format=f"{values[0].size}J",
            dim=table.columns["DATA"].dim,
            array=values,
        )
        calibrated_table = fits.BinTableHDU.from_columns(
            [column for column in table.columns if column.name != "DATA"]
            + [data_column],
            header=table.header,
            name="SUBINT",
        )
        calibrated_table.header["NBITS"] = 32
        calibrated_table.data["DAT_SCL"][:] = 1 / 512
        calibrated_table.data["DAT_OFFS"][:] = 0
        calibrated_table.data["DAT_WTS"][:] = 1
        fits.HDUList([source[0].copy(), calibrated_table]).writeto(fits_path)

    with Candidate(str(fits_path), dm=0.0, tcand=0.4, width=2) as cand:
        plan = make_plan(cand, time_size=32, frequency_size=16, dmsteps=8)
        raw = read_plan_data(cand, plan)
        assert np.dtype(cand.your_header.dtype) == np.dtype(np.int32)
        assert raw.dtype == np.dtype(np.float32)

        expected = process_cpu(cand, plan, data=raw)
        stats = {}
        actual = process_gpu_streaming(
            cand, plan, memory_budget_bytes=65536, profile=stats
        )

    assert all(np.count_nonzero(product) for product in expected)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, rtol=0.003, atol=0.003)
