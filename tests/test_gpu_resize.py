from types import SimpleNamespace

import numpy as np
import pytest
from skimage.transform import resize

from your.utils import candidate_pipeline
from your.utils.candidate_pipeline import finish_products
from your.utils.gpu import _resize_binned_ft, estimate_process_gpu_bytes, gpu_available

pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="requires CuPy and a CUDA GPU"
)


@pytest.mark.parametrize(
    ("kind", "shape", "frequency_size"),
    [
        ("downsample", (19, 32), 9),
        ("downsample_to_one", (13, 29), 1),
        ("upsample", (17, 7), 12),
        ("negative_constant", (23, 11), 4),
        ("nonfinite", (20, 13), 5),
    ],
)
def test_gpu_ft_resize_matches_skimage(kind, shape, frequency_size):
    import cupy as cp

    rng = np.random.default_rng(113)
    if kind == "negative_constant":
        ft = np.full(shape, -3.25, dtype=np.float32)
    else:
        ft = rng.normal(size=shape).astype(np.float32)
    if kind == "nonfinite":
        ft[2, 5] = np.nan
        ft[6, 9] = np.inf
        ft[12, 0] = -np.inf

    safe_ft = np.nan_to_num(
        ft,
        copy=True,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=-np.finfo(np.float32).max,
    )
    expected = resize(
        safe_ft.T,
        (frequency_size, shape[0]),
        anti_aliasing=True,
        mode="constant",
        preserve_range=True,
    ).astype(np.float32, copy=False)
    actual = _resize_binned_ft(cp, cp.asarray(ft), frequency_size).get()

    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_finish_products_accepts_pre_resized_gpu_ft(monkeypatch):
    import cupy as cp

    rng = np.random.default_rng(114)
    time_size, nchans, frequency_size, dmsteps = 29, 17, 7, 5
    plan = SimpleNamespace(
        time_size=time_size,
        nchans=nchans,
        frequency_size=frequency_size,
        dmsteps=dmsteps,
        output_dtype=np.dtype(np.float32),
    )
    ft = rng.normal(size=(time_size, nchans)).astype(np.float32)
    dmt = rng.normal(size=(dmsteps, time_size)).astype(np.float32)
    expected = finish_products(ft, dmt, plan)
    resized = _resize_binned_ft(cp, cp.asarray(ft), frequency_size).get()

    def unexpected_cpu_resize(*args, **kwargs):
        raise AssertionError("pre-resized FT should bypass CPU resize")

    monkeypatch.setattr(candidate_pipeline, "resize", unexpected_cpu_resize)
    actual = finish_products(resized, dmt, plan, ft_resized=True)

    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("kind", ["subnormal", "tiny_normal", "mixed_tiny_normal"])
def test_process_gpu_resize_sensitive_binned_ft_uses_cpu_fallback(monkeypatch, kind):
    import cupy as cp

    from your.utils import gpu
    from your.utils.candidate_pipeline import make_plan

    frequencies = np.linspace(1500.0, 1200.0, 32)
    cand = SimpleNamespace(
        dm=0.0,
        tcand=8.0,
        width=2,
        native_tsamp=1.0,
        chan_freqs=frequencies,
        your_header=SimpleNamespace(nspectra=16, foff=frequencies[1] - frequencies[0]),
        kill_mask=np.array([False]),
        flag_rfi=False,
    )
    plan = make_plan(
        cand, time_size=19, frequency_size=9, dmsteps=3, output_dtype="float32"
    )
    assert plan.time_bin == 1

    if kind == "subnormal":
        bits = np.resize(
            np.array([1, 3, 1, 2, 4, 1, 3, 2], dtype=np.uint32), plan.time_size
        )
        tiny_time = bits.view(np.float32)
        ft_binned = np.repeat(tiny_time[:, None], plan.nchans, axis=1)
    else:
        rng = np.random.default_rng(113)
        ft_binned = rng.uniform(1, 3, (plan.time_size, plan.nchans)).astype(
            np.float32
        ) * np.float32(1.1754944e-38)
    if kind == "mixed_tiny_normal":
        ft_binned[:, :4] = np.linspace(1.0, 2.0, plan.time_size)[:, None]
        ft_binned[0, 10] = np.nan
        ft_binned[1, 11] = np.inf
        ft_binned[2, 12] = -np.inf
    if kind == "tiny_normal":
        assert np.all(np.isfinite(ft_binned))
        assert np.all(np.abs(ft_binned) >= np.finfo(np.float32).tiny)
    dmt_binned = (
        np.linspace(-1.0, 1.0, plan.dmsteps * plan.time_size)
        .reshape(plan.dmsteps, plan.time_size)
        .astype(np.float32)
    )
    data = np.full((plan.read_stop - plan.read_start, plan.nchans), 2, dtype=np.float32)
    resident = cp.ascontiguousarray(cp.asarray(data.T))

    module, original_functions = gpu._kernel_functions(cp)
    functions = dict(original_functions)
    functions["bin_ft"] = lambda grid, block, args, *, stream: cp.copyto(
        args[1], cp.asarray(ft_binned)
    )
    functions["bin_dmt"] = lambda grid, block, args, *, stream: cp.copyto(
        args[1], cp.asarray(dmt_binned)
    )
    for name in functions:
        if name.startswith("dedisp_ft_") or name.startswith("dmt_tiled_"):
            functions[name] = lambda *args, **kwargs: None
    monkeypatch.setattr(gpu, "_kernel_functions", lambda cp: (module, functions))

    native_resize = gpu._resize_binned_ft
    sources = []

    def capture_source(cp, source, frequency_size):
        sources.append(source)
        return native_resize(cp, source, frequency_size)

    monkeypatch.setattr(gpu, "_resize_binned_ft", capture_source)
    profile = {}
    actual = gpu.process_gpu(
        cand,
        plan,
        resident=resident,
        resident_start=plan.read_start,
        profile=profile,
    )

    safe_ft = np.nan_to_num(
        ft_binned,
        copy=True,
        nan=0.0,
        posinf=np.finfo(np.float32).max,
        neginf=-np.finfo(np.float32).max,
    )
    np.testing.assert_array_equal(sources[0].get(), safe_ft)
    expected = finish_products(ft_binned, dmt_binned, plan)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert profile["resize_fallback_s"] >= 0


def test_gpu_resize_workspace_estimate_counts_filter_and_output():
    plan = SimpleNamespace(
        read_start=0,
        read_stop=64,
        nchans=4096,
        dmsteps=5,
        pad_before=1,
        pad_after=1,
        native_count=32,
        time_size=8,
        frequency_size=1,
    )
    resized = estimate_process_gpu_bytes(plan, np.float32)
    same_size = estimate_process_gpu_bytes(
        SimpleNamespace(**{**vars(plan), "frequency_size": plan.nchans}), np.float32
    )

    sigma = (plan.nchans / plan.frequency_size - 1) / 2
    radius = int(4 * sigma + 0.5)
    expected_resize_workspace = plan.time_size * (
        plan.frequency_size + plan.nchans
    ) * 4 + 12 * (2 * radius + 1)
    expected_guard_scratch = ((plan.time_size * plan.nchans + 31) // 32) * 4
    assert resized - same_size == expected_resize_workspace + expected_guard_scratch
