import os

import pytest

from your.candidate import Candidate
from your.utils import gpu
from your.utils.gpu import *
from your.utils.misc import crop

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
_install_dir = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dedisperse():
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=475.28400,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    cand.dedisperse(target="GPU")
    g_dedisp = cand.dedispersed
    cand.dedisperse(target="CPU")
    c_dedisp = cand.dedispersed
    assert np.isclose(np.mean(g_dedisp - c_dedisp), 0, atol=1)
    assert np.isclose(np.max(cand.dedispersed.T.sum(0)), 47527, atol=1)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dmt():
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=10,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    cand.dmtime(target="GPU")
    g_dmt = cand.dmt
    cand.dmtime(target="CPU")
    c_dmt = cand.dmt
    assert cand.dmt.shape[0] == 256
    assert np.isclose(np.mean(g_dmt - c_dmt), 0, atol=1)
    assert np.max(g_dmt - c_dmt) / np.max(g_dmt) < 0.05


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dedisp_dmt_crop():
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=10,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    cand = gpu_dedisp_and_dmt_crop(cand)
    g_dmt = cand.dmt
    g_dedisp = cand.dedispersed
    assert cand.dedispersed.shape[0] == 256
    assert cand.dmt.shape[1] == 256

    cand.dedisperse()
    cand.dmtime()
    crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - 256 // 2
    crop_start_sample_dmt = cand.dmt.shape[1] // 2 - 256 // 2
    c_dmt = crop(cand.dmt, crop_start_sample_dmt, 256, 1)
    c_dedisp = crop(cand.dedispersed, crop_start_sample_ft, 256, 0)

    assert np.isclose(np.sum(g_dmt - c_dmt), 0, atol=1)
    assert np.isclose(np.sum(g_dedisp - c_dedisp), 0, atol=1)


def _dmt_cand(dm):
    file = os.path.join(_install_dir, "data/28.fil")
    cand = Candidate(
        fp=file,
        dm=dm,
        tcand=2.0288800,
        width=2,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    return cand


# dm 10 collapses the band to 0.03 spans per channel, dm 475 to 0.82, so the
# two sit either side of the default crossover and take a kernel each
@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dm", [10, 475.284])
def test_gpu_dmt_kernels_agree(dm):
    """Which kernel runs is a speed choice, so it must not change the plane."""
    runs = gpu_dmt(_dmt_cand(dm), max_run_fraction=1.0).dmt
    channels = gpu_dmt(_dmt_cand(dm), max_run_fraction=0.0).dmt
    assert np.array_equal(runs, channels)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dm", [10, 475.284])
def test_gpu_dmt_matches_cpu(dm):
    cand = _dmt_cand(dm)
    cand.dmtime(target="CPU")
    cpu = cand.dmt.copy()
    assert np.array_equal(gpu_dmt(_dmt_cand(dm)).dmt, cpu)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dmt_picks_each_kernel():
    for dm, expected in ((10, True), (475.284, False)):
        cand = _dmt_cand(dm)
        freqs = np.asarray(cand.chan_freqs, dtype=np.float64)
        dms = cand.dm + np.linspace(-cand.dm, cand.dm, 256)
        _, nruns = run_edges(delay_table(freqs, float(cand.your_header.tsamp), dms))
        fraction = nruns.mean() / len(freqs)
        assert bool(fraction <= 0.6) is expected


# candmaker's workers call gpu_dedisp_and_dmt_crop once per candidate in a
# long-lived process, so the second and later calls are the ones that matter
@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dedisp_dmt_crop_repeatable():
    """No call may inherit the previous call's atomic sums."""
    first = gpu_dedisp_and_dmt_crop(_dmt_cand(475.284))
    for _ in range(3):
        again = gpu_dedisp_and_dmt_crop(_dmt_cand(475.284))
        assert np.array_equal(again.dedispersed, first.dedispersed)
        assert np.array_equal(again.dmt, first.dmt)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_gpu_dedisp_dmt_crop_frees_memory():
    """Device arrays are released on return, not left for the next candidate."""
    # numba queues frees and only flushes past 10 arrays or 20% of the card;
    # anything still queued here is held into the next candidate
    gpu_dedisp_and_dmt_crop(_dmt_cand(475.284))
    assert len(pending_frees()._cons) == 0


def _window_reference(cand, width=256):
    """Decimate the whole chunk, then crop, in numpy: what the atomic kernels did."""
    tdf = 1 if cand.width < 3 else cand.width // 2
    data = np.asarray(cand.data).astype(cand.your_header.dtype).T.astype(np.int64)
    nchans, nsamples = data.shape
    fdf = nchans // 256
    rows, ncols = nchans // fdf, nsamples // tdf
    cols = (nsamples // tdf // 2 - width // 2 + np.arange(width)) % ncols

    ft_delays = cuda.device_array(nchans, dtype=np.int64)
    dedisp_delays[math.ceil(nchans / 128), 128](
        cuda.to_device(np.array(cand.chan_freqs, dtype=np.float32)),
        float(cand.dm),
        float(cand.your_header.tsamp),
        ft_delays,
    )
    shifted = np.stack(
        [np.roll(data[ch], -d) for ch, d in enumerate(ft_delays.copy_to_host())]
    )
    ft = shifted[: rows * fdf, : ncols * tdf].reshape(rows, fdf, ncols, tdf).sum((1, 3))

    delays = dmt_delays(cand, np.linspace(0, 2 * cand.dm, 256))
    dmt = np.stack(
        [
            sum(np.roll(data[ch], -delays[ch, kk]) for ch in range(nchans))[
                : ncols * tdf
            ]
            .reshape(ncols, tdf)
            .sum(1)
            for kk in range(256)
        ]
    )
    return ft[:, cols].T.astype(np.float32), dmt[:, cols].astype(np.float32)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "dm, width, block_from",
    [
        (10, 2, 128),
        (475.284, 2, 128),
        (475.284, 8, 128),
        (475.284, 32, 128),
        (475.284, 256, 128),
        (475.284, 512, 128),
        # force the block prefix sums on narrow columns too, where most spans
        # have partial blocks at both ends or none whole
        (10, 8, 1),
        (475.284, 32, 1),
        (475.284, 64, 1),
        (475.284, 128, 1),
    ],
)
def test_gpu_dedisp_dmt_crop_matches_reference(dm, width, block_from):
    """One thread per kept cell gives exactly the decimate-then-crop sums."""
    cand = Candidate(
        fp=os.path.join(_install_dir, "data/28.fil"),
        dm=dm,
        tcand=2.0288800,
        width=width,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    ft, dmt = _window_reference(cand)
    cand = gpu_dedisp_and_dmt_crop(cand, block_from=block_from)
    assert np.array_equal(cand.dedispersed, ft)
    assert np.array_equal(cand.dmt, dmt)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_pinned_read_buffer_reused_and_grown():
    buffer = PinnedReadBuffer(0, granule=2**20)
    first = buffer(10)
    assert first.size == 2**20 and first.dtype == np.uint8
    assert buffer(2**20) is first
    grown = buffer(2**20 + 1)
    assert grown.size == 2**21

    cand = Candidate(
        fp=os.path.join(_install_dir, "data/28.fil"),
        dm=475.284,
        tcand=2.0288800,
        width=32,
        label=-1,
        snr=16.8128,
        min_samp=256,
        device=0,
    )
    cand.get_chunk()
    ft, dmt = _window_reference(cand)
    calls = []
    cand.read_buffer = lambda nbytes: calls.append(nbytes) or buffer(nbytes)
    cand.get_chunk()
    assert calls
    cand = gpu_dedisp_and_dmt_crop(cand)
    assert np.array_equal(cand.dedispersed, ft)
    assert np.array_equal(cand.dmt, dmt)


@pytest.mark.skipif(not cuda.is_available(), reason="requires a GPU")
def test_pinned_read_buffer_falls_back_when_it_cannot_pin(monkeypatch):
    def refuse(*args, **kwargs):
        raise CudaAPIError(2, "CUDA_ERROR_OUT_OF_MEMORY")

    buffer = PinnedReadBuffer(0, granule=2**20)
    monkeypatch.setattr(gpu, "portable_pinned_array", refuse)
    out = buffer(10)
    assert out.size == 10 and buffer.buffer is None
    monkeypatch.undo()
    assert buffer(10).size == 2**20


@pytest.mark.skipif(
    not cuda.is_available() or len(cuda.gpus) < 2, reason="requires two GPUs"
)
def test_pinned_read_buffer_serves_every_gpu():
    """One buffer, allocated in one GPU's context, feeds uploads to either GPU."""
    buffer = PinnedReadBuffer(0, granule=2**20)
    first = None
    for device in (1, 0, 1):
        cand = Candidate(
            fp=os.path.join(_install_dir, "data/28.fil"),
            dm=475.284,
            tcand=2.0288800,
            width=32,
            label=-1,
            snr=16.8128,
            min_samp=256,
            device=device,
        )
        cand.get_chunk()
        ft, dmt = _window_reference(cand)
        calls = []
        cand.read_buffer = lambda nbytes: calls.append(nbytes) or buffer(nbytes)
        cand.get_chunk()
        assert calls
        first = buffer.buffer if first is None else first
        assert buffer.buffer is first
        cand = gpu_dedisp_and_dmt_crop(cand, device=device)
        assert np.array_equal(cand.dedispersed, ft)
        assert np.array_equal(cand.dmt, dmt)
