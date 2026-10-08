import pytest

from your.utils.misc import *
from your.utils.misc import _anti_aliased_shrink, _resize


def test_pad_along_axis():
    d = np.random.random((10, 10))
    d_end = pad_along_axis(d, 12, "end", 0, mode="median")
    assert d_end.shape[0] == 12
    assert (d_end[11, :] - np.median(d, axis=0)).sum() == 0

    d_start = pad_along_axis(d, 12, "start", 1, mode="median")
    assert d_start.shape[1] == 12
    assert (d_start[:, 0] - np.median(d, axis=1)).sum() == 0

    d_both = pad_along_axis(d, 12, "both", 1, mode="median")
    assert d_both.shape[1] == 12
    assert (d_both[:, 0] - np.median(d, axis=1)).sum() == 0
    assert (d_both[:, -1] - np.median(d, axis=1)).sum() == 0

    d_same = pad_along_axis(d, 9, "end", 0, mode="median")
    assert (d_same - d).sum() == 0


def test_crop():
    d = np.random.random((10, 10))
    try:
        crop(d, 2, 20, 0)
    except OverflowError:
        pass

    d2 = crop(d, 2, 4, 0)
    assert d2.shape[0] == 4

    d3 = crop(d, 2, 4, 1)
    assert d3.shape[1] == 4

    d_same = crop(d, 2, 10, 1)
    assert (d_same - d).sum() == 0


def test_myencoder():
    c = MyEncoder()
    t = {}
    t["1"] = int(1)
    t["2"] = float(2)
    t["3"] = np.array([1, 2])
    t["4"] = 1
    t["5"] = "a"

    with open("test.json", "w") as fp:
        json.dump(t, fp, cls=MyEncoder, indent=4)

    assert os.path.isfile("test.json")
    os.remove("test.json")


def test_read_buffer_reused_and_grown():
    from your.utils.misc import ReadBuffer

    buffer = ReadBuffer(granule=1024)
    first = buffer(100)
    assert first.dtype == np.uint8 and first.size == 1024
    assert buffer(1024) is first
    grown = buffer(1025)
    assert grown.size == 2048 and grown is not first
    assert buffer(10) is grown


def _bits(a):
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


@pytest.mark.parametrize("factor", [2, 3, 4, 5, 8, 16, 17])
@pytest.mark.parametrize(
    "make",
    [
        lambda rng, shape: rng.standard_normal(shape),
        # all one sign, so skimage's clip does or does not take in zero
        lambda rng, shape: rng.uniform(100, 200, shape),
        lambda rng, shape: -rng.uniform(1, 2, shape),
        lambda rng, shape: np.full(shape, 150.0),
        lambda rng, shape: np.full(shape, -0.0),
        lambda rng, shape: (
            rng.standard_normal(shape) * 10.0 ** rng.integers(-8, 8, shape)
        ),
    ],
)
def test_resize_matches_skimage_bit_for_bit(factor, make):
    from skimage.transform import resize

    rng = np.random.default_rng(factor)
    # rows not a multiple of the kernel's block of rows
    data = make(rng, (13, factor * 24)).astype(np.float32)
    kwargs = dict(anti_aliasing=True, mode="constant")
    assert _anti_aliased_shrink(data, 24) is not None
    want = resize(data, (13, 24), **kwargs)
    got = _resize(data, 24, 1, **kwargs)
    assert got.dtype == want.dtype and got.shape == want.shape
    assert np.array_equal(_bits(got), _bits(want))


def test_resize_falls_back_to_skimage():
    from skimage.transform import resize

    rng = np.random.default_rng(0)
    kwargs = dict(anti_aliasing=True, mode="constant")
    nan = rng.standard_normal((8, 64)).astype(np.float32)
    nan[2, 10] = np.nan
    inf = rng.standard_normal((8, 64)).astype(np.float32)
    inf[3, 40] = np.inf
    for data, size in [
        (nan, 16),
        (inf, 16),
        (rng.standard_normal((8, 60)).astype(np.float32), 16),
        (rng.standard_normal((8, 64)), 16),
    ]:
        assert _anti_aliased_shrink(data, size) is None
        want = resize(data, (8, size), **kwargs)
        got = _resize(data, size, 1, **kwargs)
        assert np.array_equal(_bits(got), _bits(want))
