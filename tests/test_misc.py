import pytest

from your.utils.misc import *
from your.utils.misc import _anti_aliased_shrink, _decimate


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

    d_odd = np.arange(6).reshape(3, 2)
    d_odd_axis_0 = pad_along_axis(d_odd, 6, "both", 0, mode="median")
    assert d_odd_axis_0.shape == (6, 2)
    assert np.array_equal(d_odd_axis_0[1:-2], d_odd)

    d_odd_axis_1 = pad_along_axis(d_odd, 5, "both", 1, mode="median")
    assert d_odd_axis_1.shape == (3, 5)
    assert np.array_equal(d_odd_axis_1[:, 1:-2], d_odd)

    d_same = pad_along_axis(d, 9, "end", 0, mode="median")
    assert (d_same - d).sum() == 0


def test_decimate_with_odd_centered_padding():
    result = _decimate(np.ones((9, 3)), 2, 0, pad=True, loc="both", mode="median")

    assert result.shape == (5, 3)
    assert np.array_equal(result, np.ones((5, 3)))


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


def _bits(array):
    return array.view(np.uint32 if array.dtype == np.float32 else np.uint64)


@pytest.mark.parametrize("factor", [2, 3, 4, 5, 8, 17])
@pytest.mark.parametrize(
    "make",
    [
        lambda rng, shape: rng.standard_normal(shape),
        lambda rng, shape: rng.uniform(100, 200, shape),
        lambda rng, shape: -rng.uniform(1, 2, shape),
        lambda rng, shape: np.full(shape, 150.0),
        lambda rng, shape: np.full(shape, -0.0),
    ],
)
def test_anti_aliased_shrink_matches_skimage_bit_for_bit(factor, make):
    from skimage.transform import resize

    rng = np.random.default_rng(factor)
    data = make(rng, (13, factor * 24)).astype(np.float32)
    want = resize(data, (13, 24), anti_aliasing=True, mode="constant")
    got = _anti_aliased_shrink(data, 24)
    assert got is not None
    assert got.dtype == want.dtype and got.shape == want.shape
    assert np.array_equal(_bits(got), _bits(want))


def test_anti_aliased_shrink_falls_back_for_unsupported_inputs():
    rng = np.random.default_rng(0)
    nan = rng.standard_normal((8, 64)).astype(np.float32)
    nan[2, 10] = np.nan
    inf = rng.standard_normal((8, 64)).astype(np.float32)
    inf[3, 40] = np.inf
    data = rng.standard_normal((8, 64)).astype(np.float32)

    for values, size in [
        (nan, 16),
        (inf, 16),
        (rng.standard_normal((8, 60)).astype(np.float32), 16),
        (rng.standard_normal((8, 64)), 16),
        (data, 16.5),
        (data, 128),
    ]:
        assert _anti_aliased_shrink(values, size) is None


def test_native_resize_empty_rows_do_not_allocate_channel_scratch():
    from your._rust import gaussian_decimate

    data = np.empty((0, 2**60), dtype=np.float32)
    output = gaussian_decimate(data, np.array([1.0], dtype=np.float64), 2**60)
    assert output.shape == (0, 1)
    assert output.dtype == np.float32
