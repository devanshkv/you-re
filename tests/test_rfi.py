import os

import numpy as np
import pytest

from your import Your
from your.candidate import Candidate
from your.utils.rfi import *


@pytest.fixture(scope="session", autouse=True)
def fil_file():
    _install_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(_install_dir, "data/28.fil")


def test_calc_N():
    assert calc_N(channel_bandwidth=-1, tsamp=0.001) == 1000


def test_savgol_filter_on_noise():
    d = np.random.random(100)
    m = savgol_filter(d, -1, 15, 4)
    assert m.sum() == 0


def test_savgol_filter_on_impulse():
    d = np.random.random(100)
    d = d + np.sort(np.random.random(100)) ** 5
    d[10:12] = 5
    m = savgol_filter(d, -1, 15, 4)
    assert m[10:12].sum() == 2


def test_spectral_kurtosis_on_noise():
    d = np.random.random(100)
    sk = spectral_kurtosis(d, 1, None)
    assert np.isclose(sk, 1, atol=0.1)


def test_spectral_kurtosis_on_impulse():
    d = np.random.random(100)
    d[10:12] = 5
    sk = spectral_kurtosis(d, 1, None)
    assert np.isclose(sk, 1, atol=0.1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
def test_sk_rejection_mask_ignores_masked_zero_channels(dtype):
    rng = np.random.default_rng(42)
    data = rng.integers(1, 100, (256, 336)).astype(dtype)
    data[:, 136:] = 0

    sk = spectral_kurtosis(data, N=1000)
    actual = sk_rejection_mask(sk, sigma=4)
    expected = sk_rejection_mask(sk[:136], sigma=4)

    assert type(actual) is np.ndarray
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual[:136], expected)
    assert actual[136:].all()
    assert np.any(~actual[:136])


def test_sk_rejection_mask_rejects_all_invalid_values():
    assert sk_rejection_mask(np.ma.masked_all(3), sigma=4).all()


def test_killed_channel_majority_does_not_reject_all_candidate_channels(fil_file):
    kill_mask = np.r_[np.zeros(136, dtype=bool), np.ones(200, dtype=bool)]
    with Candidate(
        fil_file,
        dm=10,
        tcand=0.1,
        width=2,
        kill_mask=kill_mask,
        flag_rfi=True,
    ) as cand:
        cand.make_products(time_size=64, frequency_size=32, dmsteps=8)

    assert np.any(~cand.rfi_mask[:136])
    assert np.all(cand.rfi_mask[136:])
    assert np.count_nonzero(cand.dedispersed)
    assert np.count_nonzero(cand.dmt)


def test_incorrect_sigmas(fil_file):
    your_object = Your(fil_file)
    data = your_object.get_data(0, 1024)
    with pytest.raises(ValueError):
        sk_sg_filter(data, your_object, -1, 1, 1)

    with pytest.raises(ValueError):
        sk_sg_filter(data, your_object, 3, 15, -1)

    with pytest.raises(ValueError):
        sk_sg_filter(data, your_object, 0, 15, 0)

    with pytest.raises(ValueError):
        sk_sg_filter(data, foff=1.0)

    with pytest.raises(ValueError):
        sk_sg_filter(data, tsamp=1.0)
