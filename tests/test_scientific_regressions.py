from types import SimpleNamespace

import numpy as np
import pytest

from your import Your
from your.candidate import Candidate


def _reader_for(data, npol):
    class Format:
        @staticmethod
        def get_data(reader, nstart, nsamp, pol=0, npoln=1):
            return data.copy()

    reader = Your.__new__(Your)
    reader._closed = False
    reader.format = "fits"
    reader.formatclass = Format
    reader.nchans = data.shape[-1]
    reader.your_header = SimpleNamespace(
        time_decimation_factor=1,
        frequency_decimation_factor=1,
        npol=npol,
        nbits=32,
        dtype=np.float32,
    )
    return reader


def test_time_decimation_averages_adjacent_2d_samples():
    data = np.array([[0, 10], [100, 110], [200, 210], [300, 310]], dtype=np.float32)
    reader = _reader_for(data, npol=1)

    result = reader.get_data(0, 4, time_decimation_factor=2)

    np.testing.assert_array_equal(result, [[50, 60], [250, 260]])


def test_time_decimation_averages_adjacent_four_polarization_samples():
    data = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    reader = _reader_for(data, npol=4)

    result = reader.get_data(0, 4, time_decimation_factor=2, npoln=4)

    expected = data.reshape(2, 2, 4, 2).mean(axis=1)
    np.testing.assert_array_equal(result, expected)


def test_get_snr_masks_peak_at_start():
    candidate = Candidate.__new__(Candidate)
    candidate.width = 4
    time_series = np.array([10, 20, 4, 8, 12, 16, 3], dtype=np.float32)

    snr = candidate.get_snr(time_series)

    noise = time_series[3:]
    expected = (time_series.max() - noise.mean()) / noise.std()
    np.testing.assert_allclose(snr, expected)


@pytest.mark.parametrize("writeable", [True, False])
def test_get_snr_does_not_mutate_supplied_input(writeable):
    candidate = Candidate.__new__(Candidate)
    candidate.width = 2
    time_series = np.array([1, 2, 10, 4, 5], dtype=np.float32)
    original = time_series.copy()
    time_series.setflags(write=writeable)

    candidate.get_snr(time_series)

    np.testing.assert_array_equal(time_series, original)
