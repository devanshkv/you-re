from types import SimpleNamespace

import numpy as np
import pytest

from your.utils.candidate_contract import decoded_sample_dtype, parse_kill_mask


@pytest.mark.parametrize(
    "mask",
    [np.array([], dtype=bool), np.array([False]), np.array(False)],
)
def test_empty_and_false_kill_masks_disable_no_channels(mask):
    np.testing.assert_array_equal(parse_kill_mask(mask, 3), [False, False, False])


def test_kill_mask_parser_accepts_channel_masks_and_integer_indices():
    np.testing.assert_array_equal(
        parse_kill_mask(np.array([False, True, False]), 3), [False, True, False]
    )
    np.testing.assert_array_equal(
        parse_kill_mask(np.array([-1, 0]), 3), [True, False, True]
    )


@pytest.mark.parametrize(
    "mask",
    [np.array([True]), np.array([True, False]), np.array([[0]], dtype=np.int16)],
)
def test_kill_mask_parser_rejects_invalid_shapes(mask):
    with pytest.raises(ValueError, match="kill_mask"):
        parse_kill_mask(mask, 3)


def test_decoded_sample_dtype_probes_nonempty_reader_data():
    class Reader:
        your_header = SimpleNamespace(
            nspectra=4, dtype=np.int16, format="fits", nbits=32
        )

        def __init__(self):
            self.calls = []

        def get_data(self, nstart, nsamp):
            self.calls.append((nstart, nsamp))
            return np.zeros((nsamp, 2), dtype=np.float32)

    reader = Reader()

    assert decoded_sample_dtype(reader) == np.dtype(np.float32)
    assert reader.calls == [(0, 1)]


def test_decoded_sample_dtype_uses_zero_length_fallbacks_without_reading():
    class Reader:
        def __init__(self, header):
            self.your_header = header

        def get_data(self, *_args):
            raise AssertionError("empty readers must not be probed")

    psrfits = Reader(
        SimpleNamespace(nspectra=0, dtype=np.int32, format="fits", nbits=32)
    )
    generic = Reader(
        SimpleNamespace(nspectra=0, dtype=np.uint16, format="sigproc", nbits=16)
    )

    assert decoded_sample_dtype(psrfits) == np.dtype(np.float32)
    assert decoded_sample_dtype(generic) == np.dtype(np.uint16)
