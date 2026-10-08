import logging

import numpy as np

from your.utils.math import closest_number

logger = logging.getLogger(__name__)
import argparse
import functools
import json
import os

from scipy import ndimage as _ndimage
from skimage.transform import resize

from your._rust import gaussian_decimate as _rust_gaussian_decimate


class ReadBuffer:
    """
    One host buffer, reused for every chunk put in it and grown when a chunk
    is larger than any before it, so a worker making one candidate after
    another does not allocate, fault in and zero a fresh array for each.

    Each use overwrites the last, so what is in it is only good until the
    next: meant for a worker that makes one candidate at a time.

    Args:
        granule (int): allocation size is rounded up to a multiple of this
    """

    def __init__(self, granule=64 * 2**20):
        self.granule = granule
        self.buffer = None

    def __call__(self, nbytes):
        if self.buffer is None or self.buffer.size < nbytes:
            self.buffer = None
            size = -(-nbytes // self.granule) * self.granule
            self.buffer = np.empty(size, dtype=np.uint8)
        return self.buffer


def _decimate(data, decimate_factor, axis, pad=False, **kwargs):
    """
    Decimate an input array by an input factor. Optionally padding can also be done.

    Args:
        data (np.ndarray): data array to decimat
        decimate_factor (int): number of samples to combine
        axis (int): axis along which decimation is to be done
        pad (bool): Whether to apply padding if the data axis length is not a multiple of decimation factor
        **kwargs: arguments of padding

    Returns:
        np.ndarray: Decimated array

    """
    if data.shape[axis] % decimate_factor and pad is True:
        logger.info(f"padding along axis {axis}")
        pad_width = closest_number(data.shape[axis], decimate_factor)
        data = pad_along_axis(data, data.shape[axis] + pad_width, axis=axis, **kwargs)
    elif data.shape[axis] % decimate_factor and pad is False:
        raise AttributeError(
            "Axis length should be a multiple of decimate_factor. Use pad=True to force decimation"
        )

    if axis:
        return data.reshape(
            int(data.shape[0]),
            int(data.shape[1] // decimate_factor),
            int(decimate_factor),
        ).mean(2)
    else:
        return data.reshape(
            int(data.shape[0] // decimate_factor),
            int(decimate_factor),
            int(data.shape[1]),
        ).mean(1)


def _resize(data, size, axis, **kwargs):
    """
    Resize an input array to a required size

    Args:
        data (np.ndarray): data array to resize
        size (int): required size of the axis
        axis (int): axis long which resizing is to be done
        **kargs: arguments for skimage.transform resize function

    Returns:
        np.ndarray: Resized array

    """
    if axis:
        if kwargs == {"anti_aliasing": True, "mode": "constant"}:
            # the candmaker's frequency resize
            out = _anti_aliased_shrink(data, size)
            if out is not None:
                return out
        return resize(data, (data.shape[0], size), **kwargs)
    else:
        return resize(data, (size, data.shape[1]), **kwargs)


@functools.lru_cache(maxsize=8)
def _half_gaussian(factor):
    """
    The float64 Gaussian, centre first, that skimage's anti-aliasing applies
    to an axis it shrinks `factor` times: read off SciPy's own filter as its
    response to a unit impulse, so the weights are SciPy's to the bit.

    Returns:
        numpy.ndarray: the centre and one side, or None if not symmetric
    """
    sigma = (factor - 1) / 2
    # SciPy's radius at its default truncate of 4, and one more for a check
    radius = int(4.0 * sigma + 0.5)
    impulse = np.zeros(2 * radius + 3)
    impulse[radius + 1] = 1.0
    response = _ndimage.gaussian_filter1d(impulse, sigma, mode="constant")
    if response[0] != 0 or not np.array_equal(response, response[::-1]):
        return None
    return response[radius + 1 : 2 * radius + 2].copy()


def _anti_aliased_shrink(data, size):
    """
    skimage's `resize(data, (data.shape[0], size), anti_aliasing=True,
    mode="constant")`, bit for bit, for float32 rows shrunk by an integer
    factor: the Gaussian is computed only at the columns the interpolation
    reads, instead of all of them.

    Returns:
        numpy.ndarray: the resized array, or None where skimage is needed
    """
    if (
        type(data) is not np.ndarray
        or data.dtype != np.float32
        or data.ndim != 2
        or data.size == 0
        or size <= 0
        or data.shape[1] % size
        or data.shape[1] // size < 2
    ):
        return None
    lo, hi = data.min(), data.max()
    # NaN and infinity reach further through SciPy's zero-weight terms
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    factor = data.shape[1] // size
    weights = _half_gaussian(factor)
    if weights is None:
        return None
    out = _rust_gaussian_decimate(np.ascontiguousarray(data), weights, factor)
    # skimage's clip: to the input's range, widened to take in cval (0)
    # where the output reaches it
    if not lo <= 0 <= hi and out.min() <= 0 <= out.max():
        lo, hi = min(lo, np.float32(0)), max(hi, np.float32(0))
    np.clip(out, lo, hi, out=out)
    return out


def crop(data, start_sample, length, axis):
    """
    Crops the input array to a required size

    Args:
        data (np.ndarray): Data array to crop
        start_sample (int): Sample to start the output cropped array
        length (int): Final Length along the axis of the output
        axis (int): Axis to crop

    Returns:
        np.ndarray: Cropped array

    """
    if data.shape[axis] > start_sample + length:
        if axis:
            return data[:, start_sample : start_sample + length]
        else:
            return data[start_sample : start_sample + length, :]
    elif data.shape[axis] == length:
        return data
    else:
        raise OverflowError("Specified length exceeds the size of data")


def pad_along_axis(array: np.ndarray, target_length, loc="end", axis=0, **kwargs):
    """
    Pads data along the required axis on the input array to reach a target size

    Args:
        array (np.ndarray): Input array to pad
        target_length (int): Required length of the axis
        loc (int): Location to pad: start: pad in beginning, end: pad in end, else: pad equally on both sides
        axis (int): Axis to pad along
        **kwargs: args for np.pad

    Returns:
        np.ndarray: Padded array

    """
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]

    if loc == "start":
        npad[axis] = (int(pad_size), 0)
    elif loc == "end":
        npad[axis] = (0, int(pad_size))
    else:
        npad[axis] = (int(pad_size // 2), int(pad_size // 2))

    return np.pad(array, pad_width=npad, **kwargs)


def check_file_exist(file):
    """

    Args:
        file: Path of file to check

    Raises:
        IOError

    """
    if not os.path.isfile(file):
        raise IOError(f"{file} not found")


class MyEncoder(json.JSONEncoder):
    """
    Custom Encoder Class to convert any class to a JSON serializable object

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class YourArgparseFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """
    Allows both Raw Text Formatting and Default Args
    """

    pass
