"""Small NumPy-only contracts shared by candidate processing paths."""

import numpy as np


def parse_kill_mask(mask, nchans):
    """Return a channel mask where True marks a channel to suppress."""
    mask = np.asarray(mask)
    killed = np.zeros(nchans, dtype=bool)
    if not mask.size:
        return killed
    if mask.dtype.kind == "b":
        if mask.ndim == 0 and not bool(mask):
            return killed
        if mask.ndim == 1 and mask.size == 1 and not mask[0]:
            return killed
        if mask.shape != (nchans,):
            raise ValueError("boolean kill_mask must have one value per channel")
        return mask.astype(bool, copy=True)
    if mask.dtype.kind in "iu":
        if mask.ndim != 1:
            raise ValueError("integer kill_mask must be one-dimensional")
        try:
            killed[mask] = True
        except IndexError as exc:
            raise ValueError("integer kill_mask contains an invalid channel") from exc
        return killed
    raise ValueError("kill_mask must be a boolean mask or channel indices")


def decoded_sample_dtype(reader):
    """Return the dtype produced by the reader's decode path."""
    header = reader.your_header
    if int(getattr(header, "nspectra", 0)) > 0:
        sample = np.asarray(reader.get_data(0, 1))
        if sample.shape[0] > 0:
            return sample.dtype

    if (
        str(getattr(header, "format", "")).lower() == "fits"
        and int(getattr(header, "nbits", 0)) == 32
    ):
        return np.dtype(np.float32)
    return np.dtype(header.dtype)
