import logging

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter as sg

try:
    from your._rust import rfi_stats as _rust_rfi_stats
except ImportError:
    _rust_rfi_stats = None

logger = logging.getLogger(__name__)


def savgol_filter(bandpass, channel_bandwidth, frequency_window=15, sigma=6):
    """
    Apply savgol filter to the data. See [Agarwal el al. 2020](https://arxiv.org/abs/2003.14272) for details.

    Args:
        bandpass (numpy.ndarray): bandpass of the data
        channel_bandwidth (float): channel bandwidth (MHz)
        frequency_window (float): frequency window (MHz)
        sigma (float): sigma value to apply cutoff on

    Returns:
        numpy.ndarray: mask for channels

    """
    window = int(np.ceil(frequency_window / np.abs(channel_bandwidth)) // 2 * 2 + 1)
    if window < 41:
        logger.warning(
            "Window size is less than 41 channels. Setting it to 41 channels."
        )
        window = 41

    if window > len(bandpass):
        logger.warning(
            "Window size is larger than the number of channels. Setting it to total number of channels."
        )
        nch = len(bandpass)
        if nch % 2:
            window = nch - 1
        else:
            window = nch

    logger.debug(f"Window size for savgol filter is {window}.")

    y = sg(bandpass, window, 2)
    sub = bandpass - y
    sigma = sigma * np.std(sub)
    mask = (sub > sigma) | (sub < -sigma)
    return mask


def spectral_kurtosis(data, N=1, d=None):
    """
    Compute spectral kurtosis. See [Nita et al. (2016)](https://doi.org/10.1109/RFINT.2016.7833535) for details.

    Args:
        data (numpy.ndarray): 2D frequency time data
        N (int): Number of accumulations on the FPGA
        d (float): shape factor

    Returns:
         numpy.ndarray: Spectral Kurtosis along frequency axis

    """
    M = data.shape[0]
    if (
        _rust_rfi_stats is not None
        and type(data) is np.ndarray
        and data.dtype == np.uint8
        and data.ndim == 2
        and data.flags.c_contiguous
    ):
        stats = _rust_rfi_stats(data)
        if stats is not None:
            S1, S2, total, count, variance = stats
            channel_mask = S1 == 0
            S1 = np.ma.array(S1, mask=channel_mask)[()]
            S2 = np.ma.array(S2, mask=channel_mask)[()]
            if d is None:
                mean = total / count
                d = (mean / np.sqrt(variance / count)) ** 2
            return ((M * d * N) + 1) * ((M * S2 / (S1**2)) - 1) / (M - 1)

    zero_mask = data == 0
    if type(data) is np.ndarray and data.dtype.kind in "iu" and data.ndim in (1, 2):
        count = data.size - np.count_nonzero(zero_mask)
    else:
        count = 0
    if count:
        # Integer inputs are finite; zeros contribute nothing to either sum.
        work = data.astype(float)
        channel_mask = zero_mask.all(axis=0)
        S1 = np.ma.array(work.sum(0), mask=channel_mask)[()]
        S2 = np.ma.array((work**2).sum(0), mask=channel_mask)[()]
        if d is None:
            # Preserve the original mean/std reduction order, including F-order data.
            mean = work.ravel().sum() / count
            variance_mean = work.sum() / count
            work -= variance_mean
            work[zero_mask] = 0
            np.square(work, out=work)
            d = (mean / np.sqrt(work.sum() / count)) ** 2
    else:
        data = np.ma.array(data.astype(float), mask=zero_mask)
        S1 = data.sum(0)
        S2 = (data**2).sum(0)
        if d is None:
            d = (np.nanmean(data.ravel()) / np.nanstd(data)) ** 2
    return ((M * d * N) + 1) * ((M * S2 / (S1**2)) - 1) / (M - 1)


def sk_filter(data, channel_bandwidth, tsamp, N=None, d=None, sigma=5):
    """
    Apply Spectral Kurtosis filter to the data

    Args:
        data (numpy.ndarray): 2D frequency time data
        channel_bandwidth (float): channel bandwidth (MHz)
        tsamp (float): sampling time (seconds)
        N (int): Number of accumulations on the FPGA
        d (float): shape factor
        sigma (float): sigma value to apply cutoff on

    Returns:
         numpy.ndarray: mask for channels

    """
    if not N:
        N = calc_N(channel_bandwidth, tsamp)
    sk = spectral_kurtosis(data, d=d, N=N)
    nan_mask = np.isnan(sk)
    sk[nan_mask] = np.nan
    sk_c = sk[~nan_mask]
    std = 1.4826 * stats.median_abs_deviation(sk_c)
    h = np.median(sk_c) + sigma * std
    l = np.median(sk_c) - sigma * std
    mask = (sk < h) & (sk > l)
    return ~mask


def calc_N(channel_bandwidth, tsamp):
    """

    Calculates number of accumulations on FPGA

    Args:
        channel_bandwidth (float): channel bandwidth (MHz)
        tsamp (float): sampling time (seconds)

    Returns:
        int: FPGA accumulation length

    """

    tn = np.abs(1 / (channel_bandwidth * 10**6))
    return np.round(tsamp / tn)


def sk_sg_filter(
    data,
    your_object=None,
    spectral_kurtosis_sigma=6,
    savgol_frequency_window=15,
    savgol_sigma=5,
    foff=None,
    tsamp=None,
):
    """

    Apply Spectral Kurtosis and Savgol filter to the data

    Args:
        data (numpy.ndarray): 2D frequency time data
        your_object: Your object
        spectral_kurtosis_sigma (float): sigma value to apply cutoff on for SK filter
        savgol_frequency_window (float): frequency window for savgol filter(MHz)
        savgol_sigma (float): sigma value to apply cutoff on for savgol filter
        foff (float): Channel bandwidth (MHz). If provided, will be used instead of `your_object`
        tsamp (float): Sampling interval (seconds). If provided, will be used instead of `your_object`

    Returns:
         numpy.ndarray: mask for channels

    """
    if your_object:
        if foff == None:
            foff = your_object.your_header.foff
        if tsamp == None:
            tsamp = your_object.your_header.tsamp
    else:
        if not foff or not tsamp:
            raise ValueError("foff and tsamp cannot be None while your_object is None")

    if (spectral_kurtosis_sigma == 0) and (savgol_sigma) == 0:
        raise ValueError(
            "Both savgol_sigma and spectral_kurtosis_sigma cannot be zero."
        )

    mask = np.zeros(data.shape[1], dtype=np.bool_)
    if spectral_kurtosis_sigma > 0:
        logger.debug(
            f"Applying spectral kurtosis filter with sigma={spectral_kurtosis_sigma}"
        )
        sk_mask = sk_filter(
            data=data,
            channel_bandwidth=foff,
            tsamp=tsamp,
            sigma=spectral_kurtosis_sigma,
        )
        mask[sk_mask] = True
    elif spectral_kurtosis_sigma == 0:
        sk_mask = np.zeros(data.shape[1], dtype=np.bool_)
        pass
    else:
        raise ValueError("spectral_kurtosis_sigma can't be negative")

    if savgol_sigma > 0:
        bp = data.sum(0)[~sk_mask]
        logger.debug(
            f"Applying savgol filter with frequency_window={savgol_frequency_window} and sigma={savgol_sigma}"
        )
        sg_mask = savgol_filter(
            bandpass=bp,
            channel_bandwidth=foff,
            frequency_window=savgol_frequency_window,
            sigma=savgol_sigma,
        )
        mask[np.where(mask == False)[0][sg_mask]] = True
    elif savgol_sigma == 0:
        pass
    else:
        raise ValueError("savgol_sigma can't be negative")

    return mask
