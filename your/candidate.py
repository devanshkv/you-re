#!/usr/bin/env python3

import os
from operator import index

import h5py
import numpy as np
from numba import njit, prange
from scipy.optimize import golden

from your import Your
from your._rust import crop_planes as _rust_crop_planes
from your._rust import dedisperse as _rust_dedisperse
from your._rust import dedispersets as _rust_dedispersets
from your._rust import dmtime as _rust_dmtime
from your.utils.gpu import gpu_dedisperse, gpu_dmt
from your.utils.misc import *
from your.utils.misc import _decimate, _resize
from your.utils.rfi import sk_sg_filter

_RUST_DEDISPERSETS_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
    np.dtype(np.float64),
}

_RUST_CROP_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.int32),
}

logger = logging.getLogger(__name__)


def crop_window(nsamples, decimate_factor, time_size):
    """
    Which columns the candmaker keeps of a plane averaged over
    `decimate_factor` samples by `decimate(..., pad=True)` and then cropped
    to its middle `time_size` columns by `crop`.

    Args:
        nsamples (int): samples in the chunk
        decimate_factor (int): samples averaged into each column
        time_size (int): columns kept

    Returns:
        tuple: (first column, number of columns), or None where `crop` would
        raise, or where a kept column would take in the median padding, which
        needs the whole full-resolution row
    """
    ncols = -(-nsamples // decimate_factor)
    col0 = ncols // 2 - time_size // 2
    if ncols > col0 + time_size:
        kept = time_size
    elif ncols == time_size:
        col0, kept = 0, ncols
    else:
        return None
    if nsamples % decimate_factor and col0 + kept == ncols:
        return None
    return col0, kept


@njit(parallel=True, cache=True)
def _channel_histogram(data, counts):
    """
    Count each 8-bit value in each channel: `counts[ch, v]` is how many
    samples of channel `ch` are `v`. Threads take a band of 256 channels
    each, reading whole cache lines of the chunk, and count into a private
    32-bit table small enough to stay in cache before storing it.
    """
    nsamples, nchans = data.shape
    for band in prange((nchans + 255) // 256):
        c0 = band * 256
        c1 = min(c0 + 256, nchans)
        local = np.zeros((c1 - c0, 256), dtype=np.uint32)
        for s in range(nsamples):
            for ch in range(c0, c1):
                local[ch - c0, data[s, ch]] += 1
        for ch in range(c0, c1):
            for v in range(256):
                counts[ch, v] = local[ch - c0, v]


def channel_median(data):
    """
    `np.median(data, axis=0)`, from per-channel value counts for 8-bit data:
    the same middle order statistics, and for an even count the same float64
    mean of the two, without partitioning every channel of a large chunk.

    Args:
        data (numpy.ndarray): (nsamples, nchans) data

    Returns:
        numpy.ndarray: (nchans,) float64 median of each channel
    """
    if data.dtype != np.uint8 or data.ndim != 2 or data.shape[0] == 0:
        return np.median(data, axis=0)
    nsamples, nchans = data.shape
    counts = np.zeros((nchans, 256), dtype=np.int64)
    _channel_histogram(data, counts)
    below = np.cumsum(counts, axis=1)

    def order_statistic(k):
        # the smallest value with more than k samples at or below it
        return (below <= k).sum(axis=1)

    hi = order_statistic(nsamples // 2)
    if nsamples % 2:
        return hi.astype(np.float64)
    lo = order_statistic(nsamples // 2 - 1)
    return (lo.astype(np.float64) + hi) / 2


def median_fill(data, dtype):
    """
    Each channel's median cast to `dtype`, as multiplying a ones array by the
    median and casting back did: the value padding takes.

    Args:
        data (numpy.ndarray): (rows, nchans) data read
        dtype: dtype of the padded chunk

    Returns:
        numpy.ndarray: (1, nchans) fill row
    """
    return (np.ones((1, data.shape[1]), dtype=dtype) * channel_median(data)).astype(
        dtype
    )


def pad_with_median(data, nsamples, row0, dtype):
    """
    Place `data` at `row0` of `nsamples` rows, the rest filled with each
    channel's median cast to `dtype`, as multiplying a ones array by the
    median and casting back did, without the float64 copy of the chunk.

    Args:
        data (numpy.ndarray): (rows, nchans) data read
        nsamples (int): rows of the padded chunk
        row0 (int): row the data starts at
        dtype: dtype of the padded chunk

    Returns:
        numpy.ndarray: (nsamples, nchans) padded chunk
    """
    out = np.empty((nsamples, data.shape[1]), dtype=dtype)
    out[:] = median_fill(data, dtype)
    out[row0 : row0 + data.shape[0]] = data
    return out


def _time_bounds(nt, time_range):
    if time_range is None:
        return 0, nt
    start, stop = map(index, time_range)
    if not 0 <= start <= stop <= nt:
        raise ValueError("expected 0 <= start <= stop <= data length")
    return start, stop


def _dedispersed_channel(column, delay, start, stop):
    """Slice the existing circular-shift convention without a full-channel copy."""
    nt = len(column)
    if start == stop:
        return column[:0]
    # Delays outside the input length are identity shifts in the original slices.
    pivot = -int(delay) if -nt < delay < nt else 0
    source = (start + pivot) % nt
    count = min(stop - start, nt - source)
    return np.concatenate(
        (column[source : source + count], column[: stop - start - count])
    )


class Candidate(Your):
    """
    Candidate Class

    Args:
        fp Union[str, list]: String or a list of files. It can either filterbank or psrfits files.
        dm (float): Dispersion Measure of the candidate
        tcand (float): start time of the candidate in seconds at the highest frequency channel
        width (int): pulse width of the candidate in samples
        label (int): 1 for pulsars/FRBs, 0 for RFI
        snr (float): Signal to Noise Ratio
        min_samp (int): Minimum number of time samples
        device (int): GPU ID if using GPUs
        kill_mask (numpy.ndarray): Boolean mask of channels to kill
        spectral_kurtosis_sigma (float): Sigma for spectral kurtosis filter
        savgol_frequency_window (float): Filter window for savgol filter
        savgol_sigma (float):  Sigma for savgol filter
        flag_rfi (bool): To turn on RFI flagging
    """

    def __init__(
        self,
        fp=None,
        dm=None,
        tcand=0,
        width=0,
        label=-1,
        snr=0,
        min_samp=256,
        device=0,
        kill_mask=np.array([False]),
        spectral_kurtosis_sigma=4,
        savgol_frequency_window=15,
        savgol_sigma=4,
        flag_rfi=False,
    ):
        Your.__init__(self, fp)
        try:
            self._reset_candidate(
                dm=dm,
                tcand=tcand,
                width=width,
                label=label,
                snr=snr,
                min_samp=min_samp,
                device=device,
                kill_mask=kill_mask,
                spectral_kurtosis_sigma=spectral_kurtosis_sigma,
                savgol_frequency_window=savgol_frequency_window,
                savgol_sigma=savgol_sigma,
                flag_rfi=flag_rfi,
            )
        except BaseException:
            self.close()
            raise

    def _reset_candidate(
        self,
        *,
        dm=None,
        tcand=0,
        width=0,
        label=-1,
        snr=0,
        min_samp=256,
        device=0,
        kill_mask=np.array([False]),
        spectral_kurtosis_sigma=4,
        savgol_frequency_window=15,
        savgol_sigma=4,
        flag_rfi=False,
    ):
        """Reset candidate-specific state while retaining its input reader."""
        self.your_header.time_decimation_factor = 1
        self.your_header.frequency_decimation_factor = 1
        self.dm = dm
        self.tcand = tcand
        self.width = width
        self.label = label
        self.snr = snr
        self.id = f"cand_tstart_{self.tstart:.12f}_tcand_{self.tcand:.7f}_dm_{self.dm:.5f}_snr_{self.snr:.5f}"
        self.data = None
        self.dedispersed = None
        self.dmt = None
        self.device = device
        self.min_samp = min_samp
        self.dm_opt = -1
        self.snr_opt = -1
        self.kill_mask = kill_mask
        self.spectral_kurtosis_sigma = spectral_kurtosis_sigma
        self.savgol_frequency_window = savgol_frequency_window
        self.savgol_sigma = savgol_sigma
        self.flag_rfi = flag_rfi
        self.rfi_mask = np.array([False])
        logger.debug(
            f"Initiated a cand object with "
            f"dm: {self.dm}, "
            f"snr: {self.snr}, "
            f"width:{self.width}, "
            f"tcand: {self.tcand}"
        )
        return self

    def save_h5(self, out_dir=None, fnout=None):
        """
        Save the candidate to a hdf5 file

        Args:
            out_dir (str): path to the output directory
            fnout (str): output name of the file

        Returns:
            str: output name of the file

        """
        cand_id = self.id
        if fnout is None:
            fnout = cand_id + ".h5"
        if out_dir is not None:
            if out_dir[-1] != "/":
                out_dir = out_dir + "/"
            fnout = out_dir + fnout
        logger.info(f"Saving h5 file {fnout}.")
        with h5py.File(fnout, "w") as f:
            f.attrs["cand_id"] = cand_id
            f.attrs["tcand"] = self.tcand
            f.attrs["dm"] = self.dm
            f.attrs["dm_opt"] = self.dm_opt
            f.attrs["snr"] = self.snr
            f.attrs["snr_opt"] = self.snr_opt
            f.attrs["width"] = self.width
            f.attrs["label"] = self.label
            f.attrs["rfi_mask"] = self.rfi_mask
            f.attrs["kill_mask"] = self.kill_mask

            f.attrs["filelist"] = self.your_header.filelist

            # Copy over header information as attributes
            file_header = vars(self.your_header)
            for key in file_header.keys():
                if key == "dtype":
                    f.attrs[key] = np.dtype(file_header[key]).name
                else:
                    f.attrs[key] = file_header[key]

            f.attrs["tsamp"] = self.your_header.tsamp
            f.attrs["nchans"] = self.your_header.nchans
            f.attrs["foff"] = self.your_header.foff
            f.attrs["nspectra"] = self.your_header.nspectra

            freq_time_dset = f.create_dataset(
                "data_freq_time",
                data=self.dedispersed,
                dtype=self.dedispersed.dtype,
                compression="gzip",
                compression_opts=9,
            )
            freq_time_dset.dims[0].label = b"time"
            freq_time_dset.dims[1].label = b"frequency"

            if self.dmt is not None:
                dm_time_dset = f.create_dataset(
                    "data_dm_time",
                    data=self.dmt,
                    dtype=self.dmt.dtype,
                    compression="gzip",
                    compression_opts=9,
                )
                dm_time_dset.dims[0].label = b"dm"
                dm_time_dset.dims[1].label = b"time"
        return fnout

    def dispersion_delay(self, dms=None):
        """
        Calculate the dispersion delay for the candidate DM or at given dispersion DM

        Args:
            dms (Union[float,np.ndarray]): DM or a list of DMs

        Returns:
            Union[float, np.ndarray]: dispersion delay in seconds

        """
        if dms is None:
            dms = self.dm

        return (
            4148808.0
            * dms
            * (1 / np.min(self.chan_freqs) ** 2 - 1 / np.max(self.chan_freqs) ** 2)
            / 1000
        )

    def get_chunk(self, tstart=None, tstop=None, for_preprocessing=True):
        """
        Get a chunk of data. The data is saved in `self.data`.

        Args:
            tstart (float): start time of the chunk in seconds
            tstop (float): stop time of the chunk in seconds
            for_preprocessing (bool): if the data is to be preprocessed later. This will modify the number of samples
            read based on the width of the candidate

        """
        if tstart is None:
            tstart = (
                self.tcand - self.dispersion_delay() - self.width * self.native_tsamp
            )
        if tstop is None:
            tstop = (
                self.tcand + self.dispersion_delay() + self.width * self.native_tsamp
            )
        logger.debug(f"tstart is {tstart}")
        logger.debug(f"tstop is {tstop}")

        nstart = int(tstart / self.native_tsamp)
        nsamp = int((tstop - tstart) / self.native_tsamp)
        nsamp_read = nsamp

        if for_preprocessing:
            if self.width > 2 and nsamp_read // (self.width // 2) < self.min_samp:
                nsamp_read = self.min_samp * self.width // 2
                nstart_read = nstart - (nsamp_read - nsamp) // 2
            elif nsamp_read < self.min_samp:
                nsamp_read = self.min_samp
                nstart_read = nstart - (nsamp_read - nsamp) // 2
            else:
                nstart_read = nstart
        else:
            nstart_read = nstart
        logging.debug(
            f"nstart_read is {nstart_read}, nsamp_read is {nsamp_read},"
            f"nstart is {nstart}, nsamp is {nsamp}"
        )

        nspectra = int(self.your_header.nspectra)
        if nstart_read >= 0 and nstart_read + nsamp_read <= nspectra:
            logging.debug(
                f"All the data available in the file, no need to pad. \n"
                f"nstart_read({nstart_read})>=0 and \n"
                f"nstart_read({nstart_read})+nsamp_read({nsamp_read})<=nspectra({nspectra})"
            )
            data = self.get_data(nstart=nstart_read, nsamp=nsamp_read)
        elif nstart_read < 0:
            if nstart_read + nsamp_read <= nspectra:
                logging.debug(
                    f"nstart_read({nstart_read})<0 and nstart_read({nstart_read})\n"
                    f"+nsamp_read({nsamp_read})<=nspectra({nspectra})"
                )
                logging.info("Padding with median in the beginning")
                data = self._read_padded(
                    nsamp_read, -nstart_read, 0, nsamp_read + nstart_read
                )
            else:
                logging.debug(
                    f"nstart_read({nstart_read})<0 and nstart_read({nstart_read})"
                    f"+nsamp_read({nsamp_read})>nspectra({nspectra})"
                )
                logging.info("Padding with median in the beginning and the end")
                data = self._read_padded(nsamp_read, -nstart_read, 0, nspectra)
        else:
            logging.debug(
                f"nstart_read({nstart_read})>=0 and nstart_read({nstart_read})"
                f"+nsamp_read({nsamp_read})>nspectra({nspectra})"
            )
            logging.info("Padding with median in the end")
            data = self._read_padded(nsamp_read, 0, nstart_read, nspectra - nstart_read)

        # no copy when the data read already has the header's dtype, so a
        # chunk read into a reused buffer stays there
        self.data = data.astype(self.your_header.dtype, copy=False)

        if self.kill_mask.any():
            logger.info("Applying the kill mask")
            assert len(self.kill_mask) == self.data.shape[1]
            if self.kill_mask.dtype == np.bool_ and self.kill_mask.ndim == 1:
                np.copyto(self.data, 0, where=self.kill_mask[None, :])
            else:
                self.data[:, self.kill_mask] = 0

        if self.flag_rfi:
            mask = sk_sg_filter(
                data=self.data,
                your_object=self,
                spectral_kurtosis_sigma=self.spectral_kurtosis_sigma,
                savgol_frequency_window=self.savgol_frequency_window,
                savgol_sigma=self.savgol_sigma,
            )
            self.rfi_mask = mask
            np.copyto(self.data, 0, where=self.rfi_mask[None, :])
        return self

    def _read_padded(self, nsamples, row0, nstart, nsamp):
        """
        `get_data(nstart, nsamp)` placed at row `row0` of `nsamples` rows, the
        rest each channel's median, as `pad_with_median` makes it. With a
        `read_buffer`, the samples are read straight into their place in it
        and only the padding is written, instead of copying the read into a
        new padded array; readers that do not fill the buffer in place get
        `pad_with_median`.

        Args:
            nsamples (int): rows of the padded chunk
            row0 (int): row the samples read start at
            nstart (int): first sample to read
            nsamp (int): samples to read

        Returns:
            numpy.ndarray: (nsamples, nchans) padded chunk
        """
        dtype = self.your_header.dtype
        read_buffer = getattr(self, "read_buffer", None)
        if read_buffer is None:
            return pad_with_median(
                self.get_data(nstart=nstart, nsamp=nsamp), nsamples, row0, dtype
            )
        rowbytes = self.your_header.nchans * np.dtype(dtype).itemsize
        full = read_buffer(nsamples * rowbytes)
        offset = row0 * rowbytes

        def in_place(nbytes):
            if offset + nbytes <= full.size:
                return full[offset : offset + nbytes]
            return np.empty(nbytes, dtype=np.uint8)

        self.read_buffer = in_place
        try:
            d = self.get_data(nstart=nstart, nsamp=nsamp)
        finally:
            self.read_buffer = read_buffer
        out = full[: nsamples * rowbytes].view(dtype).reshape(nsamples, -1)
        placed = out[row0 : row0 + nsamp]
        if (
            d.dtype != dtype
            or d.shape != placed.shape
            or not d.flags.c_contiguous
            or d.__array_interface__["data"][0] != placed.__array_interface__["data"][0]
        ):
            return pad_with_median(d, nsamples, row0, dtype)
        fill = median_fill(d, dtype)
        out[:row0] = fill
        out[row0 + nsamp :] = fill
        return out

    def dedisperse(self, dms=None, target="CPU", *, time_range=None):
        """
        Dedisperse a chunk of data. Saves the dedispersed chunk in `self.dedispersed`.

        Note:
            Our method rolls the data around while dedispersing it.

        Args:
            time_range (tuple): Optional (start, stop) samples in the full shifted output, before decimation. CPU only.
            dms (float): The DM to dedisperse the data at.
            target (str): 'CPU' to run the code on the CPU or 'GPU' to run it on a GPU.

        """

        if time_range is not None and target != "CPU":
            raise ValueError("time_range is supported only on CPU")
        if dms is None:
            dms = self.dm
        if self.data is not None:
            if target == "CPU":
                nt, nf = self.data.shape
                assert nf == len(self.chan_freqs)
                delay_time = (
                    4148808.0
                    * dms
                    * (1 / (self.chan_freqs[0]) ** 2 - 1 / (self.chan_freqs) ** 2)
                    / 1000
                )
                delay_bins = np.round(delay_time / self.native_tsamp).astype("int64")
                start, stop = _time_bounds(nt, time_range)
                if (
                    type(self.data) is np.ndarray
                    and self.data.dtype in _RUST_DEDISPERSETS_DTYPES
                    # Retain NumPy's float64-to-float32 warning/error policy.
                    and self.data.dtype != np.float64
                    and self.data.flags.aligned
                    and self.data.flags.c_contiguous
                    # Long, narrow outputs can be faster with NumPy column copies.
                    and (stop - start <= 1024 or nf >= 512)
                ):
                    self.dedispersed = _rust_dedisperse(
                        self.data, delay_bins, start, stop
                    )
                else:
                    self.dedispersed = np.empty((stop - start, nf), dtype=np.float32)
                    for ii in range(nf):
                        self.dedispersed[:, ii] = _dedispersed_channel(
                            self.data[:, ii], delay_bins[ii], start, stop
                        )
            elif target == "GPU":
                gpu_dedisperse(self, device=self.device)
        else:
            logger.warning("No data in self.data, run self.get_chunk() first")
            self.dedispersed = None
        return self

    def dedispersets(self, dms=None, *, time_range=None):
        """
        Create a dedispersed time series

        Note:
            Our method rolls the data around while dedispersing it.

        Args:
            time_range (tuple): Optional (start, stop) samples in the full shifted output, before decimation. CPU only.
            dms (float): The DM to dedisperse the data at.

        Returns:
            numpy.ndarray: Dedispersed time series.

        """
        if dms is None:
            dms = self.dm
        if self.data is not None:
            nt, nf = self.data.shape
            assert nf == len(self.chan_freqs)
            delay_time = (
                4148808.0
                * dms
                * (1 / (self.chan_freqs[0]) ** 2 - 1 / (self.chan_freqs) ** 2)
                / 1000
            )
            delay_bins = np.round(delay_time / self.native_tsamp).astype("int64")
            start, stop = _time_bounds(nt, time_range)
            if (
                type(self.data) is np.ndarray
                and self.data.dtype in _RUST_DEDISPERSETS_DTYPES
                and self.data.flags.aligned
            ):
                return _rust_dedispersets(self.data, delay_bins, start, stop)
            ts = np.zeros(stop - start, dtype=np.float32)
            for ii in range(nf):
                ts += _dedispersed_channel(
                    self.data[:, ii], delay_bins[ii], start, stop
                )
            return ts

    def crop_planes(
        self,
        decimate_factor,
        time_size=256,
        dmsteps=256,
        *,
        upstream_rounding=False,
        threads=None,
    ):
        """
        The DM-time and frequency-time planes `dmtime` and `dedisperse` make,
        averaged over `decimate_factor` samples and cropped to their middle
        `time_size` columns as the candmaker does with `decimate(pad=True)`
        and `crop`, computing only the columns kept, in one threaded pass.
        Saves them in `self.dmt` and `self.dedispersed`.

        Args:
            decimate_factor (int): samples averaged into each column
            time_size (int): columns kept, about the middle
            dmsteps (int): Number of DMs to dedisperse at.
            upstream_rounding (bool): exact integer sums through each
                channel's running sum, divided once, when False; when True,
                round as the full-array route does (float32 per sample, then
                NumPy's float32 mean), matching it bit for bit where a
                column's float32 sum passes 2**24, at the cost of summing
                every sample
            threads (int): threads to use; all cores when None

        Returns:
            tuple: (DM-time plane, frequency-time plane), or None, leaving
            both alone, when the full-array route is needed: data other than
            8-32 bit integers, or a crop that takes in the median padding.
        """
        if (
            type(self.data) is not np.ndarray
            or self.data.dtype not in _RUST_CROP_DTYPES
            or self.data.ndim != 2
        ):
            return None
        nt, nf = self.data.shape
        window = crop_window(nt, decimate_factor, time_size)
        if window is None:
            return None
        col0, kept = window

        freqs = self.chan_freqs
        assert nf == len(freqs)
        frequency_term = 1 / freqs[0] ** 2 - 1 / freqs**2

        def shifts(delays):
            # delays of the whole chunk or more leave a channel where it is,
            # as the full-array route's slices do
            delays = delays.astype(np.int64)
            return np.where(np.abs(delays) < nt, delays % nt, 0)

        dm_list = self.dm + np.linspace(-self.dm, self.dm, dmsteps)
        dmt_delays = np.round(
            (4148808.0 * dm_list[:, None])
            * frequency_term[None, :]
            / 1000.0
            / self.native_tsamp
        )
        ft_delays = np.round(
            4148808.0 * self.dm * frequency_term / 1000 / self.native_tsamp
        )
        self.dmt, self.dedispersed = _rust_crop_planes(
            np.ascontiguousarray(self.data),
            np.ascontiguousarray(shifts(dmt_delays)),
            shifts(ft_delays),
            decimate_factor,
            col0,
            kept,
            threads=threads or os.cpu_count() or 1,
            upstream_rounding=upstream_rounding,
        )
        return self.dmt, self.dedispersed

    def dmtime(self, dmsteps=256, target="CPU", *, time_range=None):
        """
        Generates DM-time array of the candidate by dedispersing at adjacent DM values. Saves the data in `self.dmt`.

        Note:
            Our method rolls the data around while dedispersing it.

        Args:
            time_range (tuple): Optional (start, stop) samples in the full shifted output, before decimation. CPU only.
            dmsteps (int): Number of DMs to dedisperse at.
            target (str): 'CPU' to run the code on the CPU or 'GPU' to run it on a GPU.

        """
        if time_range is not None and target != "CPU":
            raise ValueError("time_range is supported only on CPU")
        if target == "CPU":
            range_dm = self.dm
            dm_list = self.dm + np.linspace(-range_dm, range_dm, dmsteps)
            start, stop = _time_bounds(self.data.shape[0], time_range)
            if (
                type(self) is Candidate
                and type(self.data) is np.ndarray
                and self.data.dtype in _RUST_DEDISPERSETS_DTYPES
                and self.data.flags.aligned
                and (
                    stop - start <= 1024
                    # Keep small or sparse wide batches on the scalar path;
                    # packing's full-input pass needs enough repeated DM work.
                    or (
                        self.data.flags.c_contiguous
                        and len(dm_list) >= 32
                        and 16 * self.data.shape[0] <= len(dm_list) * (stop - start)
                    )
                )
                and getattr(self.dedispersets, "__func__", None)
                is Candidate.dedispersets
                and len(dm_list)
            ):
                freqs = self.chan_freqs
                assert self.data.shape[1] == len(freqs)
                frequency_term = 1 / freqs[0] ** 2 - 1 / freqs**2
                self.dmt = _rust_dmtime(
                    self.data, dm_list, frequency_term, self.native_tsamp, start, stop
                )
            else:
                self.dmt = np.empty((dmsteps, stop - start), dtype=np.float32)
                for ii, dm in enumerate(dm_list):
                    self.dmt[ii, :] = self.dedispersets(dms=dm, time_range=time_range)
        elif target == "GPU":
            gpu_dmt(self, device=self.device)
        return self

    def get_snr(self, time_series=None):
        """
        Calculates the SNR of the candidate

        Args:
            time_series (np.ndarray): time series array to calculate the SNR of

        Returns:
            float: SNR
        """
        if time_series is None and self.dedispersed is None:
            return None
        if time_series is None:
            x = self.dedispersed.mean(1)
        else:
            x = time_series
        argmax = np.argmax(x)
        mask = np.ones(len(x), dtype=np.bool_)
        mask[max(0, argmax - self.width // 2) : argmax + self.width // 2] = 0
        x = x - x[mask].mean()
        std = np.std(x[mask])
        return x.max() / std

    def optimize_dm(self):
        """
        Calculate more precise value of the DM by interpolating between DM values to maximise the SNR

        Note:
            This function has not been fully tested.

        Returns:
            Optimized DM, optimised SNR
        """
        if self.data is None:
            return None

        def dm2snr(dm):
            time_series = self.dedispersets(dm)
            return -self.get_snr(time_series)

        try:
            out = golden(
                dm2snr,
                full_output=True,
                brack=(-self.dm / 2, self.dm, 2 * self.dm),
                tol=1e-3,
            )
        except (ValueError, TypeError):
            out = golden(dm2snr, full_output=True, tol=1e-3)
        self.dm_opt = out[0]
        self.snr_opt = -out[1]
        return out[0], -out[1]

    def decimate(self, key, decimate_factor, axis, pad=False, **kwargs):
        """
        Decimate FT or DMT data.

        Todo:
            * Update candidate parameters as per decimation factor

        Args:
            key (str): Keywords to chose which data to decimate ('dmt' or 'ft')
            decimate_factor (int): Number of samples to average
            axis (int): Axis to decimate along
            pad (bool): Optional argument if padding is to be done
            **kwargs: kwargs for numpy.pad
        """
        if key == "dmt":
            logger.debug(
                f"Decimating dmt along axis {axis}, with factor {decimate_factor},  pre-decimation shape: {self.dmt.shape}"
            )
            self.dmt = _decimate(self.dmt, decimate_factor, axis, pad, **kwargs)
            logger.debug(
                f"Decimated dmt along axis {axis}, post-decimation shape: {self.dmt.shape}"
            )
        elif key == "ft":
            logger.debug(
                f"Decimating ft along axis {axis}, with factor {decimate_factor}, pre-decimation shape: {self.dedispersed.shape}"
            )
            self.dedispersed = _decimate(
                self.dedispersed, decimate_factor, axis, pad, **kwargs
            )
            logger.debug(
                f"Decimated ft along axis {axis}, post-decimation shape: {self.dedispersed.shape}"
            )
        else:
            raise AttributeError(
                'Key can either be "dmt": DM-Time or "ft": Frequency-Time'
            )
        return self

    def resize(self, key, size, axis, **kwargs):
        """
        Resize FT or DMT data

        Todo:
            * Update candidate parameters as per final size

        Args:
            key (str): Keywords to chose which data to resize ('dmt' or 'ft')
            size: Final size of the data array required
            axis (int): Axis to resize alone
            **kwargs: Arguments for skimage.transform resize function

        """
        if key == "dmt":
            self.dmt = _resize(self.dmt, size, axis, **kwargs)
        elif key == "ft":
            self.dedispersed = _resize(self.dedispersed, size, axis, **kwargs)
        else:
            raise AttributeError(
                'Key can either be "dmt": DM-Time or "ft": Frequency-Time'
            )
        return self
