#!/usr/bin/env python3

import hashlib
import json
import logging
import os
from contextlib import contextmanager
from operator import index
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from scipy.optimize import golden

from your import Your
from your._rust import dedisperse as _rust_dedisperse
from your._rust import dedispersets as _rust_dedispersets
from your.utils.gpu import gpu_dedisperse, gpu_dmt
from your.utils.misc import _decimate, _resize
from your.utils.misc import crop as crop
from your.utils.rfi import sk_sg_filter

_RUST_DEDISPERSETS_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float32),
    np.dtype(np.float64),
}

logger = logging.getLogger(__name__)


def _time_bounds(nt, time_range):
    if time_range is None:
        return 0, nt
    start, stop = map(index, time_range)
    if not 0 <= start <= stop <= nt:
        raise ValueError("expected 0 <= start <= stop <= data length")
    return start, stop


@contextmanager
def _atomic_h5(filename):
    """Publish a complete HDF5 file, preserving an existing file on failure."""
    filename = os.path.abspath(os.fspath(filename))
    with NamedTemporaryFile(
        prefix=".candidate-", suffix=".h5", dir=os.path.dirname(filename), delete=False
    ) as temporary:
        staging = temporary.name
    try:
        with h5py.File(staging, "w") as handle:
            yield handle
        os.replace(staging, filename)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)


class Candidate(Your):
    """
    Candidate Class

    Args:
        fp Union[str, list]: Filterbank or PSRFITS filename or list of filenames.
        dm (float): Dispersion Measure of the candidate
        tcand (float): Arrival time in seconds at the highest-frequency channel.
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
        self.processing_metadata = {}
        self._chunk_start = None
        self._chunk_shape = None
        self._update_id()
        logger.debug(
            f"Initiated a cand object with "
            f"dm: {self.dm}, "
            f"snr: {self.snr}, "
            f"width:{self.width}, "
            f"tcand: {self.tcand}"
        )
        return self

    def _update_id(self):
        """Refresh the output identity from the current candidate parameters."""
        identity = (
            tuple(os.path.realpath(name) for name in self.your_header.filelist),
            float(self.tstart),
            float(self.tcand),
            float(self.dm),
            float(self.snr),
            int(self.width),
            int(self.label),
            int(self.min_samp),
            np.asarray(self.kill_mask).tolist(),
            bool(self.flag_rfi),
            float(self.spectral_kurtosis_sigma),
            float(self.savgol_frequency_window),
            float(self.savgol_sigma),
        )
        token = hashlib.blake2b(
            json.dumps(identity, separators=(",", ":")).encode(), digest_size=12
        ).hexdigest()
        self.id = (
            f"cand_tstart_{self.tstart:.12f}_tcand_{self.tcand:.7f}"
            f"_dm_{self.dm:.5f}_snr_{self.snr:.5f}_{token}"
        )
        return self.id

    def save_h5(self, out_dir=None, fnout=None):
        """
        Save the candidate to a hdf5 file

        Args:
            out_dir (str): path to the output directory
            fnout (str): output name of the file

        Returns:
            str: output name of the file

        """
        cand_id = self._update_id()
        if fnout is None:
            fnout = cand_id + ".h5"
        fnout = os.fspath(fnout)
        if out_dir is not None:
            fnout = os.path.join(os.fspath(out_dir), fnout)
        logger.info(f"Saving h5 file {fnout}.")
        with _atomic_h5(fnout) as f:
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
            f.attrs["nchans"] = self.dedispersed.shape[1]
            f.attrs["foff"] = self.your_header.foff
            f.attrs["nspectra"] = self.your_header.nspectra

            for key, value in self.processing_metadata.items():
                f.attrs[key] = value

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
            for_preprocessing (bool): Extend the read according to candidate width
                when the data will be preprocessed later.

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
                d = self.get_data(nstart=0, nsamp=nsamp_read + nstart_read)
                dmedian = np.median(d, axis=0)
                data = (
                    np.ones(
                        (nsamp_read, self.your_header.nchans),
                        dtype=self.your_header.dtype,
                    )
                    * dmedian[None, :]
                )
                data[-nstart_read:, :] = d
            else:
                logging.debug(
                    f"nstart_read({nstart_read})<0 and nstart_read({nstart_read})"
                    f"+nsamp_read({nsamp_read})>nspectra({nspectra})"
                )
                logging.info("Padding with median in the beginning and the end")
                d = self.get_data(nstart=0, nsamp=nspectra)
                dmedian = np.median(d, axis=0)
                data = (
                    np.ones(
                        (nsamp_read, self.your_header.nchans),
                        dtype=self.your_header.dtype,
                    )
                    * dmedian[None, :]
                )
                data[-nstart_read : -nstart_read + nspectra, :] = d
        else:
            logging.debug(
                f"nstart_read({nstart_read})>=0 and nstart_read({nstart_read})"
                f"+nsamp_read({nsamp_read})>nspectra({nspectra})"
            )
            logging.info("Padding with median in the end")
            d = self.get_data(nstart=nstart_read, nsamp=nspectra - nstart_read)
            dmedian = np.median(d, axis=0)
            data = (
                np.ones(
                    (nsamp_read, self.your_header.nchans), dtype=self.your_header.dtype
                )
                * dmedian[None, :]
            )
            data[: nspectra - nstart_read, :] = d

        self.data = data.astype(self.your_header.dtype)
        self._chunk_start = nstart_read
        self._chunk_shape = self.data.shape

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

    def set_processed_products(self, plan, ft, dmt):
        """Attach finished candidate images and their sampling/processing metadata."""
        for product, shape in (
            (ft, (plan.time_size, plan.frequency_size)),
            (dmt, (plan.dmsteps, plan.time_size)),
        ):
            if product.shape != shape or product.dtype != plan.output_dtype:
                raise ValueError(
                    "finished candidate product has unexpected shape or dtype"
                )
            if not np.isfinite(product).all():
                raise ValueError("finished candidate products must be finite")
        self.dedispersed, self.dmt = ft, dmt
        self.processing_metadata = {
            "candidate_processing_version": 1,
            "ft_detrended": True,
            "products_normalized": True,
            "output_dtype": np.dtype(ft.dtype).name,
            "time_bin": plan.time_bin,
            "native_tsamp": plan.native_tsamp,
            "tsamp": plan.native_tsamp * plan.time_bin,
            "output_start_sample": plan.output_start,
            "input_start_sample": plan.required_start,
            "input_stop_sample": plan.required_stop,
            "padding_before": plan.pad_before,
            "padding_after": plan.pad_after,
            "time_reference": "highest_frequency",
            "dm_time_reduction": "channel_mean",
            "input_nchans": self.your_header.nchans,
            "foff": self.your_header.foff
            * self.your_header.nchans
            / plan.frequency_size,
            "dm_trials": plan.dm_values,
        }
        return self

    def make_products(
        self,
        *,
        target="CPU",
        time_size=256,
        frequency_size=256,
        dmsteps=256,
        output_dtype="float16",
        profile=None,
        cpu_threads=1,
    ):
        """Produce fixed-size, non-circular, detrended candidate products."""
        from your.utils.candidate_pipeline import make_plan, process_cpu

        self._update_id()
        plan = make_plan(self, time_size, frequency_size, dmsteps, output_dtype)
        if target == "CPU":
            ft, dmt = process_cpu(self, plan, profile=profile, threads=cpu_threads)
        elif target == "GPU":
            from your.utils.candidate_pipeline import read_plan_data
            from your.utils.candidate_scheduler import _gpu_memory_budget
            from your.utils.gpu import _cupy, estimate_process_gpu_bytes, process_gpu
            from your.utils.gpu_streaming import process_gpu_streaming

            cp = _cupy()
            with cp.cuda.Device(self.device):
                budget = _gpu_memory_budget(cp)
                streaming = (
                    estimate_process_gpu_bytes(plan, self.your_header.dtype) > budget
                )
                if not streaming:
                    data = read_plan_data(self, plan)
                    try:
                        ft, dmt = process_gpu(
                            self, plan, data=data, device=self.device, profile=profile
                        )
                    except MemoryError as error:
                        error.__traceback__ = None
                        streaming = True
                if streaming:
                    cp.get_default_memory_pool().free_all_blocks()
                    ft, dmt = process_gpu_streaming(
                        self,
                        plan,
                        memory_budget_bytes=budget,
                        device=self.device,
                        profile=profile,
                    )
        else:
            raise ValueError("target must be 'CPU' or 'GPU'")
        return self.set_processed_products(plan, ft, dmt)

    def _direct_input(self, dm_values, start, stop):
        """Read real halo samples; preserve edits to the already loaded chunk."""
        frequencies = np.asarray(self.chan_freqs, dtype=np.float64)
        if len(frequencies) != self.data.shape[1] or np.any(frequencies <= 0):
            raise ValueError("expected one positive frequency per input channel")
        dms = np.asarray(dm_values, dtype=np.float64)
        if not np.all(np.isfinite(dms)) or not np.all(np.isfinite(frequencies)):
            raise ValueError("DMs and frequencies must be finite")
        delays = np.rint(
            (4148808.0 * dms[:, None])
            * (1 / frequencies[None, :] ** 2 - 1 / frequencies.max() ** 2)
            / 1000.0
            / self.native_tsamp
        ).astype(np.int64)
        if not len(delays) or start == stop:
            return self.data[:0], delays, start
        low = start + int(delays.min())
        high = stop + int(delays.max())
        base = getattr(self, "_chunk_start", None)
        from_file = base is not None and self.data.shape == self._chunk_shape
        if from_file:
            total = int(self.your_header.nspectra)
            origin = min(total, max(0, base + low)) - base
            end = min(total, max(base + origin, base + high)) - base
            if end > origin:
                source = np.array(self.get_data(base + origin, end - origin), copy=True)
            else:
                source = np.empty((0, self.data.shape[1]), dtype=self.data.dtype)
        else:
            origin = max(0, min(len(self.data), low))
            end = max(origin, min(len(self.data), high))
            source = np.array(self.data[origin:end], copy=True)
        lo, hi = max(origin, 0), min(end, len(self.data))
        if hi > lo:
            source[lo - origin : hi - origin] = self.data[lo:hi]
        # Apply the same masks to preserved edits and newly read halo samples.
        from your.utils.candidate_contract import parse_kill_mask

        source[:, parse_kill_mask(getattr(self, "kill_mask", []), source.shape[1])] = 0
        rfi_mask = np.asarray(getattr(self, "rfi_mask", []))
        if rfi_mask.dtype.kind == "b" and rfi_mask.shape == (source.shape[1],):
            source[:, rfi_mask] = 0
        return source, delays, origin

    def _direct_cpu(self, dm_values, start, stop, *, keep_channels=False):
        source, delays, origin = self._direct_input(dm_values, start, stop)
        nf = self.data.shape[1]
        from your.utils.candidate_pipeline import _channel_medians

        fill = _channel_medians(source, nf).astype(np.float32)
        if source.dtype.kind == "f" and not np.all(np.isfinite(source)):
            source = np.where(np.isfinite(source), source, fill[None, :])
        shape = (stop - start, nf) if keep_channels else (len(delays), stop - start)
        output = np.zeros(shape, dtype=np.float32)
        times = np.arange(start, stop, dtype=np.int64)
        fully_covered = (
            len(delays) > 0
            and stop > start
            and source.dtype in _RUST_DEDISPERSETS_DTYPES
            and source.dtype != np.float64
            and start + int(delays.min()) >= origin
            and stop + int(delays.max()) <= origin + len(source)
        )
        if fully_covered:
            if keep_channels:
                return _rust_dedisperse(
                    source, -delays[0], start - origin, stop - origin
                )
            for row, offsets in enumerate(delays):
                output[row] = _rust_dedispersets(
                    source, -offsets, start - origin, stop - origin
                )
            return output / nf
        for row, offsets in enumerate(delays):
            for channel, offset in enumerate(offsets):
                indices = times + offset - origin
                valid = (indices >= 0) & (indices < len(source))
                values = np.full(stop - start, fill[channel], dtype=np.float32)
                values[valid] = source[indices[valid], channel]
                if keep_channels:
                    output[:, channel] = values
                else:
                    output[row] += values
        return output if keep_channels else output / nf

    def dedisperse(self, dms=None, target="CPU", *, time_range=None):
        """Dedisperse without wrapping; load real file halo when available."""
        if self.data is None:
            logger.warning("No data in self.data, run self.get_chunk() first")
            self.dedispersed = None
            return self
        start, stop = _time_bounds(len(self.data), time_range)
        dm = self.dm if dms is None else dms
        if target == "CPU":
            self.dedispersed = self._direct_cpu([dm], start, stop, keep_channels=True)
        elif target == "GPU":
            gpu_dedisperse(self, device=self.device, dms=dm, time_range=time_range)
        else:
            raise ValueError("target must be 'CPU' or 'GPU'")
        return self

    def dedispersets(self, dms=None, *, time_range=None):
        """Return the channel-summed, non-circular dedispersed time series."""
        if self.data is None:
            return None
        start, stop = _time_bounds(len(self.data), time_range)
        ft = self._direct_cpu(
            [self.dm if dms is None else dms], start, stop, keep_channels=True
        )
        return ft.sum(axis=1, dtype=np.float32)

    def dmtime(self, dmsteps=256, target="CPU", *, time_range=None):
        """Return channel-averaged DM trials from zero to twice the candidate DM."""
        dmsteps = index(dmsteps)
        if dmsteps < 1:
            raise ValueError("dmsteps must be positive")
        if self.data is None:
            self.dmt = None
            return self
        start, stop = _time_bounds(len(self.data), time_range)
        if target == "CPU":
            self.dmt = self._direct_cpu(
                np.linspace(0, 2 * self.dm, dmsteps), start, stop
            )
        elif target == "GPU":
            gpu_dmt(self, device=self.device, dmsteps=dmsteps, time_range=time_range)
        else:
            raise ValueError("target must be 'CPU' or 'GPU'")
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
        Interpolate between DM values to maximize SNR.

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
                f"Decimating dmt axis {axis}, factor {decimate_factor}, "
                f"shape: {self.dmt.shape}"
            )
            self.dmt = _decimate(self.dmt, decimate_factor, axis, pad, **kwargs)
            logger.debug(f"Decimated dmt axis {axis}, shape: {self.dmt.shape}")
        elif key == "ft":
            logger.debug(
                f"Decimating ft axis {axis}, factor {decimate_factor}, "
                f"shape: {self.dedispersed.shape}"
            )
            self.dedispersed = _decimate(
                self.dedispersed, decimate_factor, axis, pad, **kwargs
            )
            logger.debug(f"Decimated ft axis {axis}, shape: {self.dedispersed.shape}")
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
