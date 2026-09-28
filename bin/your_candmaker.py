#!/usr/bin/env python3

import argparse
import glob
import logging
import os

from rich.logging import RichHandler

from your.utils.math import normalise

os.environ["OPENBLAS_NUM_THREADS"] = (
    "1"  # stop numpy multithreading regardless of the backend
)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import textwrap
from datetime import datetime
from multiprocessing import Pool
from multiprocessing.util import Finalize

import numpy as np
import pandas as pd

from your.candidate import Candidate, crop
from your.utils.gpu import PinnedReadBuffer, gpu_dedisp_and_dmt_crop
from your.utils.misc import YourArgparseFormatter

logger = logging.getLogger()

_worker_candidate = None
_worker_candidate_key = None
_worker_candidate_finalizer = None
# this process's page-locked read buffer, made on its first GPU candidate
_read_buffer = None


def cpu_dedisp_dmt(cand, args):
    pulse_width = cand.width
    if pulse_width == 1:
        time_decimate_factor = 1
    else:
        time_decimate_factor = pulse_width // 2
    logger.debug(f"Time decimation factor {time_decimate_factor}")

    # Plan the same crop on the padded, decimated time axis. Only bypass full
    # arrays when every requested bin lies inside the unpadded shifted data.
    nt = cand.data.shape[0]
    time_range = None
    if time_decimate_factor > 0 and args.time_size > 0:
        decimated_nt = (nt + time_decimate_factor - 1) // time_decimate_factor
        crop_start = decimated_nt // 2 - args.time_size // 2
        crop_stop = crop_start + args.time_size
        if (
            crop_start >= 0
            and (crop_stop < decimated_nt or args.time_size == decimated_nt)
            and crop_stop * time_decimate_factor <= nt
        ):
            time_range = (
                crop_start * time_decimate_factor,
                crop_stop * time_decimate_factor,
            )
    # ponytail: crops needing median padding retain the full-array path;
    # add bounded median handling only if those cases dominate real workloads.
    cand.dmtime(time_range=time_range)
    logger.info("Made DMT")
    if args.opt_dm:
        logger.info("Optimising DM")
        logger.warning("This feature is experimental!")
        cand.optimize_dm()
    else:
        cand.dm_opt = -1
        cand.snr_opt = -1
    cand.dedisperse(time_range=time_range)
    logger.info("Made Dedispersed profile")

    # Frequency - Time reshaping
    if time_decimate_factor != 1:
        cand.decimate(
            key="ft",
            axis=0,
            pad=True,
            decimate_factor=time_decimate_factor,
            mode="median",
        )
    if time_range is None:
        crop_start_sample_ft = cand.dedispersed.shape[0] // 2 - args.time_size // 2
        cand.dedispersed = crop(
            cand.dedispersed, crop_start_sample_ft, args.time_size, 0
        )
    logger.info(f"Decimated Time axis of FT to tsize: {cand.dedispersed.shape[0]}")
    # DM-time reshaping
    if time_decimate_factor != 1:
        cand.decimate(
            key="dmt",
            axis=1,
            pad=True,
            decimate_factor=time_decimate_factor,
            mode="median",
        )
    if time_range is None:
        crop_start_sample_dmt = cand.dmt.shape[1] // 2 - args.time_size // 2
        cand.dmt = crop(cand.dmt, crop_start_sample_dmt, args.time_size, 1)
    logger.info(
        f"Decimated DM-Time to dmsize: {cand.dmt.shape[0]} and tsize: {cand.dmt.shape[1]}"
    )
    return cand


def _input_files(filename, num_files):
    fname, ext = os.path.splitext(filename)
    if ext == ".fits" or ext == ".sf":
        if num_files == 1:
            return [filename]
        files = glob.glob(fname[:-5] + "*fits")
        if len(files) != num_files:
            raise ValueError(
                "Number of fits files found was not equal to num_files in cand csv."
            )
        return sorted(files)
    if ext == ".fil":
        return [filename]
    raise TypeError("Can only work with list of fits file or filterbanks")


def _reader_key(files):
    return tuple(
        (path, stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        for path in files
        for stat in [os.stat(path)]
    )


def _clear_worker_candidate():
    global _worker_candidate, _worker_candidate_key
    if _worker_candidate is not None:
        try:
            _worker_candidate.fits.close()
        except Exception:
            logger.debug("Could not close cached FITS reader", exc_info=True)
    _worker_candidate = None
    _worker_candidate_key = None


def _init_cand_worker():
    global _worker_candidate_finalizer
    _clear_worker_candidate()
    _worker_candidate_finalizer = Finalize(
        None, _clear_worker_candidate, exitpriority=10
    )


def cand2h5(cand_val):
    return _cand2h5(cand_val)


def _cand2h5(cand_val, *, candidate=None, files=None):
    """
    TODO: Add option to use cand.resize for reshaping FT and DMT
    Generates h5 file of candidate with resized frequency-time and DM-time arrays
    :param cand_val: List of candidate parameters (filename, snr, width, dm, label, tcand(s))
    :type cand_val: Candidate
    :return: None
    """
    (
        filename,
        snr,
        width,
        dm,
        label,
        tcand,
        kill_mask_path,
        num_files,
        args,
        gpu_id,
    ) = cand_val
    if os.path.exists(str(kill_mask_path)):
        logger.info(f"Using mask {kill_mask_path}")
        kill_chans = np.loadtxt(kill_mask_path, dtype=np.int32)
    else:
        logger.debug("No Kill Mask")

    if files is None:
        files = _input_files(filename, num_files)

    logger.debug(f"Source file list: {files}")
    candidate_kwargs = dict(
        snr=snr,
        width=width,
        dm=dm,
        label=label,
        tcand=tcand,
        device=gpu_id,
        spectral_kurtosis_sigma=args.spectral_kurtosis_sigma,
        savgol_frequency_window=args.savgol_frequency_window,
        savgol_sigma=args.savgol_sigma,
        flag_rfi=args.flag_rfi,
    )
    if candidate is None:
        cand = Candidate(files, **candidate_kwargs)
    else:
        cand = candidate._reset_candidate(**candidate_kwargs)
    if os.path.exists(str(kill_mask_path)):
        kill_mask = np.zeros(cand.nchans, dtype=np.bool_)
        kill_mask[kill_chans] = True
        cand.kill_mask = kill_mask
    if gpu_id >= 0:
        # this worker makes one candidate at a time, so every chunk can be
        # read into the same page-locked buffer and uploaded from there, to
        # whichever GPU the candidate goes to
        global _read_buffer
        if _read_buffer is None:
            _read_buffer = PinnedReadBuffer(gpu_id)
        cand.read_buffer = _read_buffer
    else:
        cand.read_buffer = None
    cand.get_chunk(for_preprocessing=True)
    if cand.format == "fil":
        cand.fp.close()

    logger.info("Got Chunk")

    if gpu_id >= 0:
        logger.debug(f"Using the GPU {gpu_id}")
        try:
            cand = gpu_dedisp_and_dmt_crop(cand, device=gpu_id)
        except CudaAPIError:
            logger.info(
                "Ran into a CudaAPIError, using the CPU version for this candidate"
            )
            cand = cpu_dedisp_dmt(cand, args)
    else:
        cand = cpu_dedisp_dmt(cand, args)

    cand.resize(
        key="ft", size=args.frequency_size, axis=1, anti_aliasing=True, mode="constant"
    )
    logger.info(f"Resized Frequency axis of FT to fsize: {cand.dedispersed.shape[1]}")
    cand.dmt = normalise(cand.dmt)
    cand.dedispersed = normalise(cand.dedispersed)
    fout = cand.save_h5(out_dir=args.fout)
    logger.debug(f"Filesize of {fout} is {os.path.getsize(fout)}")
    if not os.path.isfile(fout):
        raise IOError(f"File with {cand.id} not written")
    if os.path.getsize(fout) < 100 * 1024:
        raise ValueError(f"File with id: {cand.id} has issues! Its size is too less.")
    logger.info(fout)
    if candidate is None:
        del cand
    else:
        cand.data = cand.dedispersed = cand.dmt = None
    return None


def cached_cand2h5(cand_val):
    """Reuse one FITS Candidate within each serial Pool worker."""
    global _worker_candidate, _worker_candidate_key
    try:
        files = _input_files(cand_val[0], cand_val[7])
        if os.path.splitext(cand_val[0])[1] not in (".fits", ".sf"):
            _clear_worker_candidate()
            return _cand2h5(cand_val, files=files)

        key = _reader_key(files)
        if key != _worker_candidate_key:
            _clear_worker_candidate()
            _worker_candidate = Candidate(files, dm=cand_val[3])
            _worker_candidate_key = key
        return _cand2h5(cand_val, candidate=_worker_candidate, files=files)
    except BaseException:
        _clear_worker_candidate()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="your_candmaker.py",
        description="Your candmaker! Make h5 candidates from the candidate csv files",
        formatter_class=YourArgparseFormatter,
        epilog=textwrap.dedent(
            """\
        `your_candmaker.py` can be used to make candidate cutout files. Some additional notes for this script: 
        - These files are generated in HDF file format. 
        - The output candidates have been preprocessed and consists of Dedispersed Frequency-Time and DM-Time information of the candidate. 
        - The input should be a csv file containing the parameters of the candidates. The input csv file should contain the following fields: 
                - file: Filterbank or PSRFITs file containing the data. In case of multiple files, this should contain the name of first file. 
                - snr: Signal to Noise of the candidate.
                - width: Width of candidate as log2(number of samples). 
                - dm: DM of candidate
                - label: Label of candidate (can be just set to 0, if not known)
                - stime: Start time (seconds) of the candidate.
                - chan_mask_path: Path of the channel mask file. 
                - num_files: Number of files. 
            """
        ),
    )
    parser.add_argument("-v", "--verbose", help="Be verbose", action="store_true")
    parser.add_argument(
        "-fs",
        "--frequency_size",
        type=int,
        help="Frequency size after rebinning",
        default=256,
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU). To use multiple GPUs (say with id 2 and 3 use -g 2 3",
        nargs="+",
        required=False,
        default=[-1],
        type=int,
    )
    parser.add_argument(
        "-ts", "--time_size", type=int, help="Time length after rebinning", default=256
    )
    parser.add_argument(
        "-c",
        "--cand_param_file",
        help="csv file with candidate parameters",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        help="number of processors to use in parallel (default: 2)",
        default=2,
    )
    parser.add_argument(
        "-o",
        "--fout",
        help="Output file directory for candidate h5",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-r",
        "--flag_rfi",
        help="Turn on RFI flagging",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-sksig",
        "--spectral_kurtosis_sigma",
        help="Sigma for spectral kurtosis filter",
        type=float,
        default=4,
        required=False,
    )
    parser.add_argument(
        "-sgsig",
        "--savgol_sigma",
        help="Sigma for savgol filter",
        type=float,
        default=4,
        required=False,
    )
    parser.add_argument(
        "-sgfw",
        "--savgol_frequency_window",
        help="Filter window for savgol filter (MHz)",
        type=float,
        default=15,
        required=False,
    )
    parser.add_argument(
        "-opt",
        "--opt_dm",
        dest="opt_dm",
        help="Optimise DM",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no_log_file", help="Do not write a log file", action="store_true"
    )
    values = parser.parse_args()

    logging_format = (
        "%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s"
    )
    log_filename = (
        values.fout
        + "/"
        + datetime.utcnow().strftime("your_candmaker_%Y_%m_%d_%H_%M_%S_%f.log")
    )

    if not values.no_log_file:
        if values.verbose:
            logging.basicConfig(
                filename=log_filename,
                level=logging.DEBUG,
                format=logging_format,
            )
        else:
            logging.basicConfig(
                filename=log_filename, level=logging.INFO, format=logging_format
            )
    else:
        if values.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format=logging_format,
                handlers=[RichHandler(rich_tracebacks=True)],
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=logging_format,
                handlers=[RichHandler(rich_tracebacks=True)],
            )

    logging.info("Input Arguments:-")
    for arg, value in sorted(vars(values).items()):
        logging.info("%s: %r", arg, value)

    if -1 not in values.gpu_id:
        from numba.cuda.cudadrv.driver import CudaAPIError

        for gpu_ids in values.gpu_id:
            logger.info(f"Using the GPU {gpu_ids}")
        if len(values.gpu_id) > 1:
            from itertools import cycle

            gpu_id_cycler = cycle(range(len(values.gpu_id)))
    else:
        logger.info("Using CPUs only")

    cand_pars = pd.read_csv(values.cand_param_file)
    # Randomly shuffle the candidates, this is so that the high DM candidates are spread through out
    # Else they will clog the GPU memory at once
    cand_pars.sample(frac=1).reset_index(drop=True)
    process_list = []
    for index, row in cand_pars.iterrows():
        if len(values.gpu_id) > 1:
            # If there are more than one GPUs cycle the candidates between them.
            gpu_id = next(gpu_id_cycler)
        else:
            gpu_id = values.gpu_id[0]
        process_list.append(
            [
                row["file"],
                row["snr"],
                2 ** row["width"],
                row["dm"],
                row["label"],
                row["stime"],
                row["chan_mask_path"],
                row["num_files"],
                values,
                gpu_id,
            ]
        )

    with Pool(processes=values.nproc, initializer=_init_cand_worker) as pool:
        pool.map(cached_cand2h5, process_list, chunksize=1)
