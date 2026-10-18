#!/usr/bin/env python3

import argparse
import glob
import logging
import os
from time import perf_counter

from rich.logging import RichHandler

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

from your.candidate import Candidate
from your.utils.candidate_pipeline import make_compact_plan
from your.utils.candidate_scheduler import (
    CandidateJob,
    run_jobs,
)
from your.utils.misc import YourArgparseFormatter

logger = logging.getLogger()

_worker_candidate = None
_worker_candidate_key = None
_worker_candidate_finalizer = None


def cpu_dedisp_dmt(cand, args, profile=None):
    if args.opt_dm:
        cand.get_chunk(for_preprocessing=True)
        cand.optimize_dm()
    thread_options = (
        {"cpu_threads": worker_threads(args.nproc)} if hasattr(args, "nproc") else {}
    )
    return cand.make_products(
        target="CPU",
        time_size=args.time_size,
        frequency_size=args.frequency_size,
        dmsteps=args.dmsteps,
        output_dtype=args.output_dtype,
        profile=profile,
        **thread_options,
    )


def worker_threads(nproc):
    """Divide the available cores between CPU candidate workers."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    return max(1, cores // nproc)


def _input_files(filename, num_files):
    filename = os.path.realpath(os.fspath(filename))
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
    """Write one finished candidate and return its timing report."""
    started = perf_counter()
    profile = {}
    if len(cand_val) not in (10, 11):
        raise ValueError(
            "candidate input must contain 10 fields and optional row index"
        )
    row_index = int(cand_val[10]) if len(cand_val) == 11 else None
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
    ) = cand_val[:10]
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
        cand._reset_candidate(**candidate_kwargs, kill_mask=kill_mask)
    try:
        if gpu_id >= 0:
            if args.opt_dm:
                raise ValueError("DM optimization is available only on the CPU path")
            cand.make_products(
                target="GPU",
                time_size=args.time_size,
                frequency_size=args.frequency_size,
                dmsteps=args.dmsteps,
                output_dtype=args.output_dtype,
                profile=profile,
            )
        else:
            cpu_dedisp_dmt(cand, args, profile=profile)
        save_start = perf_counter()
        output_filename = None if row_index is None else f"{cand.id}_{row_index}.h5"
        if output_filename is None:
            fout = cand.save_h5(out_dir=args.fout)
        else:
            fout = cand.save_h5(out_dir=args.fout, fnout=output_filename)
        profile["save_s"] = perf_counter() - save_start
        with __import__("h5py").File(fout, "r") as output:
            expected = {
                "data_freq_time": (args.time_size, args.frequency_size),
                "data_dm_time": (args.dmsteps, args.time_size),
            }
            for name, shape in expected.items():
                if output[name].shape != shape or output[name].dtype != np.dtype(
                    args.output_dtype
                ):
                    raise ValueError(f"Invalid candidate dataset {name} in {fout}")
    finally:
        if candidate is None:
            cand.close()
    logger.info(fout)
    if candidate is None:
        del cand
    else:
        cand.data = cand.dedispersed = cand.dmt = None
    profile["total_s"] = perf_counter() - started
    return {"output_file": fout, "device": gpu_id, "profile": profile}


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
        Make fixed-size frequency-time and DM-time HDF5 candidate products.
        The CSV fields are:
            file: Filterbank or first PSRFITS filename.
            snr: Candidate signal-to-noise ratio.
            width: log2(pulse width in native samples).
            dm: Candidate dispersion measure.
            label: Candidate label, or zero if unknown.
            stime: Arrival time in seconds at the highest-frequency channel.
            chan_mask_path: Channel mask filename, or empty.
            num_files: Number of files in the observation.
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
        help="GPU IDs, for example -g 2 3; use -1 alone for CPU",
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
        "--dmsteps",
        type=int,
        default=256,
        help="Number of DM trials spanning zero to twice candidate DM",
    )
    parser.add_argument(
        "--output_dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument(
        "--gpu_memory_mb",
        type=int,
        default=None,
        help="Bound GPU working memory in MiB (default: available memory budget)",
    )
    parser.add_argument(
        "--gpu_inflight",
        type=int,
        default=2,
        help="Maximum simultaneous candidates per GPU",
    )
    parser.add_argument(
        "--profile_json",
        default=None,
        help="Write GPU stage timings and I/O counters as JSON",
    )
    parser.add_argument(
        "--no_log_file", help="Do not write a log file", action="store_true"
    )
    values = parser.parse_args()
    os.makedirs(values.fout, exist_ok=True)

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

    if any(gpu < -1 for gpu in values.gpu_id) or (
        -1 in values.gpu_id and len(values.gpu_id) > 1
    ):
        parser.error("use -1 alone for CPU, or one or more nonnegative GPU IDs")
    if len(set(values.gpu_id)) != len(values.gpu_id):
        parser.error("GPU IDs must be unique")
    if (
        min(
            values.time_size,
            values.frequency_size,
            values.dmsteps,
            values.gpu_inflight,
            values.nproc,
        )
        < 1
    ):
        parser.error(
            "output dimensions, dmsteps, gpu_inflight, and nproc must be positive"
        )
    if values.gpu_memory_mb is not None and values.gpu_memory_mb <= 0:
        parser.error("gpu_memory_mb must be positive")
    if values.opt_dm and values.gpu_id != [-1]:
        parser.error("DM optimization is available only on CPU")
    os.makedirs(values.fout, exist_ok=True)
    cand_pars = pd.read_csv(values.cand_param_file)
    process_list = []
    for _, row in cand_pars.iterrows():
        process_list.append(
            [
                row["file"],
                row["snr"],
                2 ** int(row["width"]),
                row["dm"],
                row["label"],
                row["stime"],
                row["chan_mask_path"],
                int(row["num_files"]),
                values,
                values.gpu_id[0],
            ]
        )

    if values.gpu_id == [-1]:
        with Pool(processes=values.nproc, initializer=_init_cand_worker) as pool:
            cpu_process_list = [
                [*item, index] for index, item in enumerate(process_list)
            ]
            reports = pool.map(cached_cand2h5, cpu_process_list, chunksize=1)
        if values.profile_json:
            import json
            from pathlib import Path

            Path(values.profile_json).write_text(json.dumps(reports, indent=2))
    else:
        jobs = []
        rows_by_files = {}
        for i, item in enumerate(process_list):
            filename, nfiles = item[0], item[7]
            files = tuple(_input_files(filename, nfiles))
            rows_by_files.setdefault(files, []).append((i, item))

        output_names = set()
        for files, rows in rows_by_files.items():
            reader = Candidate(list(files), dm=rows[0][1][3])
            try:
                for i, item in rows:
                    (
                        filename,
                        snr,
                        width,
                        dm,
                        label,
                        tcand,
                        mask_path,
                        nfiles,
                        _,
                        _,
                    ) = item
                    kwargs = dict(
                        dm=dm,
                        snr=snr,
                        width=width,
                        label=label,
                        tcand=tcand,
                        flag_rfi=values.flag_rfi,
                        spectral_kurtosis_sigma=values.spectral_kurtosis_sigma,
                        savgol_frequency_window=values.savgol_frequency_window,
                        savgol_sigma=values.savgol_sigma,
                    )
                    if os.path.exists(str(mask_path)):
                        mask = np.zeros(reader.nchans, dtype=bool)
                        mask[np.atleast_1d(np.loadtxt(mask_path, dtype=np.int32))] = (
                            True
                        )
                        kwargs["kill_mask"] = mask
                    reader._reset_candidate(**kwargs)

                    output_filename = f"{reader.id}.h5"
                    if output_filename in output_names:
                        output_filename = f"{reader.id}_{i}.h5"
                    suffix = 1
                    while output_filename in output_names:
                        output_filename = f"{reader.id}_{i}_{suffix}.h5"
                        suffix += 1

                    plan = make_compact_plan(
                        reader,
                        values.time_size,
                        values.frequency_size,
                        values.dmsteps,
                        values.output_dtype,
                    )
                    jobs.append(CandidateJob(files, kwargs, plan, i, output_filename))
                    output_names.add(output_filename)
            finally:
                reader.close()
        budget = (
            None if values.gpu_memory_mb is None else values.gpu_memory_mb * 1024**2
        )
        reports = run_jobs(
            jobs,
            gpu_ids=values.gpu_id,
            output_dir=values.fout,
            memory_budget_bytes=budget,
            max_inflight=values.gpu_inflight,
            in_process=(len(values.gpu_id) == 1),
        )
        if values.profile_json:
            import json
            from pathlib import Path

            Path(values.profile_json).write_text(json.dumps(reports, indent=2))
