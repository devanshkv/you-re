"""Spawned, per-GPU scheduler for candidate product generation."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class CandidateJob:
    """One candidate and its already-computed output plan."""

    files: tuple[str, ...]
    candidate_kwargs: Mapping[str, Any]
    plan: Any
    index: int
    output_filename: str | None = None

    def __post_init__(self):
        files = tuple(os.fspath(path) for path in self.files)
        if not files:
            raise ValueError("candidate job files must not be empty")
        if not isinstance(self.candidate_kwargs, Mapping):
            raise TypeError("candidate_kwargs must be a mapping")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "candidate_kwargs", dict(self.candidate_kwargs))
        object.__setattr__(self, "index", int(self.index))
        if self.output_filename is not None:
            filename = os.fspath(self.output_filename)
            if not filename or Path(filename).name != filename:
                raise ValueError("output_filename must be a non-empty filename")
            object.__setattr__(self, "output_filename", filename)


@dataclass(frozen=True)
class CompactCandidatePlan:
    """Scalar candidate plan metadata; delay tables are built only when run."""

    time_size: int
    frequency_size: int
    dmsteps: int
    output_dtype: Any
    native_count: int
    required_start: int
    required_stop: int
    read_start: int
    read_stop: int
    nspectra: int
    native_tsamp: float
    nchans: int
    pad_before: int
    pad_after: int

    @classmethod
    def from_plan(cls, plan):
        return cls(
            time_size=int(plan.time_size),
            frequency_size=int(plan.frequency_size),
            dmsteps=int(plan.dmsteps),
            output_dtype=np.dtype(plan.output_dtype),
            native_count=int(plan.native_count),
            required_start=int(plan.required_start),
            required_stop=int(plan.required_stop),
            read_start=int(plan.read_start),
            read_stop=int(plan.read_stop),
            nspectra=int(plan.nspectra),
            native_tsamp=float(plan.native_tsamp),
            nchans=int(plan.nchans),
            pad_before=int(plan.pad_before),
            pad_after=int(plan.pad_after),
        )

    def materialize(self, candidate):
        from your.utils.candidate_pipeline import make_plan

        return make_plan(
            candidate,
            self.time_size,
            self.frequency_size,
            self.dmsteps,
            self.output_dtype,
        )


@dataclass(frozen=True)
class CandidateBatch:
    """A set of candidates that share one resident input interval."""

    jobs: tuple[CandidateJob, ...]
    read_start: int
    read_stop: int
    estimated_peak_bytes: int
    workspace_bytes: int
    streaming: bool = False
    drop_cache_before: bool = False


@dataclass(frozen=True)
class _Work:
    files: tuple[str, ...]
    jobs: tuple[CandidateJob, ...]


def choose_memory_budget(free_bytes: int, safety_fraction: float = 0.8) -> int:
    """Return the fraction of currently free device memory reserved for this run."""
    if int(free_bytes) < 0:
        raise ValueError("free_bytes must be non-negative")
    if not 0 < float(safety_fraction) <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    return int(int(free_bytes) * float(safety_fraction))


def _job_interval(
    job: CandidateJob, start_name: str, stop_name: str
) -> tuple[int, int]:
    start = int(getattr(job.plan, start_name))
    stop = int(getattr(job.plan, stop_name))
    if stop < start:
        raise ValueError(
            f"candidate {job.index} has an invalid {start_name}/{stop_name} interval"
        )
    return start, stop


def group_jobs_by_files(
    jobs: Iterable[CandidateJob],
) -> list[tuple[tuple[str, ...], tuple[CandidateJob, ...]]]:
    """Group exact input tuples and sort jobs by their required intervals."""
    groups: dict[tuple[str, ...], list[CandidateJob]] = {}
    indices: set[int] = set()
    for job in jobs:
        if job.index in indices:
            raise ValueError(f"duplicate candidate job index {job.index}")
        indices.add(job.index)
        groups.setdefault(job.files, []).append(job)
    return [
        (
            files,
            tuple(
                sorted(
                    group,
                    key=lambda job: (
                        *_job_interval(job, "required_start", "required_stop"),
                        job.index,
                    ),
                )
            ),
        )
        for files, group in sorted(groups.items())
    ]


def _resident_bytes(
    interval: tuple[int, int] | None, nchans: int, itemsize: int
) -> int:
    if interval is None:
        return 0
    return max(0, interval[1] - interval[0]) * nchans * itemsize


def _missing_intervals(
    current: tuple[int, int] | None, target: tuple[int, int]
) -> list[tuple[int, int]]:
    """Return target spans absent from the current resident allocation."""
    start, stop = target
    if current is None or max(current[0], start) >= min(current[1], stop):
        return [(start, stop)] if stop > start else []
    missing = []
    if start < current[0]:
        missing.append((start, min(stop, current[0])))
    if stop > current[1]:
        missing.append((max(start, current[1]), stop))
    return [(left, right) for left, right in missing if right > left]


def _upload_samples(missing, nchans, itemsize, available_bytes=None):
    limit = 64 * 1024 * 1024
    if available_bytes is not None:
        limit = min(limit, available_bytes)
    return min(
        max((stop - start for start, stop in missing), default=0),
        max(1, limit // (nchans * itemsize)),
    )


def _transition_peak(
    current: tuple[int, int] | None,
    target: tuple[int, int],
    nchans: int,
    itemsize: int,
    workspace_bytes: int,
    memory_budget_bytes: int,
) -> tuple[int, tuple[int, int], bool]:
    old_bytes = _resident_bytes(current, nchans, itemsize)
    new_bytes = _resident_bytes(target, nchans, itemsize)

    def upload_peak(previous):
        live_bytes = new_bytes + _resident_bytes(previous, nchans, itemsize)
        samples = _upload_samples(
            _missing_intervals(previous, target),
            nchans,
            itemsize,
            memory_budget_bytes - live_bytes,
        )
        return live_bytes + samples * nchans * itemsize

    reload_peak = max(old_bytes, upload_peak(None), new_bytes + workspace_bytes)
    drop_cache = False
    if current is None:
        resize_peak = upload_peak(None)
        final_interval = target
    elif current[0] <= target[0] and target[1] <= current[1]:
        resize_peak = old_bytes
        final_interval = current
        if (
            current != target
            and old_bytes + workspace_bytes > memory_budget_bytes
            and reload_peak <= memory_budget_bytes
        ):
            resize_peak = reload_peak
            final_interval = target
            drop_cache = True
    elif max(current[0], target[0]) < min(current[1], target[1]):
        resize_peak = upload_peak(current)
        if resize_peak > memory_budget_bytes and reload_peak <= memory_budget_bytes:
            # Re-read the target after releasing the overlapping allocation.
            resize_peak = reload_peak
            drop_cache = True
        final_interval = target
    else:
        # Disjoint windows can release the old allocation before replacing it.
        resize_peak = max(old_bytes, upload_peak(None))
        final_interval = target
    peak = max(
        resize_peak, _resident_bytes(final_interval, nchans, itemsize) + workspace_bytes
    )
    return peak, final_interval, drop_cache


def plan_batches(
    jobs: Iterable[CandidateJob],
    *,
    nchans: int,
    input_dtype: Any,
    memory_budget_bytes: int,
    workspace_estimator: Callable[[Any, np.dtype], int],
    max_inflight: int = 2,
    initial_resident: tuple[int, int] | None = None,
) -> list[CandidateBatch]:
    """Plan bounded batches using resident, resize-peak, and per-job workspace bytes."""
    jobs = list(jobs)
    if not jobs:
        return []
    if max_inflight < 1:
        raise ValueError("max_inflight must be at least one")
    if nchans < 1:
        raise ValueError("nchans must be positive")
    if memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive")
    if len({job.files for job in jobs}) != 1:
        raise ValueError("plan_batches accepts jobs from one input file tuple")
    dtype = np.dtype(input_dtype)
    itemsize = dtype.itemsize
    if initial_resident is not None and initial_resident[1] < initial_resident[0]:
        raise ValueError("initial_resident must be an increasing interval")

    ordered = sorted(
        jobs,
        key=lambda job: (
            *_job_interval(job, "required_start", "required_stop"),
            job.index,
        ),
    )
    batches: list[CandidateBatch] = []
    resident = initial_resident
    active: list[CandidateJob] = []
    active_workspace = 0

    def metrics(candidate_jobs: list[CandidateJob], workspace: int):
        start = min(
            _job_interval(job, "read_start", "read_stop")[0] for job in candidate_jobs
        )
        stop = max(
            _job_interval(job, "read_start", "read_stop")[1] for job in candidate_jobs
        )
        if stop < start:
            raise ValueError(f"batch has an invalid read interval {(start, stop)}")
        peak, final_interval, drop_cache = _transition_peak(
            resident,
            (start, stop),
            nchans,
            itemsize,
            workspace,
            memory_budget_bytes,
        )
        return (start, stop), peak, final_interval, drop_cache

    def flush():
        nonlocal active, active_workspace, resident
        if not active:
            return
        interval, peak, final_interval, drop_cache = metrics(active, active_workspace)
        batches.append(
            CandidateBatch(
                tuple(active),
                interval[0],
                interval[1],
                peak,
                active_workspace,
                drop_cache_before=drop_cache,
            )
        )
        resident = final_interval
        active = []
        active_workspace = 0

    for job in ordered:
        read_interval = _job_interval(job, "read_start", "read_stop")
        if read_interval[1] < read_interval[0]:
            raise ValueError(f"candidate {job.index} has an invalid read interval")
        workspace = int(workspace_estimator(job.plan, dtype))
        if workspace < 0:
            raise ValueError("workspace estimates must be non-negative")

        proposed = active + [job]
        proposed_workspace = active_workspace + workspace
        if len(proposed) <= max_inflight:
            interval, peak, _, _ = metrics(proposed, proposed_workspace)
        else:
            peak = memory_budget_bytes + 1

        if peak <= memory_budget_bytes:
            active = proposed
            active_workspace = proposed_workspace
            continue

        flush()
        interval, peak, _, _ = metrics([job], workspace)
        if peak > memory_budget_bytes:
            batches.append(
                CandidateBatch(
                    (job,), interval[0], interval[1], peak, workspace, streaming=True
                )
            )
            resident = None
            continue
        active = [job]
        active_workspace = workspace

    flush()
    return batches


def _assign_work(
    groups: list[tuple[tuple[str, ...], tuple[CandidateJob, ...]]],
    gpu_ids: tuple[int, ...],
) -> dict[int, list[_Work]]:
    assigned = {gpu_id: [] for gpu_id in gpu_ids}
    loads = {gpu_id: 0 for gpu_id in gpu_ids}
    for files, jobs in groups:
        count = min(len(jobs), len(gpu_ids))
        selected = sorted(gpu_ids, key=lambda gpu_id: (loads[gpu_id], gpu_id))[:count]
        selected.sort(key=gpu_ids.index)
        for chunk_index, gpu_id in enumerate(selected):
            start = len(jobs) * chunk_index // count
            stop = len(jobs) * (chunk_index + 1) // count
            chunk = jobs[start:stop]
            assigned[gpu_id].append(_Work(files, chunk))
            loads[gpu_id] += len(chunk)
    return assigned


class _ResidentCache:
    def __init__(
        self,
        cp,
        reader,
        device: int,
        nchans: int,
        dtype: np.dtype,
        memory_budget_bytes: int | None = None,
    ):
        self.cp = cp
        self.reader = reader
        self.device = device
        self.nchans = nchans
        self.dtype = dtype
        self.memory_budget_bytes = memory_budget_bytes
        self.array = None
        self.start = None
        self.stream = cp.cuda.Stream(non_blocking=True)

    @property
    def interval(self):
        if self.array is None:
            return None
        return self.start, self.start + self.array.shape[1]

    def ensure(self, start: int, stop: int) -> dict[str, float | int]:
        if (
            self.array is not None
            and self.start <= start
            and stop <= self.start + self.array.shape[1]
        ):
            return {"read_bytes": 0, "upload_bytes": 0, "read_s": 0.0, "upload_s": 0.0}

        from your.utils.gpu import _corner_turn

        cp = self.cp
        previous = self.array
        previous_start = self.start
        previous_stop = (
            previous_start + previous.shape[1] if previous is not None else None
        )
        overlaps = previous is not None and max(previous_start, start) < min(
            previous_stop, stop
        )
        if not overlaps:
            previous = None
            previous_start = None
            previous_stop = None
            self.array = None
            self.start = None

        with cp.cuda.Device(self.device):
            new = cp.empty((self.nchans, stop - start), dtype=self.dtype)

        if previous is not None:
            overlap_start = max(start, previous_start)
            overlap_stop = min(stop, previous_stop)
            if overlap_stop > overlap_start:
                dst = new.data.ptr + (overlap_start - start) * self.dtype.itemsize
                src = (
                    previous.data.ptr
                    + (overlap_start - previous_start) * self.dtype.itemsize
                )
                cp.cuda.runtime.memcpy2DAsync(
                    dst,
                    new.strides[0],
                    src,
                    previous.strides[0],
                    (overlap_stop - overlap_start) * self.dtype.itemsize,
                    self.nchans,
                    cp.cuda.runtime.memcpyDeviceToDevice,
                    self.stream.ptr,
                )
                self.stream.synchronize()

        read_bytes = 0
        upload_bytes = 0
        read_seconds = 0.0
        upload_seconds = 0.0
        previous_interval = (
            (previous_start, previous_stop) if previous is not None else None
        )
        missing = _missing_intervals(previous_interval, (start, stop))
        available_bytes = (
            None
            if self.memory_budget_bytes is None
            else self.memory_budget_bytes
            - new.nbytes
            - (previous.nbytes if previous is not None else 0)
        )
        samples_per_upload = _upload_samples(
            missing, self.nchans, self.dtype.itemsize, available_bytes
        )
        if missing:
            if (
                available_bytes is not None
                and available_bytes < self.nchans * self.dtype.itemsize
            ):
                raise MemoryError("GPU cache upload staging exceeds the memory budget")
            # Keep staging out of the pool so job workspaces can use its memory.
            with cp.cuda.Device(self.device), cp.cuda.using_allocator(None):
                try:
                    staging = cp.empty(
                        (samples_per_upload, self.nchans), dtype=self.dtype
                    )
                except Exception as exc:
                    if not _is_out_of_memory(exc):
                        raise
                    cp.get_default_memory_pool().free_all_blocks()
                    staging = cp.empty(
                        (samples_per_upload, self.nchans), dtype=self.dtype
                    )

        for missing_start, missing_stop in missing:
            for read_start in range(missing_start, missing_stop, samples_per_upload):
                read_stop = min(read_start + samples_per_upload, missing_stop)
                t0 = perf_counter()
                host_time_major = np.asarray(
                    self.reader.get_data(read_start, read_stop - read_start)
                )
                expected = (read_stop - read_start, self.nchans)
                if host_time_major.shape != expected:
                    raise ValueError(
                        f"reader returned shape {host_time_major.shape}; "
                        f"expected {expected}"
                    )
                if host_time_major.dtype != self.dtype:
                    raise TypeError(
                        f"reader returned dtype {host_time_major.dtype}; "
                        f"expected decoded sample dtype {self.dtype}"
                    )
                count = (read_stop - read_start) * self.nchans
                pinned = cp.cuda.alloc_pinned_memory(count * self.dtype.itemsize)
                host = np.frombuffer(pinned, dtype=self.dtype, count=count).reshape(
                    read_stop - read_start, self.nchans
                )
                np.copyto(host, host_time_major, casting="no")
                read_seconds += perf_counter() - t0
                read_bytes += int(host.nbytes)
                upload_bytes += int(host.nbytes)

                upload_start = cp.cuda.Event()
                upload_stop = cp.cuda.Event()
                upload_start.record(self.stream)
                try:
                    cp.cuda.runtime.memcpyAsync(
                        staging.data.ptr,
                        host.ctypes.data,
                        host.nbytes,
                        cp.cuda.runtime.memcpyHostToDevice,
                        self.stream.ptr,
                    )
                    _corner_turn(
                        cp,
                        staging,
                        new,
                        read_stop - read_start,
                        read_start - start,
                        self.stream,
                    )
                    upload_stop.record(self.stream)
                    upload_stop.synchronize()
                except BaseException:
                    self.stream.synchronize()
                    raise
                upload_seconds += (
                    cp.cuda.get_elapsed_time(upload_start, upload_stop) / 1000.0
                )
                del host, pinned, host_time_major

        if missing:
            del staging
        self.array = new
        self.start = start
        del previous
        return {
            "read_bytes": read_bytes,
            "upload_bytes": upload_bytes,
            "read_s": read_seconds,
            "upload_s": upload_seconds,
        }

    def close(self):
        self.stream.synchronize()
        self.array = None
        self.start = None


def _candidate_for_job(base, job: CandidateJob, device: int):
    candidate = copy.copy(base)
    candidate.your_header = copy.copy(base.your_header)
    kwargs = dict(job.candidate_kwargs)
    kwargs.pop("fp", None)
    kwargs["device"] = device
    candidate._reset_candidate(**kwargs)
    return candidate


def _materialize_plan(candidate, plan):
    materialize = getattr(plan, "materialize", None)
    return materialize(candidate) if callable(materialize) else plan


def _run_gpu_job(cp, process_gpu, base, job, resident, resident_start, device):
    with cp.cuda.Device(device):
        stream = cp.cuda.Stream(non_blocking=True)
        candidate = _candidate_for_job(base, job, device)
        plan = _materialize_plan(candidate, job.plan)
        profile: dict[str, Any] = {}
        start = perf_counter()
        ft, dmt = process_gpu(
            candidate,
            plan,
            data=None,
            device=device,
            resident=resident,
            resident_start=resident_start,
            stream=stream,
            profile=profile,
        )
        stream.synchronize()
        candidate.set_processed_products(plan, ft, dmt)
        return candidate, profile, perf_counter() - start


def _run_streamed_job(
    cp, process_gpu_streaming, base, job, device, memory_budget_bytes
):
    with cp.cuda.Device(device):
        candidate = _candidate_for_job(base, job, device)
        plan = _materialize_plan(candidate, job.plan)
        profile: dict[str, Any] = {}
        start = perf_counter()
        ft, dmt = process_gpu_streaming(
            candidate,
            plan,
            memory_budget_bytes=memory_budget_bytes,
            device=device,
            profile=profile,
        )
        candidate.set_processed_products(plan, ft, dmt)
        return candidate, profile, perf_counter() - start


def _queue_candidate_save(
    writer,
    candidate,
    job,
    profile,
    process_seconds,
    device,
    output_dir,
    *,
    details=None,
):
    profile.update(details or {})
    profile["gpu_processing_s"] = float(process_seconds)
    profile["total_without_save_s"] = float(
        process_seconds
        + profile.get("cache_read_s", 0.0)
        + profile.get("cache_upload_s", 0.0)
        + profile.get("runtime_oom_s", 0.0)
    )
    return {
        "future": writer.submit(
            _save_candidate, candidate, output_dir, job.output_filename
        ),
        "device": device,
        "job": job,
        "profile": profile,
    }


def _complete_candidate_save(pending):
    output_file, save_seconds = pending["future"].result()
    job = pending["job"]
    profile = pending["profile"]
    profile["save_s"] = float(save_seconds)
    profile["total_s"] = float(profile.pop("total_without_save_s") + save_seconds)
    profile = _json_safe(profile)
    return {
        "type": "result",
        "device": pending["device"],
        "index": job.index,
        "files": job.files,
        "output_file": output_file,
        "read_bytes": int(profile.get("read_bytes", 0)),
        "upload_bytes": int(profile.get("upload_bytes", 0)),
        "profile": profile,
    }


def _queue_streamed_job(
    cp,
    process_gpu_streaming,
    writer,
    base,
    job,
    device,
    memory_budget_bytes,
    output_dir,
    batch_number,
    estimated_peak_bytes,
    *,
    runtime_oom=False,
    runtime_oom_s=0.0,
    cache_stats=None,
):
    candidate, profile, process_seconds = _run_streamed_job(
        cp, process_gpu_streaming, base, job, device, memory_budget_bytes
    )
    cache_stats = cache_stats or {}
    profile["read_bytes"] = int(
        profile.get("read_bytes", 0) + cache_stats.get("read_bytes", 0)
    )
    profile["upload_bytes"] = int(
        profile.get("upload_bytes", 0) + cache_stats.get("upload_bytes", 0)
    )
    profile.update(
        {
            "runtime_oom_s": float(runtime_oom_s),
            "cache_read_s": float(cache_stats.get("read_s", 0.0)),
            "cache_upload_s": float(cache_stats.get("upload_s", 0.0)),
            "streaming": True,
            "runtime_oom_streaming": bool(runtime_oom),
            "resident_start": None,
            "resident_stop": None,
            "resident_bytes": 0,
            "batch_number": int(batch_number),
            "estimated_peak_bytes": int(estimated_peak_bytes),
        }
    )
    profile.setdefault("read_bytes", 0)
    profile.setdefault("upload_bytes", 0)
    return _queue_candidate_save(
        writer,
        candidate,
        job,
        profile,
        process_seconds,
        device,
        output_dir,
    )


def _save_candidate(candidate, output_dir: str | None, output_filename=None):
    started = perf_counter()
    if output_filename is None:
        filename = candidate.save_h5(out_dir=output_dir)
    else:
        filename = candidate.save_h5(out_dir=output_dir, fnout=output_filename)
    return os.path.abspath(filename), perf_counter() - started


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()


def _clear_exception_tracebacks(error: BaseException) -> None:
    """Drop traceback-held temporary arrays before retrying an OOM."""
    pending = [error]
    visited = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        current.__traceback__ = None


def _run_batch(cp, process_gpu, base, batch, resident, device, max_inflight):
    results = {}
    errors = []
    worker_count = min(max_inflight, len(batch.jobs))
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix=f"gpu-{device}"
    ) as pool:
        futures = {
            pool.submit(
                _run_gpu_job,
                cp,
                process_gpu,
                base,
                job,
                resident.array,
                resident.start,
                device,
            ): job
            for job in batch.jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results[job.index] = (job, *future.result())
            except BaseException as exc:
                errors.append(exc)
                if _is_out_of_memory(exc):
                    _clear_exception_tracebacks(exc)
    if errors:
        non_oom_error = next(
            (error for error in errors if not _is_out_of_memory(error)), None
        )
        if non_oom_error is not None:
            raise non_oom_error
        error_message = str(errors[0])
        for error in errors:
            _clear_exception_tracebacks(error)
        results.clear()
        errors.clear()
        futures.clear()
        del future
        if len(batch.jobs) == 1:
            raise MemoryError(error_message) from None
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        for job in batch.jobs:
            candidate, profile, process_seconds = _run_gpu_job(
                cp, process_gpu, base, job, resident.array, resident.start, device
            )
            profile["oom_serial_retry"] = 1
            results[job.index] = (job, candidate, profile, process_seconds)

    return results


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _preflight_output_paths(jobs, output_dir):
    names = [job.output_filename for job in jobs]
    if not any(name is not None for name in names):
        return
    if any(name is None for name in names):
        raise ValueError("all candidate jobs must set output_filename when any do")
    output_root = Path(output_dir or ".").resolve()
    seen: dict[Path, int] = {}
    for job in jobs:
        path = (output_root / job.output_filename).resolve()
        previous = seen.get(path)
        if previous is not None:
            raise ValueError(
                f"candidate jobs {previous} and {job.index} share output path {path}"
            )
        seen[path] = job.index


def _gpu_worker(
    device,
    job_queue,
    result_queue,
    output_dir,
    memory_budget_bytes,
    safety_fraction,
    max_inflight,
):
    active_index = None
    try:
        import cupy as cp

        from your.candidate import Candidate
        from your.utils.candidate_contract import decoded_sample_dtype
        from your.utils.gpu import estimate_process_gpu_bytes, process_gpu

        with cp.cuda.Device(device):
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            available = choose_memory_budget(free_bytes, safety_fraction)
            budget = (
                available
                if memory_budget_bytes is None
                else min(int(memory_budget_bytes), available)
            )
            if budget < 1:
                raise MemoryError(f"GPU {device} has no usable free memory")
            writer = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"h5-{device}"
            )
            pending_writes = deque()
            max_pending_writes = max(1, 2 * max_inflight)

            def collect_pending(block=False):
                while pending_writes:
                    pending = pending_writes[0]
                    if not block and not pending["future"].done():
                        break
                    pending_writes.popleft()
                    result_queue.put(_complete_candidate_save(pending))
                    block = False

            def queue_write(pending):
                pending_writes.append(pending)
                collect_pending(block=len(pending_writes) >= max_pending_writes)

            try:
                while True:
                    work = job_queue.get()
                    if work is None:
                        break
                    if not isinstance(work, _Work):
                        raise TypeError(
                            "scheduler worker received an invalid work item"
                        )
                    first = work.jobs[0]
                    kwargs = dict(first.candidate_kwargs)
                    kwargs.pop("fp", None)
                    kwargs["device"] = device
                    base = Candidate(fp=list(work.files), **kwargs)
                    try:
                        nchans = int(base.your_header.nchans)
                        input_dtype = decoded_sample_dtype(base)
                        cache = _ResidentCache(
                            cp, base, device, nchans, input_dtype, budget
                        )
                        batches = plan_batches(
                            work.jobs,
                            nchans=nchans,
                            input_dtype=input_dtype,
                            memory_budget_bytes=budget,
                            workspace_estimator=estimate_process_gpu_bytes,
                            max_inflight=max_inflight,
                            initial_resident=cache.interval,
                        )
                        batch_number = 0
                        for batch in batches:
                            active_index = batch.jobs[0].index
                            if batch.streaming:
                                cache.close()
                                cp.get_default_memory_pool().free_all_blocks()
                                cp.get_default_pinned_memory_pool().free_all_blocks()
                                from your.utils.gpu_streaming import (
                                    process_gpu_streaming,
                                )

                                job = batch.jobs[0]
                                queue_write(
                                    _queue_streamed_job(
                                        cp,
                                        process_gpu_streaming,
                                        writer,
                                        base,
                                        job,
                                        device,
                                        budget,
                                        output_dir,
                                        batch_number,
                                        batch.estimated_peak_bytes,
                                    )
                                )
                                collect_pending()
                            else:
                                if batch.drop_cache_before:
                                    cache.close()
                                    cp.get_default_memory_pool().free_all_blocks()
                                    pinned_pool = cp.get_default_pinned_memory_pool()
                                    pinned_pool.free_all_blocks()
                                cache_stats = {
                                    "read_bytes": 0,
                                    "upload_bytes": 0,
                                    "read_s": 0.0,
                                    "upload_s": 0.0,
                                }
                                attempt_start = perf_counter()
                                runtime_oom = False
                                try:
                                    cache_stats = cache.ensure(
                                        batch.read_start, batch.read_stop
                                    )
                                    results = _run_batch(
                                        cp,
                                        process_gpu,
                                        base,
                                        batch,
                                        cache,
                                        device,
                                        max_inflight,
                                    )
                                except BaseException as exc:
                                    if not _is_out_of_memory(exc):
                                        raise
                                    runtime_oom_s = perf_counter() - attempt_start
                                    _clear_exception_tracebacks(exc)
                                    runtime_oom = True
                                if runtime_oom:
                                    cache.close()
                                    cp.get_default_memory_pool().free_all_blocks()
                                    pinned_pool = cp.get_default_pinned_memory_pool()
                                    pinned_pool.free_all_blocks()
                                    from your.utils.gpu_streaming import (
                                        process_gpu_streaming,
                                    )

                                    for job in batch.jobs:
                                        active_index = job.index
                                        queue_write(
                                            _queue_streamed_job(
                                                cp,
                                                process_gpu_streaming,
                                                writer,
                                                base,
                                                job,
                                                device,
                                                budget,
                                                output_dir,
                                                batch_number,
                                                batch.estimated_peak_bytes,
                                                runtime_oom=True,
                                                runtime_oom_s=runtime_oom_s,
                                                cache_stats=(
                                                    cache_stats
                                                    if job.index == batch.jobs[0].index
                                                    else None
                                                ),
                                            )
                                        )
                                        collect_pending()
                                    batch_number += 1
                                    continue
                                for job in batch.jobs:
                                    active_index = job.index
                                    owner = job.index == batch.jobs[0].index
                                    _, candidate, profile, process_seconds = results[
                                        job.index
                                    ]
                                    details = {
                                        "read_bytes": int(
                                            cache_stats["read_bytes"] if owner else 0
                                        ),
                                        "upload_bytes": int(
                                            cache_stats["upload_bytes"] if owner else 0
                                        ),
                                        "batch_read_bytes": int(
                                            cache_stats["read_bytes"]
                                        ),
                                        "batch_upload_bytes": int(
                                            cache_stats["upload_bytes"]
                                        ),
                                        "cache_read_s": float(
                                            cache_stats["read_s"] if owner else 0.0
                                        ),
                                        "cache_upload_s": float(
                                            cache_stats["upload_s"] if owner else 0.0
                                        ),
                                        "resident_start": int(cache.start),
                                        "resident_stop": int(
                                            cache.start + cache.array.shape[1]
                                        ),
                                        "resident_bytes": int(cache.array.nbytes),
                                        "estimated_peak_bytes": int(
                                            batch.estimated_peak_bytes
                                        ),
                                        "workspace_bytes": int(batch.workspace_bytes),
                                        "batch_number": int(batch_number),
                                        "streaming": False,
                                        "oom_serial_retry": int(
                                            profile.get("oom_serial_retry", 0)
                                        ),
                                    }
                                    pending = _queue_candidate_save(
                                        writer,
                                        candidate,
                                        job,
                                        profile,
                                        process_seconds,
                                        device,
                                        output_dir,
                                        details=details,
                                    )
                                    queue_write(pending)
                                collect_pending()
                            batch_number += 1
                        cache.close()
                    finally:
                        base.close()
                while pending_writes:
                    collect_pending(block=True)
                writer.shutdown(wait=True)
            finally:
                writer.shutdown(wait=True, cancel_futures=True)
        result_queue.put({"type": "done", "device": device})
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "device": device,
                "index": active_index,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def run_jobs(
    jobs: Iterable[CandidateJob],
    *,
    gpu_ids: Iterable[int],
    output_dir: str | os.PathLike[str] | None = None,
    memory_budget_bytes: int | None = None,
    safety_fraction: float = 0.8,
    max_inflight: int = 2,
    output_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run candidate jobs in one spawned process per requested GPU.

    Output HDF5 files are written by a bounded, single-thread writer per GPU process.
    ``output_callback`` runs in the parent process and receives each completed report.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    gpu_ids = tuple(int(gpu_id) for gpu_id in gpu_ids)
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError("gpu_ids must contain distinct non-negative device IDs")
    if max_inflight < 1:
        raise ValueError("max_inflight must be at least one")
    if memory_budget_bytes is not None and memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive")
    choose_memory_budget(1, safety_fraction)
    _preflight_output_paths(jobs, output_dir)
    if output_dir is not None:
        output_dir = os.fspath(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    groups = group_jobs_by_files(jobs)
    assigned = _assign_work(groups, gpu_ids)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    workers = {}
    for device in gpu_ids:
        job_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_gpu_worker,
            args=(
                device,
                job_queue,
                result_queue,
                output_dir,
                memory_budget_bytes,
                safety_fraction,
                max_inflight,
            ),
            name=f"candidate-gpu-{device}",
        )
        process.start()
        workers[device] = {
            "process": process,
            "queue": job_queue,
            "work": iter(assigned[device]),
            "pending": None,
            "sent_stop": False,
            "done": False,
        }

    reports: dict[int, dict[str, Any]] = {}
    completed_successfully = False
    try:
        completed_workers = 0
        while completed_workers < len(workers):
            for state in workers.values():
                if state["sent_stop"]:
                    continue
                if state["pending"] is None:
                    try:
                        state["pending"] = next(state["work"])
                    except StopIteration:
                        state["pending"] = None
                        try:
                            state["queue"].put_nowait(None)
                        except queue.Full:
                            continue
                        state["sent_stop"] = True
                        continue
                try:
                    state["queue"].put_nowait(state["pending"])
                except queue.Full:
                    pass
                else:
                    state["pending"] = None

            try:
                message = result_queue.get(timeout=0.2)
            except queue.Empty:
                for device, state in workers.items():
                    process = state["process"]
                    if process.exitcode is not None and not state["done"]:
                        raise RuntimeError(
                            f"GPU worker {device} exited with status "
                            f"{process.exitcode} before reporting completion"
                        )
                continue

            if message["type"] == "result":
                index = int(message["index"])
                if index in reports:
                    raise RuntimeError(
                        f"GPU workers returned duplicate candidate index {index}"
                    )
                reports[index] = message
                if output_callback is not None:
                    output_callback(message)
            elif message["type"] == "error":
                location = (
                    f"candidate {message['index']}"
                    if message["index"] is not None
                    else "worker startup"
                )
                raise RuntimeError(
                    f"GPU {message['device']} failed during {location}: "
                    f"{message['error']}\n"
                    f"{message['traceback']}"
                )
            elif message["type"] == "done":
                state = workers[message["device"]]
                if not state["done"]:
                    state["done"] = True
                    completed_workers += 1

        for device, state in workers.items():
            state["process"].join(timeout=5)
            if state["process"].exitcode != 0:
                raise RuntimeError(
                    f"GPU worker {device} exited with status "
                    f"{state['process'].exitcode}"
                )
        if len(reports) != len(jobs):
            missing = sorted({job.index for job in jobs} - reports.keys())
            raise RuntimeError(
                f"GPU workers completed without reports for candidate indices {missing}"
            )
        completed_successfully = True
        return [reports[index] for index in sorted(reports)]
    finally:
        for state in workers.values():
            process = state["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if not completed_successfully:
                # A failed worker can leave this feeder writing to an unread pipe.
                state["queue"].cancel_join_thread()
            state["queue"].close()
        result_queue.close()
