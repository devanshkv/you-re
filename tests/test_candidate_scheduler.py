import gc
import pickle
import subprocess
import sys
import textwrap
import types
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from ctypes import memmove
from dataclasses import dataclass
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest

import your.utils.candidate_scheduler as scheduler
from your.utils.candidate_scheduler import (
    CandidateBatch,
    CandidateJob,
    CompactCandidatePlan,
    _assign_work,
    _clear_exception_tracebacks,
    _complete_candidate_save,
    _materialize_plan,
    _missing_intervals,
    _preflight_output_paths,
    _queue_candidate_save,
    _ResidentCache,
    _run_batch,
    _transition_peak,
    choose_memory_budget,
    group_jobs_by_files,
    plan_batches,
)


@dataclass
class Plan:
    required_start: int
    required_stop: int
    read_start: int
    read_stop: int
    workspace: int = 0


def job(index, required, readable, workspace=0, files=("obs.fil",)):
    return CandidateJob(
        files=files,
        candidate_kwargs={"dm": float(index)},
        plan=Plan(*required, *readable, workspace),
        index=index,
    )


def estimate(plan, _dtype):
    return plan.workspace


def _install_fake_cupy(
    monkeypatch,
    driver_free_bytes=1000,
    pool_free_bytes=0,
    pool_limit=0,
    pool_used_bytes=0,
):
    class Pool:
        free_calls = 0
        get_limit_calls = 0
        used_calls = 0
        default_pool_calls = 0

        def free_bytes(self):
            self.free_calls += 1
            return pool_free_bytes

        def get_limit(self):
            self.get_limit_calls += 1
            return pool_limit

        def used_bytes(self):
            self.used_calls += 1
            if not pool_limit:
                raise AssertionError("unlimited pool usage must not be queried")
            return pool_used_bytes

    pool = Pool()

    def get_default_memory_pool():
        pool.default_pool_calls += 1
        return pool

    cupy = types.ModuleType("cupy")
    cupy.cuda = SimpleNamespace(
        Device=lambda _device: nullcontext(),
        runtime=SimpleNamespace(
            memGetInfo=lambda: (driver_free_bytes, driver_free_bytes * 2)
        ),
    )
    cupy.get_default_memory_pool = get_default_memory_pool
    cupy.get_default_pinned_memory_pool = lambda: pool
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    return pool


def test_jobs_group_by_exact_files_and_required_interval():
    jobs = [
        job(3, (20, 30), (20, 25), files=("b.fits",)),
        job(2, (5, 10), (5, 8)),
        job(1, (0, 5), (0, 3), files=("b.fits",)),
    ]

    groups = group_jobs_by_files(jobs)

    assert [files for files, _ in groups] == [("b.fits",), ("obs.fil",)]
    assert [[candidate.index for candidate in grouped] for _, grouped in groups] == [
        [1, 3],
        [2],
    ]


def test_assignment_balances_singleton_observations_across_requested_gpu_ids():
    jobs = [
        job(0, (0, 1), (0, 1), files=("a.fil",)),
        job(1, (0, 1), (0, 1), files=("b.fil",)),
        *[
            job(index, (index, index + 1), (index, index + 1), files=("c.fil",))
            for index in range(2, 8)
        ],
    ]

    assigned = _assign_work(group_jobs_by_files(jobs), (2, 3))

    assert [assigned[gpu_id][0].jobs[0].index for gpu_id in (2, 3)] == [0, 1]
    chunks = {}
    for gpu_id in (2, 3):
        work = next(item for item in assigned[gpu_id] if item.files == ("c.fil",))
        chunks[gpu_id] = [candidate.index for candidate in work.jobs]
    assert chunks == {2: [2, 3, 4], 3: [5, 6, 7]}
    assert sorted(
        candidate.index
        for work_items in assigned.values()
        for work in work_items
        for candidate in work.jobs
    ) == list(range(8))


def test_batch_budget_counts_resident_and_each_inflight_workspace():
    jobs = [
        job(0, (0, 10), (0, 10), workspace=5),
        job(1, (1, 11), (5, 15), workspace=7),
    ]

    batches = plan_batches(
        jobs,
        nchans=2,
        input_dtype=np.uint8,
        memory_budget_bytes=44,
        workspace_estimator=estimate,
        max_inflight=2,
    )

    assert len(batches) == 1
    assert [candidate.index for candidate in batches[0].jobs] == [0, 1]
    assert (batches[0].read_start, batches[0].read_stop) == (0, 15)
    assert batches[0].workspace_bytes == 12
    assert batches[0].estimated_peak_bytes == 44


def test_workspace_pressure_splits_a_batch():
    jobs = [
        job(0, (0, 10), (0, 10), workspace=8),
        job(1, (1, 11), (5, 10), workspace=8),
    ]

    batches = plan_batches(
        jobs,
        nchans=2,
        input_dtype=np.uint8,
        memory_budget_bytes=28,
        workspace_estimator=estimate,
        max_inflight=2,
    )

    assert [[candidate.index for candidate in batch.jobs] for batch in batches] == [
        [0],
        [1],
    ]
    assert [batch.estimated_peak_bytes for batch in batches] == [28, 28]


def test_overlap_resize_drops_cache_when_reload_and_workspace_fit():
    jobs = [
        job(0, (0, 10), (0, 10)),
        job(1, (1, 11), (5, 15)),
    ]

    batches = plan_batches(
        jobs,
        nchans=1,
        input_dtype=np.uint8,
        memory_budget_bytes=19,
        workspace_estimator=estimate,
        max_inflight=1,
    )

    assert [batch.estimated_peak_bytes for batch in batches] == [19, 19]
    assert [batch.streaming for batch in batches] == [False, False]
    assert [batch.drop_cache_before for batch in batches] == [False, True]
    assert [candidate.index for batch in batches for candidate in batch.jobs] == [0, 1]


def test_overlap_still_streams_when_reloaded_interval_plus_workspace_exceeds_budget():
    jobs = [
        job(0, (0, 10), (0, 10)),
        job(1, (1, 11), (5, 15), workspace=10),
    ]

    batches = plan_batches(
        jobs,
        nchans=1,
        input_dtype=np.uint8,
        memory_budget_bytes=19,
        workspace_estimator=estimate,
        max_inflight=1,
    )

    assert [batch.estimated_peak_bytes for batch in batches] == [19, 21]
    assert [batch.streaming for batch in batches] == [False, True]
    assert not batches[1].drop_cache_before


def test_disjoint_cache_move_releases_old_resident_before_allocation():
    jobs = [
        job(0, (0, 10), (0, 10)),
        job(1, (10, 20), (10, 20)),
    ]

    batches = plan_batches(
        jobs,
        nchans=1,
        input_dtype=np.uint8,
        memory_budget_bytes=11,
        workspace_estimator=estimate,
        max_inflight=1,
    )

    assert len(batches) == 2
    assert not any(batch.streaming for batch in batches)
    assert [batch.estimated_peak_bytes for batch in batches] == [11, 11]


def test_upload_staging_is_counted_and_shrinks_to_the_available_budget():
    assert _transition_peak(None, (0, 100), 4, 2, 0, 850) == (848, (0, 100), False)
    assert _transition_peak(None, (0, 100), 4, 2, 0, 807) == (808, (0, 100), False)
    assert _transition_peak((0, 100), (20, 80), 4, 2, 0, 800) == (800, (0, 100), False)
    large = 128 * 1024 * 1024
    assert _transition_peak(None, (0, large), 1, 1, 0, 4 * large)[0] == (
        large + 64 * 1024 * 1024
    )


def test_staging_minimum_sends_a_full_budget_resident_to_streaming():
    batches = plan_batches(
        [job(0, (0, 10), (0, 10))],
        nchans=2,
        input_dtype=np.uint16,
        memory_budget_bytes=40,
        workspace_estimator=estimate,
    )
    assert batches[0].streaming
    assert batches[0].estimated_peak_bytes == 44


def test_choose_memory_budget_validates_fraction():
    assert choose_memory_budget(101, 0.75) == 75
    with pytest.raises(ValueError, match="safety_fraction"):
        choose_memory_budget(101, 0)


def test_zero_readable_samples_still_form_a_valid_batch():
    batches = plan_batches(
        [job(4, (0, 5), (5, 5), workspace=3)],
        nchans=4,
        input_dtype=np.uint8,
        memory_budget_bytes=3,
        workspace_estimator=estimate,
    )

    assert len(batches) == 1
    assert (batches[0].read_start, batches[0].read_stop) == (5, 5)
    assert not batches[0].streaming
    assert batches[0].estimated_peak_bytes == 3


def test_initial_resident_uploads_the_interval_once_and_growth_only_uploads_edges():
    assert _missing_intervals(None, (3, 9)) == [(3, 9)]
    assert _missing_intervals((3, 9), (3, 9)) == []
    assert _missing_intervals((5, 10), (2, 13)) == [(2, 5), (10, 13)]
    assert _missing_intervals((0, 4), (8, 8)) == []


def test_compact_plan_keeps_scalars_and_materializes_on_demand(monkeypatch):
    delay_table = np.zeros((256, 4096), dtype=np.int64)

    class FullPlan:
        time_size = 4
        frequency_size = 3
        dmsteps = 256
        output_dtype = np.dtype(np.float32)
        native_count = 8
        required_start = 0
        required_stop = 10
        read_start = 0
        read_stop = 10
        nspectra = 100
        native_tsamp = 0.001
        nchans = 256
        pad_before = 0
        pad_after = 0
        delays = delay_table
        dm_values = np.zeros(256)

    compact = CompactCandidatePlan.from_plan(FullPlan())
    assert all(value is not delay_table for value in vars(compact).values())
    assert not any(isinstance(value, np.ndarray) for value in vars(compact).values())
    assert len(pickle.dumps(compact)) < 4096

    expected = object()
    calls = []

    def make_plan(candidate, *args):
        calls.append((candidate, args))
        return expected

    monkeypatch.setattr("your.utils.candidate_pipeline.make_plan", make_plan)
    candidate = object()
    assert _materialize_plan(candidate, compact) is expected
    assert calls == [(candidate, (4, 3, 256, np.dtype(np.float32)))]


def test_batch_planning_materializes_only_the_job_that_starts_running(monkeypatch):
    delay_table = np.empty((256, 4096), dtype=np.int64)

    class FullPlan:
        time_size = 4
        frequency_size = 3
        dmsteps = 256
        output_dtype = np.dtype(np.float32)
        native_count = 8
        required_start = 0
        required_stop = 1
        read_start = 0
        read_stop = 1
        nspectra = 100
        native_tsamp = 0.001
        nchans = 4096
        pad_before = 0
        pad_after = 0
        delays = delay_table
        dm_values = np.zeros(256)

    compact = CompactCandidatePlan.from_plan(FullPlan())
    materialize_calls = []
    full_plan = object()

    def make_plan(candidate, *args):
        materialize_calls.append(candidate)
        return full_plan

    monkeypatch.setattr("your.utils.candidate_pipeline.make_plan", make_plan)
    jobs = [CandidateJob(("obs.fil",), {}, compact, index) for index in range(100)]
    batches = plan_batches(
        jobs,
        nchans=4096,
        input_dtype=np.uint8,
        memory_budget_bytes=100_000,
        workspace_estimator=lambda _plan, _dtype: 1,
        max_inflight=1,
    )

    assert len(batches) == 100
    assert materialize_calls == []
    candidate = object()
    assert _materialize_plan(candidate, batches[0].jobs[0].plan) is full_plan
    assert materialize_calls == [candidate]


def test_explicit_output_paths_are_preflighted_before_workers_start(tmp_path):
    first = job(0, (0, 1), (0, 1))
    second = CandidateJob(first.files, first.candidate_kwargs, first.plan, 1, "same.h5")
    first = CandidateJob(first.files, first.candidate_kwargs, first.plan, 0, "same.h5")

    with pytest.raises(ValueError, match="share output path"):
        _preflight_output_paths([first, second], tmp_path)


@pytest.fixture
def fake_cupy(monkeypatch):
    def empty(shape, dtype):
        values = np.empty(shape, dtype=dtype)
        return SimpleNamespace(
            values=values,
            shape=values.shape,
            strides=values.strides,
            dtype=values.dtype,
            nbytes=values.nbytes,
            data=SimpleNamespace(ptr=values.ctypes.data),
        )

    def copy_2d(dst, dst_pitch, src, src_pitch, width, height, *_args):
        for row in range(height):
            memmove(dst + row * dst_pitch, src + row * src_pitch, width)

    def corner_turn(_cp, source, destination, nsamples, time_offset, _stream):
        destination.values[:, time_offset : time_offset + nsamples] = source.values[
            :nsamples
        ].T

    monkeypatch.setattr("your.utils.gpu._corner_turn", corner_turn)
    return SimpleNamespace(
        empty=empty,
        cuda=SimpleNamespace(
            Device=lambda _device: nullcontext(),
            using_allocator=lambda _allocator: nullcontext(),
            Stream=lambda non_blocking: SimpleNamespace(
                ptr=0, synchronize=lambda: None
            ),
            Event=lambda: SimpleNamespace(
                record=lambda _stream: None, synchronize=lambda: None
            ),
            alloc_pinned_memory=bytearray,
            get_elapsed_time=lambda *_args: 0.0,
            runtime=SimpleNamespace(
                memcpy2DAsync=copy_2d,
                memcpyAsync=lambda dst, src, size, *_: memmove(dst, src, size),
                memcpyHostToDevice=1,
                memcpyDeviceToDevice=2,
            ),
        ),
    )


@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
@pytest.mark.parametrize("retry_succeeds", [True, False])
def test_resident_cache_retries_staging_oom_after_freeing_unused_pool_blocks(
    fake_cupy, error_type, retry_succeeds
):
    source = np.arange(13 * 7, dtype=np.uint16).reshape(13, 7)
    empty = fake_cupy.empty
    allocations = []
    resident = []
    cleared = []

    def allocate(shape, dtype):
        allocations.append(shape)
        if len(allocations) == 2 or (len(allocations) == 3 and not retry_succeeds):
            raise error_type("cudaErrorMemoryAllocation: out of memory")
        result = empty(shape, dtype)
        if len(allocations) == 1:
            resident.append(weakref.ref(result.values))
        return result

    def clear_unused():
        assert resident[0]() is not None
        cleared.append(True)

    fake_cupy.empty = allocate
    fake_cupy.get_default_memory_pool = lambda: SimpleNamespace(
        free_all_blocks=clear_unused
    )
    cache = _ResidentCache(
        fake_cupy, SimpleNamespace(get_data=lambda *_args: source), 0, 7, source.dtype
    )
    if retry_succeeds:
        stats = cache.ensure(0, len(source))
        np.testing.assert_array_equal(cache.array.values, source.T)
        assert stats["read_bytes"] == stats["upload_bytes"] == source.nbytes
    else:
        with pytest.raises(error_type, match="out of memory"):
            cache.ensure(0, len(source))
    assert cleared == [True]
    assert allocations == [(7, 13), (13, 7), (13, 7)]


def test_resident_cache_does_not_retry_non_oom_staging_errors(fake_cupy):
    source = np.zeros((3, 4), dtype=np.uint8)
    empty = fake_cupy.empty
    allocations = []
    cleared = []

    def allocate(shape, dtype):
        allocations.append(shape)
        if len(allocations) == 2:
            raise RuntimeError("invalid CUDA device")
        return empty(shape, dtype)

    fake_cupy.empty = allocate
    fake_cupy.get_default_memory_pool = lambda: SimpleNamespace(
        free_all_blocks=lambda: cleared.append(True)
    )
    cache = _ResidentCache(
        fake_cupy, SimpleNamespace(get_data=lambda *_args: source), 0, 4, source.dtype
    )
    with pytest.raises(RuntimeError, match="invalid CUDA device"):
        cache.ensure(0, len(source))
    assert not cleared
    assert allocations == [(4, 3), (3, 4)]


def test_resident_cache_rejects_dtype_mismatch_without_narrowing(fake_cupy):
    class Reader:
        @staticmethod
        def get_data(_start, _count):
            return np.ones((1, 1), dtype=np.float32)

    cache = _ResidentCache(fake_cupy, Reader(), 0, 1, np.dtype(np.int32))
    with pytest.raises(TypeError, match="decoded sample dtype"):
        cache.ensure(0, 1)


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.int32, np.float16, np.float32, np.float64],
)
@pytest.mark.parametrize("nchans", [3, 17])
@pytest.mark.parametrize("strided", [False, True])
def test_resident_cache_copies_exact_transpose_with_partial_block(
    fake_cupy, dtype, nchans, strided
):
    start, count = 11, 385
    values = np.arange((start + count) * 2 * nchans * 2).reshape(
        (start + count) * 2, nchans * 2
    )
    source = (values - 4096).astype(dtype)[::2, ::-2]
    if not strided:
        source = source.copy()
    reader = SimpleNamespace(get_data=lambda left, size: source[left : left + size])
    cache = _ResidentCache(fake_cupy, reader, 0, nchans, np.dtype(dtype))

    stats = cache.ensure(start, start + count)

    expected = source[start : start + count].T
    np.testing.assert_array_equal(cache.array.values, expected)
    assert cache.array.values.dtype == expected.dtype
    assert cache.interval == (start, start + count)
    assert stats["read_bytes"] == stats["upload_bytes"] == expected.nbytes


def test_resident_cache_growth_reuses_overlap_and_bounds_staging(fake_cupy):
    source = np.arange(60 * 37, dtype=np.uint16).reshape(60, 37)
    reads = []

    def get_data(start, count):
        reads.append((start, count))
        return source[start : start + count]

    cache = _ResidentCache(
        fake_cupy,
        SimpleNamespace(get_data=get_data),
        0,
        37,
        source.dtype,
        memory_budget_bytes=74 * 49,
    )
    initial = cache.ensure(10, 30)
    growth = cache.ensure(6, 34)
    contained = cache.ensure(8, 32)

    np.testing.assert_array_equal(cache.array.values, source[6:34].T)
    assert reads == [
        (10, 20),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (30, 1),
        (31, 1),
        (32, 1),
        (33, 1),
    ]
    assert initial["upload_bytes"] == 20 * 74
    assert growth["read_bytes"] == growth["upload_bytes"] == 8 * 74
    assert contained == dict(read_bytes=0, upload_bytes=0, read_s=0.0, upload_s=0.0)


def test_resident_cache_rejects_a_budget_without_one_staging_row(fake_cupy):
    source = np.zeros((10, 4), dtype=np.uint16)
    cache = _ResidentCache(
        fake_cupy,
        SimpleNamespace(get_data=lambda *_args: source),
        0,
        4,
        source.dtype,
        memory_budget_bytes=source.nbytes,
    )
    with pytest.raises(MemoryError, match="upload staging"):
        cache.ensure(0, len(source))


def test_resident_cache_keeps_pinned_upload_alive_on_kernel_failure(
    fake_cupy, monkeypatch
):
    class Pinned(bytearray):
        pass

    pinned = []
    synchronized = []

    def allocate(size):
        buffer = Pinned(size)
        pinned.append(weakref.ref(buffer))
        return buffer

    def synchronize():
        assert pinned[0]() is not None
        synchronized.append(True)

    def fail_corner_turn(*_args):
        raise RuntimeError("corner turn launch failed")

    fake_cupy.cuda.alloc_pinned_memory = allocate
    fake_cupy.cuda.Stream = lambda **_kwargs: SimpleNamespace(
        ptr=0, synchronize=synchronize
    )
    monkeypatch.setattr("your.utils.gpu._corner_turn", fail_corner_turn)
    source = np.zeros((3, 4), dtype=np.uint8)
    cache = _ResidentCache(
        fake_cupy, SimpleNamespace(get_data=lambda *_args: source), 0, 4, source.dtype
    )
    with pytest.raises(RuntimeError, match="corner turn launch failed"):
        cache.ensure(0, len(source))
    assert synchronized == [True]


def test_hdf5_save_is_queued_without_blocking_gpu_worker():
    started = Event()
    release = Event()

    class DelayedCandidate:
        id = "candidate"

        def save_h5(self, out_dir=None, fnout=None):
            started.set()
            release.wait(timeout=1)
            return fnout

    writer = ThreadPoolExecutor(max_workers=1)
    try:
        pending = _queue_candidate_save(
            writer,
            DelayedCandidate(),
            job(0, (0, 1), (0, 1)),
            {},
            0.25,
            3,
            None,
        )
        assert started.wait(timeout=1)
        assert not pending["future"].done()
        release.set()
        report = _complete_candidate_save(pending)
    finally:
        release.set()
        writer.shutdown(wait=True)

    assert report["device"] == 3
    assert report["index"] == 0
    assert report["profile"]["gpu_processing_s"] == 0.25
    assert report["profile"]["total_s"] >= 0.25


def test_queued_save_uses_preflighted_filename():
    saved = []

    class Candidate:
        def save_h5(self, out_dir=None, fnout=None):
            saved.append((out_dir, fnout))
            return fnout

    writer = ThreadPoolExecutor(max_workers=1)
    try:
        pending = _queue_candidate_save(
            writer,
            Candidate(),
            CandidateJob(("obs.fil",), {}, Plan(0, 1, 0, 1), 4, "stable-candidate.h5"),
            {},
            0.01,
            0,
            "output",
        )
        report = _complete_candidate_save(pending)
    finally:
        writer.shutdown(wait=True)

    assert saved == [("output", "stable-candidate.h5")]
    assert report["output_file"].endswith("stable-candidate.h5")


def test_run_jobs_in_process_returns_sorted_results(monkeypatch):
    def inline_worker(device, job_queue, result_queue, *_args):
        work = []
        while (item := job_queue.get()) is not None:
            work.append(item)
        assert len(work) == 1
        for index in (1, 0):
            result_queue.put({"type": "result", "device": device, "index": index})
        result_queue.put({"type": "done", "device": device})

    monkeypatch.setattr(scheduler, "_gpu_worker", inline_worker)
    reports = scheduler.run_jobs(
        [job(1, (0, 1), (0, 1)), job(0, (1, 2), (1, 2))],
        gpu_ids=[3],
        in_process=True,
    )

    assert [report["index"] for report in reports] == [0, 1]


def test_run_jobs_default_names_preserve_same_candidate_plans(tmp_path, monkeypatch):
    from pathlib import Path

    jobs = [
        CandidateJob(("obs.fil",), {"dm": 1.0}, Plan(0, 1, 0, 1), 0),
        CandidateJob(("obs.fil",), {"dm": 1.0}, Plan(1, 2, 1, 2), 1),
    ]

    class Candidate:
        id = "same-candidate"

        def __init__(self, plan):
            self.plan = plan

        def save_h5(self, out_dir=None, fnout=None):
            path = Path(out_dir) / fnout
            path.write_text(f"{self.plan.required_start}:{self.plan.required_stop}")
            return str(path)

    def fake_worker(device, job_queue, result_queue, output_dir, *_args):
        writer = ThreadPoolExecutor(max_workers=1)
        try:
            while (work := job_queue.get()) is not None:
                for candidate_job in work.jobs:
                    pending = _queue_candidate_save(
                        writer,
                        Candidate(candidate_job.plan),
                        candidate_job,
                        {},
                        0.01,
                        device,
                        output_dir,
                    )
                    result_queue.put(_complete_candidate_save(pending))
            result_queue.put({"type": "done", "device": device})
        finally:
            writer.shutdown(wait=True)

    monkeypatch.setattr(scheduler, "_gpu_worker", fake_worker)
    reports = scheduler.run_jobs(
        jobs, gpu_ids=[0], output_dir=tmp_path, in_process=True
    )

    paths = [Path(report["output_file"]) for report in reports]
    assert [path.name for path in paths] == [
        "same-candidate_0.h5",
        "same-candidate_1.h5",
    ]
    assert [path.read_text() for path in paths] == ["0:1", "1:2"]


def test_run_jobs_in_process_reports_worker_errors(monkeypatch):
    def inline_worker(device, _job_queue, result_queue, *_args):
        result_queue.put(
            {
                "type": "error",
                "device": device,
                "index": 0,
                "error": "synthetic worker failure",
                "traceback": "worker traceback",
            }
        )

    monkeypatch.setattr(scheduler, "_gpu_worker", inline_worker)
    with pytest.raises(RuntimeError, match="GPU 2 failed during candidate 0"):
        scheduler.run_jobs([job(0, (0, 1), (0, 1))], gpu_ids=[2], in_process=True)


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        (
            [
                {"type": "result", "index": 0},
                {"type": "result", "index": 0},
                {"type": "done"},
            ],
            "duplicate candidate index 0",
        ),
        (
            [{"type": "result", "index": 0}, {"type": "done"}],
            r"without reports for candidate indices \[1\]",
        ),
        (
            [
                {"type": "result", "index": 0},
                {"type": "result", "index": 2},
                {"type": "done"},
            ],
            r"without reports for candidate indices \[1\]",
        ),
        ([{"type": "result", "index": 0}], "before reporting completion"),
    ],
    ids=[
        "duplicate-result",
        "missing-result",
        "wrong-result-indices",
        "missing-completion",
    ],
)
def test_run_jobs_in_process_guards_completion(monkeypatch, messages, error):
    def inline_worker(device, job_queue, result_queue, *_args):
        while job_queue.get() is not None:
            pass
        for message in messages:
            result_queue.put({"device": device, **message})

    monkeypatch.setattr(scheduler, "_gpu_worker", inline_worker)
    with pytest.raises(RuntimeError, match=error):
        scheduler.run_jobs(
            [job(0, (0, 1), (0, 1)), job(1, (1, 2), (1, 2))],
            gpu_ids=[0],
            in_process=True,
        )


def test_spawned_run_jobs_rejects_equal_count_wrong_candidate_indices(tmp_path):
    from pathlib import Path

    repo = Path(__file__).parents[1]
    script = tmp_path / "wrong_result_indices.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            import sys
            sys.path.insert(0, {str(repo)!r})

            import your.utils.candidate_scheduler as scheduler
            from your.utils.candidate_scheduler import CandidateJob

            class Plan:
                required_start = 0
                required_stop = 1

            def wrong_indices_worker(device, job_queue, result_queue, *_args):
                while job_queue.get() is not None:
                    pass
                result_queue.put({{"type": "result", "device": device, "index": 0}})
                result_queue.put({{"type": "result", "device": device, "index": 2}})
                result_queue.put({{"type": "done", "device": device}})

            if __name__ == "__main__":
                scheduler._gpu_worker = wrong_indices_worker
                jobs = [
                    CandidateJob(("obs.fil",), {{}}, Plan(), index)
                    for index in (0, 1)
                ]
                try:
                    scheduler.run_jobs(jobs, gpu_ids=[0])
                except RuntimeError as error:
                    print(f"EXPECTED {{error}}", flush=True)
                else:
                    print("UNEXPECTED SUCCESS", flush=True)
            """
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        "EXPECTED GPU workers completed without reports for candidate indices [1]"
        in completed.stdout
    )


def test_run_jobs_in_process_requires_one_gpu_and_no_callback():
    candidate_job = job(0, (0, 1), (0, 1))
    with pytest.raises(ValueError, match="exactly one GPU ID"):
        scheduler.run_jobs([candidate_job], gpu_ids=[0, 1], in_process=True)
    with pytest.raises(ValueError, match="output_callback"):
        scheduler.run_jobs(
            [candidate_job],
            gpu_ids=[0],
            output_callback=lambda _report: None,
            in_process=True,
        )


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_run_jobs_in_process_propagates_interrupt_and_shuts_down_writer(
    monkeypatch, interrupt_type
):
    import importlib

    _install_fake_cupy(monkeypatch)
    candidate_module = importlib.import_module("your.candidate")
    contract_module = importlib.import_module("your.utils.candidate_contract")
    failure = interrupt_type("worker interrupted")
    candidate_job = job(0, (0, 1), (0, 1))
    closed_candidates = []

    class Candidate:
        def __init__(self, *_args, **_kwargs):
            self.your_header = SimpleNamespace(nchans=1)

        def close(self):
            closed_candidates.append(self)

    class Resident:
        interval = None

        def __init__(self, *_args):
            pass

        def ensure(self, *_args):
            return {
                "read_bytes": 0,
                "upload_bytes": 0,
                "read_s": 0,
                "upload_s": 0,
            }

        def close(self):
            pass

    monkeypatch.setattr(candidate_module, "Candidate", Candidate)
    monkeypatch.setattr(
        contract_module,
        "decoded_sample_dtype",
        lambda _candidate: np.dtype(np.uint8),
    )
    monkeypatch.setattr(
        scheduler,
        "_ResidentCache",
        Resident,
    )
    monkeypatch.setattr(
        scheduler,
        "plan_batches",
        lambda *_args, **_kwargs: [CandidateBatch((candidate_job,), 0, 1, 0, 0)],
    )

    def interrupt_batch(*_args):
        raise failure

    monkeypatch.setattr(scheduler, "_run_batch", interrupt_batch)
    real_executor = scheduler.ThreadPoolExecutor
    shutdown_calls = []

    class TrackingExecutor(real_executor):
        def shutdown(self, *args, **kwargs):
            shutdown_calls.append(kwargs.copy())
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(scheduler, "ThreadPoolExecutor", TrackingExecutor)
    with pytest.raises(interrupt_type) as caught:
        scheduler.run_jobs([candidate_job], gpu_ids=[0], in_process=True)

    assert caught.value is failure
    assert len(closed_candidates) == 1
    assert shutdown_calls == [{"wait": True, "cancel_futures": True}]


@pytest.mark.parametrize(
    (
        "pool_free_bytes",
        "safety_fraction",
        "pool_limit",
        "pool_used_bytes",
        "explicit_budget",
        "expected",
    ),
    [
        (200, 0.5, 0, 0, None, 600),
        (200, 1.0, 0, 0, None, 1200),
        (200, 0.5, 500, 0, None, 500),
        (200, 0.5, 800, 300, None, 500),
        (200, 0.5, 800, 300, 250, 250),
        (200, 0.5, 500, 500, None, None),
    ],
    ids=[
        "unlimited-cached-pool",
        "unlimited-all-free",
        "smaller-pool-limit",
        "active-pool-usage",
        "explicit-cap",
        "exhausted-pool-limit",
    ],
)
def test_worker_budget_respects_pool_limit_and_active_bytes(
    monkeypatch,
    pool_free_bytes,
    safety_fraction,
    pool_limit,
    pool_used_bytes,
    explicit_budget,
    expected,
):
    import importlib

    pool = _install_fake_cupy(
        monkeypatch,
        pool_free_bytes=pool_free_bytes,
        pool_limit=pool_limit,
        pool_used_bytes=pool_used_bytes,
    )
    candidate_module = importlib.import_module("your.candidate")
    contract_module = importlib.import_module("your.utils.candidate_contract")
    gpu_module = importlib.import_module("your.utils.gpu")

    class Candidate:
        def __init__(self, *_args, **_kwargs):
            self.your_header = SimpleNamespace(nchans=1)

        def close(self):
            pass

    class Resident:
        interval = None

        def __init__(self, *_args):
            pass

        def close(self):
            pass

    budgets = []
    monkeypatch.setattr(candidate_module, "Candidate", Candidate)
    monkeypatch.setattr(
        contract_module,
        "decoded_sample_dtype",
        lambda _candidate: np.dtype(np.uint8),
    )
    monkeypatch.setattr(gpu_module, "estimate_process_gpu_bytes", lambda *_args: 0)
    monkeypatch.setattr(scheduler, "_ResidentCache", Resident)
    monkeypatch.setattr(
        scheduler,
        "plan_batches",
        lambda *_args, memory_budget_bytes, **_kwargs: (
            budgets.append(memory_budget_bytes) or []
        ),
    )

    candidate_job = job(0, (0, 1), (0, 1))
    work = scheduler._Work(candidate_job.files, (candidate_job,))
    queued_work = iter((work, None))
    messages = []
    scheduler._gpu_worker(
        0,
        SimpleNamespace(get=lambda: next(queued_work)),
        SimpleNamespace(put=messages.append),
        None,
        explicit_budget,
        safety_fraction,
        1,
    )

    if expected is None:
        assert budgets == []
        assert messages[0]["type"] == "error"
        assert "no usable free memory" in messages[0]["error"]
    else:
        assert budgets == [expected]
        assert messages == [{"type": "done", "device": 0}]
    assert pool.free_calls == 1
    assert pool.get_limit_calls == 1
    assert pool.used_calls == bool(pool_limit)
    assert pool.default_pool_calls == 1


def test_run_jobs_failure_does_not_wait_for_unconsumed_queue_payload(tmp_path):
    from pathlib import Path

    repo = Path(__file__).parents[1]
    script = tmp_path / "queue_failure.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            import sys
            import time

            import numpy as np
            sys.path.insert(0, {str(repo)!r})
            import your.utils.candidate_scheduler as scheduler
            from your.utils.candidate_scheduler import CandidateJob

            class Plan:
                required_start = 0
                required_stop = 1

            def fail_without_consuming(device, _job_queue, result_queue, *_args):
                time.sleep(0.5)
                result_queue.put({{
                    "type": "error",
                    "device": device,
                    "index": 0,
                    "error": "synthetic worker failure",
                    "traceback": "",
                }})

            if __name__ == "__main__":
                scheduler._gpu_worker = fail_without_consuming
                job = CandidateJob(
                    ("obs.fil",),
                    {{"payload": np.zeros(16 * 1024 * 1024, dtype=np.uint8)}},
                    Plan(),
                    0,
                )
                try:
                    scheduler.run_jobs([job], gpu_ids=[0])
                except RuntimeError as error:
                    print(f"RUN_JOBS_RETURNED {{error}}", flush=True)
                print("SCRIPT_FINISHED", flush=True)
            """
        )
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "GPU 0 failed during candidate 0" in completed.stdout
    assert "SCRIPT_FINISHED" in completed.stdout


def test_candmaker_cpu_duplicate_rows_get_distinct_files_and_keep_10_field_api(
    tmp_path, monkeypatch
):
    import importlib.util
    from pathlib import Path

    import h5py

    script = Path(__file__).parents[1] / "bin" / "your_candmaker.py"
    spec = importlib.util.spec_from_file_location("your_candmaker_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Candidate:
        def __init__(self, _files, **_kwargs):
            self.id = "candidate-stable-id"
            self.nchans = 2
            self.data = self.dedispersed = self.dmt = None

        def make_products(
            self,
            *,
            target,
            time_size,
            frequency_size,
            dmsteps,
            output_dtype,
            profile,
        ):
            self.dedispersed = np.zeros((time_size, frequency_size), dtype=output_dtype)
            self.dmt = np.zeros((dmsteps, time_size), dtype=output_dtype)

        def save_h5(self, out_dir=None, fnout=None):
            filename = fnout or f"{self.id}.h5"
            path = Path(out_dir) / filename
            with h5py.File(path, "w") as output:
                output.create_dataset("data_freq_time", data=self.dedispersed)
                output.create_dataset("data_dm_time", data=self.dmt)
            return str(path)

        def close(self):
            pass

    monkeypatch.setattr(module, "Candidate", Candidate)
    args = type(
        "Args",
        (),
        {
            "opt_dm": False,
            "time_size": 2,
            "frequency_size": 3,
            "dmsteps": 4,
            "output_dtype": "float16",
            "fout": str(tmp_path),
            "spectral_kurtosis_sigma": 4,
            "savgol_frequency_window": 15,
            "savgol_sigma": 4,
            "flag_rfi": False,
        },
    )()
    candidate_row = ["obs.fil", 5.0, 4, 12.0, 1, 1.0, None, 1, args, -1]

    first = module._cand2h5([*candidate_row, 0], files=("obs.fil",))
    second = module._cand2h5([*candidate_row, 1], files=("obs.fil",))
    legacy = module._cand2h5(candidate_row, files=("obs.fil",))

    assert Path(first["output_file"]).name == "candidate-stable-id_0.h5"
    assert Path(second["output_file"]).name == "candidate-stable-id_1.h5"
    assert Path(legacy["output_file"]).name == "candidate-stable-id.h5"
    assert (
        len({first["output_file"], second["output_file"], legacy["output_file"]}) == 3
    )


def test_oom_serial_retry_releases_failed_future_temporaries(monkeypatch):
    class TrackedAllocation:
        pass

    class Pool:
        def free_all_blocks(self):
            pass

    class FakeCuPy:
        @staticmethod
        def get_default_memory_pool():
            return Pool()

        @staticmethod
        def get_default_pinned_memory_pool():
            return Pool()

    class Resident:
        array = object()
        start = 0

    refs = []
    calls = {}
    lock = Lock()
    concurrent = Barrier(2)

    def run_job(_cp, _process_gpu, _base, candidate_job, *_args):
        with lock:
            attempt = calls.get(candidate_job.index, 0) + 1
            calls[candidate_job.index] = attempt
        if attempt == 1:
            allocation = TrackedAllocation()
            refs.append(weakref.ref(allocation))
            concurrent.wait(timeout=2)
            raise MemoryError("simulated GPU allocation failure")
        gc.collect()
        assert all(reference() is None for reference in refs)
        return object(), {}, 0.0

    monkeypatch.setattr("your.utils.candidate_scheduler._run_gpu_job", run_job)
    jobs = (job(0, (0, 1), (0, 1)), job(1, (1, 2), (1, 2)))
    batch = CandidateBatch(jobs, 0, 2, 0, 0)

    results = _run_batch(FakeCuPy(), None, None, batch, Resident(), 2, 2)

    assert set(results) == {0, 1}
    assert all(result[2]["oom_serial_retry"] == 1 for result in results.values())
    assert all(reference() is None for reference in refs)


def test_single_job_batch_skips_executor(monkeypatch):
    class Resident:
        array = object()
        start = 3

    candidate_job = job(0, (0, 1), (0, 1))
    candidate = object()
    profile = {"processed": True}

    def run_job(_cp, _process_gpu, _base, actual_job, resident, start, device):
        assert actual_job is candidate_job
        assert resident is Resident.array
        assert (start, device) == (3, 2)
        return candidate, profile, 0.25

    def unexpected_executor(*_args, **_kwargs):
        raise AssertionError("singleton batches must not create an executor")

    monkeypatch.setattr(scheduler, "_run_gpu_job", run_job)
    monkeypatch.setattr(scheduler, "ThreadPoolExecutor", unexpected_executor)

    results = _run_batch(
        None,
        None,
        None,
        CandidateBatch((candidate_job,), 0, 1, 0, 0),
        Resident(),
        2,
        2,
    )

    assert results == {0: (candidate_job, candidate, profile, 0.25)}


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (ValueError("synthetic failure"), ValueError),
        (RuntimeError("cuda out of memory"), MemoryError),
    ],
)
def test_single_job_batch_preserves_errors_without_executor(
    monkeypatch, error, expected_type
):
    class Resident:
        array = object()
        start = 0

    candidate_job = job(0, (0, 1), (0, 1))

    def fail(*_args):
        raise error

    def unexpected_executor(*_args, **_kwargs):
        raise AssertionError("singleton batches must not create an executor")

    monkeypatch.setattr(scheduler, "_run_gpu_job", fail)
    monkeypatch.setattr(scheduler, "ThreadPoolExecutor", unexpected_executor)
    with pytest.raises(expected_type, match=str(error)) as caught:
        _run_batch(
            None,
            None,
            None,
            CandidateBatch((candidate_job,), 0, 1, 0, 0),
            Resident(),
            0,
            1,
        )
    if isinstance(error, ValueError):
        assert caught.value is error
    else:
        assert error.__traceback__ is None


def test_worker_oom_cleanup_drops_temporary_references_from_traceback():
    class TrackedAllocation:
        pass

    refs = []

    def fail_with_temporary():
        allocation = TrackedAllocation()
        refs.append(weakref.ref(allocation))
        raise MemoryError("simulated GPU allocation failure")

    caught = None
    try:
        fail_with_temporary()
    except MemoryError as error:
        caught = error
        _clear_exception_tracebacks(error)

    assert caught is not None
    gc.collect()
    assert refs[0]() is None
