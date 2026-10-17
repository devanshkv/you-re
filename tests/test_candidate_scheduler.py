import gc
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Event, Lock

import numpy as np
import pytest

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
    assert batches[0].estimated_peak_bytes == 42


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

    assert [batch.estimated_peak_bytes for batch in batches] == [10, 10]
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

    assert [batch.estimated_peak_bytes for batch in batches] == [10, 20]
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
        memory_budget_bytes=10,
        workspace_estimator=estimate,
        max_inflight=1,
    )

    assert len(batches) == 2
    assert [batch.estimated_peak_bytes for batch in batches] == [10, 10]


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


def test_resident_cache_rejects_dtype_mismatch_without_narrowing():
    class Stream:
        def synchronize(self):
            pass

    class Device:
        def __init__(self, _device):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    stream = Stream()

    class Cuda:
        pass

    Cuda.Stream = staticmethod(lambda non_blocking: stream)
    Cuda.Device = Device

    class FakeCuPy:
        cuda = Cuda()

        @staticmethod
        def empty(_shape, dtype):
            return type("Array", (), {"dtype": dtype})()

    class Reader:
        @staticmethod
        def get_data(_start, _count):
            return np.ones((1, 1), dtype=np.float32)

    cache = _ResidentCache(FakeCuPy(), Reader(), 0, 1, np.dtype(np.int32))
    with pytest.raises(TypeError, match="decoded sample dtype"):
        cache.ensure(0, 1)


def test_hdf5_save_is_queued_without_blocking_gpu_worker():
    started = Event()
    release = Event()

    class DelayedCandidate:
        def save_h5(self, out_dir=None):
            started.set()
            release.wait(timeout=1)
            return "candidate.h5"

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
