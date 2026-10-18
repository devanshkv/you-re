from types import SimpleNamespace

import numpy as np
import pytest

from your.utils.candidate_contract import decoded_sample_dtype, parse_kill_mask


def test_cpu_candmaker_divides_affinity_threads_between_workers(monkeypatch):
    import os
    import runpy
    from pathlib import Path

    cli = runpy.run_path(str(Path(__file__).parents[1] / "bin" / "your_candmaker.py"))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(6)), raising=False)
    assert cli["worker_threads"](2) == 3
    assert cli["worker_threads"](11) == 1
    captured = {}

    def make_products(**kwargs):
        captured.update(kwargs)
        return "processed"

    args = SimpleNamespace(
        opt_dm=False,
        time_size=9,
        frequency_size=17,
        dmsteps=3,
        output_dtype="float32",
        nproc=2,
    )
    result = cli["cpu_dedisp_dmt"](SimpleNamespace(make_products=make_products), args)
    assert result == "processed"
    assert captured["target"] == "CPU"
    assert captured["cpu_threads"] == 3


@pytest.mark.parametrize(
    "mask",
    [np.array([], dtype=bool), np.array([False]), np.array(False)],
)
def test_empty_and_false_kill_masks_disable_no_channels(mask):
    np.testing.assert_array_equal(parse_kill_mask(mask, 3), [False, False, False])


def test_kill_mask_parser_accepts_channel_masks_and_integer_indices():
    np.testing.assert_array_equal(
        parse_kill_mask(np.array([False, True, False]), 3), [False, True, False]
    )
    np.testing.assert_array_equal(
        parse_kill_mask(np.array([-1, 0]), 3), [True, False, True]
    )


@pytest.mark.parametrize(
    "mask",
    [np.array([True]), np.array([True, False]), np.array([[0]], dtype=np.int16)],
)
def test_kill_mask_parser_rejects_invalid_shapes(mask):
    with pytest.raises(ValueError, match="kill_mask"):
        parse_kill_mask(mask, 3)


def test_decoded_sample_dtype_probes_nonempty_reader_data():
    class Reader:
        your_header = SimpleNamespace(
            nspectra=4, dtype=np.int16, format="fits", nbits=32
        )

        def __init__(self):
            self.calls = []

        def get_data(self, nstart, nsamp):
            self.calls.append((nstart, nsamp))
            return np.zeros((nsamp, 2), dtype=np.float32)

    reader = Reader()

    assert decoded_sample_dtype(reader) == np.dtype(np.float32)
    assert reader.calls == [(0, 1)]


def test_decoded_sample_dtype_uses_zero_length_fallbacks_without_reading():
    class Reader:
        def __init__(self, header):
            self.your_header = header

        def get_data(self, *_args):
            raise AssertionError("empty readers must not be probed")

    psrfits = Reader(
        SimpleNamespace(nspectra=0, dtype=np.int32, format="fits", nbits=32)
    )
    generic = Reader(
        SimpleNamespace(nspectra=0, dtype=np.uint16, format="sigproc", nbits=16)
    )

    assert decoded_sample_dtype(psrfits) == np.dtype(np.float32)
    assert decoded_sample_dtype(generic) == np.dtype(np.uint16)


def test_candmaker_parent_uses_compact_plans_and_preserves_csv_width(
    tmp_path, monkeypatch
):
    import csv
    import runpy
    import sys
    from pathlib import Path

    from your import candidate as candidate_module
    from your.utils import candidate_pipeline, candidate_scheduler

    class Reader:
        instances = []

        def __init__(self, _files, **kwargs):
            self.instances.append(self)
            self.native_tsamp = 0.001
            self.chan_freqs = np.array([1400.0, 1000.0])
            self.nchans = 2
            self.your_header = SimpleNamespace(nspectra=2000)
            self.id = "candidate-id"
            self.closed = False
            self._reset_candidate(width=2, tcand=0.5, **kwargs)

        def _reset_candidate(self, **kwargs):
            self.__dict__.update(kwargs)
            return self

        def close(self):
            self.closed = True

    csv_path = tmp_path / "candidates.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "file",
                "snr",
                "width",
                "dm",
                "label",
                "stime",
                "chan_mask_path",
                "num_files",
            ]
        )
        writer.writerow(["obs.fil", 10, 3, 20, 0, 0.5, "", 1])
        writer.writerow(["obs.fil", 10, 4, 20, 0, 0.5, "", 1])

    full_planner = candidate_pipeline.make_plan
    compact_planner = candidate_pipeline.make_compact_plan
    full_calls, compact_widths, jobs = [], [], []

    def record_full(candidate, *args, **kwargs):
        full_calls.append(candidate)
        return full_planner(candidate, *args, **kwargs)

    def record_compact(candidate, *args, **kwargs):
        compact_widths.append(candidate.width)
        return compact_planner(candidate, *args, **kwargs)

    def capture_jobs(planned_jobs, **kwargs):
        assert full_calls == []
        assert kwargs["gpu_ids"] == [0]
        jobs.extend(planned_jobs)
        return []

    monkeypatch.setattr(candidate_module, "Candidate", Reader)
    monkeypatch.setattr(candidate_pipeline, "make_plan", record_full)
    monkeypatch.setattr(candidate_pipeline, "make_compact_plan", record_compact)
    monkeypatch.setattr(candidate_scheduler, "run_jobs", capture_jobs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "your_candmaker.py",
            "-c",
            str(csv_path),
            "-g",
            "0",
            "-o",
            str(tmp_path),
            "--time_size",
            "4",
            "--frequency_size",
            "3",
            "--dmsteps",
            "256",
            "--no_log_file",
        ],
    )
    runpy.run_path(
        str(Path(__file__).parents[1] / "bin" / "your_candmaker.py"),
        run_name="__main__",
    )

    assert compact_widths == [8, 16]
    assert [job.candidate_kwargs["width"] for job in jobs] == [8, 16]
    assert [job.output_filename for job in jobs] == [
        "candidate-id.h5",
        "candidate-id_1.h5",
    ]
    assert len(Reader.instances) == 1
    assert Reader.instances[0].closed
    assert full_calls == []
    assert all(
        isinstance(job.plan, candidate_scheduler.CompactCandidatePlan) for job in jobs
    )
    assert all(
        not any(isinstance(value, np.ndarray) for value in vars(job.plan).values())
        for job in jobs
    )

    active = Reader.instances[0]._reset_candidate(**jobs[0].candidate_kwargs)
    materialized = candidate_scheduler._materialize_plan(active, jobs[0].plan)
    assert full_calls == [active]
    assert materialized.delays.shape == (256, 2)
    assert materialized.required_start == jobs[0].plan.required_start
    assert materialized.required_stop == jobs[0].plan.required_stop
