"""Run unchanged and new candmaker in separate processes with stage timings.

Example: python checks/benchmark_candidates.py --csv /data/candidates.csv \
    --baseline /home/riftuser/you-re --output /tmp/candidate-benchmark --gpu-ids 0
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def instrumented_run(checkout, stage_file, argv):
    """Install wall timers before candmaker imports; each event is one append."""
    import functools
    import runpy

    sys.path.insert(0, str(checkout))
    from your.candidate import Candidate
    from your.utils import gpu, math, misc

    def wrap(owner, name, stage):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                event = (
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "stage": stage,
                            "seconds": time.perf_counter() - started,
                        }
                    )
                    + "\n"
                )
                fd = os.open(stage_file, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                try:
                    os.write(fd, event.encode())
                finally:
                    os.close(fd)

        setattr(owner, name, measured)

    wrap(Candidate, "get_chunk", "read_and_preprocess")
    wrap(Candidate, "save_h5", "write_hdf5")
    wrap(gpu, "gpu_dedisp_and_dmt_crop", "gpu_products_including_transfers")
    wrap(math, "normalise", "normalize_cpu")
    wrap(misc, "_resize", "resize_cpu")
    sys.argv = [str(checkout / "bin/your_candmaker.py"), *argv]
    runpy.run_path(sys.argv[0], run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--current", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--inflight", type=int, default=2)
    parser.add_argument("--memory-mb", type=int)
    args = parser.parse_args()
    if args.repeat < 1 or (args.limit is not None and args.limit < 1):
        parser.error("repeat and limit must be positive")
    import pandas as pd

    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.csv)
    if args.limit is not None:
        frame = frame.head(args.limit)
    for column in ("file", "chan_mask_path"):
        for idx, value in frame[column].items():
            if isinstance(value, str) and not Path(value).is_absolute():
                candidate = args.csv.resolve().parent / value
                if candidate.exists():
                    frame.loc[idx, column] = str(candidate)
                elif Path(value).exists():
                    frame.loc[idx, column] = str(Path(value).resolve())
    resolved_csv = args.output.resolve() / "candidates.csv"
    frame.to_csv(resolved_csv, index=False)
    rows = []
    # Alternate order to reduce systematic warm filesystem-cache bias.
    for repetition in range(args.repeat):
        names = (
            ("baseline", "current") if repetition % 2 == 0 else ("current", "baseline")
        )
        for name in names:
            checkout = getattr(args, name).resolve()
            folder = args.output.resolve() / f"{name}-{repetition}"
            folder.mkdir(exist_ok=True)
            stage_file = folder / "stages.jsonl"
            stage_file.write_text("")
            cli = [
                "-c",
                str(resolved_csv),
                "-o",
                str(folder),
                "-g",
                *map(str, args.gpu_ids),
                "-n",
                str(args.nproc),
                "--no_log_file",
            ]
            if name == "current":
                cli += [
                    "--profile_json",
                    str(folder / "profile.json"),
                    "--gpu_inflight",
                    str(args.inflight),
                ]
                if args.memory_mb:
                    cli += ["--gpu_memory_mb", str(args.memory_mb)]
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--instrument",
                str(checkout),
                str(stage_file),
                *cli,
            ]
            env = dict(os.environ, PYTHONPATH=str(checkout))
            started = time.perf_counter()
            with (folder / "run.log").open("w") as log:
                result = subprocess.run(
                    command, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            elapsed = time.perf_counter() - started
            stages = {}
            for line in stage_file.read_text().splitlines():
                event = json.loads(line)
                stages[event["stage"]] = (
                    stages.get(event["stage"], 0) + event["seconds"]
                )
            profile_path = folder / "profile.json"
            input_bytes = None
            if name == "current" and profile_path.exists():
                reports = json.loads(profile_path.read_text())
                for report in reports:
                    for key, value in report.get("profile", {}).items():
                        if key.endswith(("_s", "_seconds")) and isinstance(
                            value, (int, float)
                        ):
                            stages[key] = stages.get(key, 0.0) + value
                input_bytes = {
                    key: sum(
                        report.get(key, report.get("profile", {}).get(key, 0))
                        for report in reports
                    )
                    for key in ("read_bytes", "upload_bytes")
                }
            row = {
                "version": name,
                "repeat": repetition,
                "returncode": result.returncode,
                "seconds": elapsed,
                "candidates": len(frame),
                "candidates_per_second": len(frame) / elapsed
                if not result.returncode
                else None,
                "stage_worker_seconds": stages,
                "input_bytes": input_bytes,
                "checkout": str(checkout),
                "log": str(folder / "run.log"),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "summary.json").write_text(json.dumps(rows, indent=2))
    if any(row["returncode"] for row in rows):
        raise SystemExit(
            "Some runs failed; inspect run.log. Failed runs have no throughput result."
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--instrument":
        instrumented_run(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
    else:
        main()
