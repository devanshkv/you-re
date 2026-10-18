"""Measure synthetic candidate processing, excluding file I/O and GPU upload.

Run against separately built checkouts with --checkout. JSON records all warmed
wall samples and a separate diagnostic profile; NPZ files retain final images.
This is a processing benchmark, not a substitute for real-file CLI throughput.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--widths", nargs="+", type=int, default=[4, 8, 32, 128])
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument("--dm", type=float, default=80.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeat < 1
        or args.threads < 1
        or args.channels < 2
        or any(w < 1 for w in args.widths)
    ):
        parser.error("repeat, widths and channel count must be positive (channels >=2)")
    sys.path.insert(0, str(args.checkout.resolve()))

    import numpy as np

    from your.utils.candidate_pipeline import make_plan, process_cpu

    cp = None
    if args.backend == "gpu":
        import cupy as cp

        from your.utils.gpu import process_gpu

    report = {
        "checkout": str(args.checkout.resolve()),
        "backend": args.backend,
        "cpu_threads": args.threads,
        "numpy": np.__version__,
        "scope": "warmed processing and finishing; excludes I/O, GPU input upload",
        "cases": [],
    }
    if cp is not None:
        report["cupy"] = cp.__version__
        report["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for width in args.widths:
        frequencies = np.linspace(1500.0, 1200.0, args.channels)
        cand = SimpleNamespace(
            dm=args.dm,
            tcand=5.0,
            width=width,
            native_tsamp=0.000256,
            chan_freqs=frequencies,
            your_header=SimpleNamespace(
                nspectra=1000000, foff=frequencies[1] - frequencies[0]
            ),
            kill_mask=np.array([False]),
            flag_rfi=False,
        )
        plan = make_plan(cand, output_dtype="float32")
        rng = np.random.default_rng(20261018 + width)
        data = rng.integers(
            0,
            256,
            size=(plan.read_stop - plan.read_start, args.channels),
            dtype=np.uint8,
        )
        resident = None
        if cp is not None:
            resident = cp.asarray(np.ascontiguousarray(data.T))
            cp.cuda.get_current_stream().synchronize()

        def run(profile=None, *, case_data=data, case_resident=resident):
            if cp is None:
                options = {} if args.threads == 1 else {"threads": args.threads}
                return process_cpu(
                    cand, plan, data=case_data, profile=profile, **options
                )
            return process_gpu(
                cand,
                plan,
                resident=case_resident,
                resident_start=plan.read_start,
                profile=profile,
            )

        run()  # Compile kernels and warm allocations before timing.
        samples = []
        for _ in range(args.repeat):
            started = perf_counter()
            products = run()
            samples.append(perf_counter() - started)
        profile = {}
        run(profile)
        row = {
            "width": width,
            "dm": args.dm,
            "channels": args.channels,
            "input_bytes": data.nbytes,
            "seconds": samples,
            "median_seconds": float(np.median(samples)),
            "diagnostic_profile": profile,
        }
        report["cases"].append(row)
        np.savez(
            args.output.with_name(args.output.stem + f"-w{width}.npz"),
            ft=products[0],
            dmt=products[1],
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del run, resident, data
        if cp is not None:
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
