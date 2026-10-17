#!/usr/bin/env python3
"""Confirm staging winners combined with the measured DMT tile geometry.

Run only after reserving the GPU:
    CUDA_PATH=/usr/local/cuda-12.9 /tmp/your-cupy-env/bin/python \
        checks/benchmark_dmt_combined.py --rounds 3

This benchmark reads production sources, builds temporary RawModules, uploads
inputs before timing, and writes its report under /tmp by default.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import benchmark_dmt_geometry as geometry_bench  # noqa: E402
from checks import benchmark_dmt_staging as staging_bench  # noqa: E402
from checks.benchmark_ldg import make_module  # noqa: E402

DEFAULT_OUTPUT = Path("/tmp/benchmark_dmt_geometry_combined.json")
FAST_GEOMETRY_NAME = "t512_d4_ept2_c32"
CONDITIONS = ("baseline", "geometry_only", "staging_only", "combined")
CASES = (
    ("4096u8_dm80_bin128", "endpoints_guarded"),
    ("4096u8_dm700_bin128", "direct_gather"),
)


def run_case(cp, gpu, case, condition, source_modules, original_compute):
    if condition == "baseline":
        geometry = geometry_bench.BASELINE
        staging = "baseline"
    elif condition == "geometry_only":
        geometry = source_modules["fast_geometry"]
        staging = "baseline"
    elif condition == "staging_only":
        geometry = geometry_bench.BASELINE
        staging = source_modules["stage_by_case"][case.name]
    elif condition == "combined":
        geometry = source_modules["fast_geometry"]
        staging = source_modules["stage_by_case"][case.name]
    else:
        raise ValueError(condition)

    started = perf_counter()
    guard_passed = None
    source_variant = "baseline"
    if staging == "endpoints_guarded":
        # Include the monotonicity guard's CPU cost in the measured wall time.
        guard_passed = staging_bench.delay_rows_monotone(case.plan)
        source_variant = "endpoints_guarded" if guard_passed else "baseline"
    elif staging == "direct_gather":
        source_variant = "baseline"

    module = source_modules["modules"][(source_variant, geometry.name)]
    original_capacity = gpu._dmt_shared_capacity
    original_kernel_functions = gpu._kernel_functions
    gpu._TIME_TILE = geometry.time_tile
    gpu._DM_TILE = geometry.dm_tile
    gpu._CHANNEL_TILE = geometry.channel_tile
    gpu._EPT = geometry.ept
    gpu._kernel_functions = lambda _cp, m=module: m
    geometry_bench.install_ept_launch(gpu, original_compute, geometry.ept)
    if staging == "direct_gather":
        gpu._dmt_shared_capacity = lambda _delays, _start, _stop, _limit: 0
    try:
        products = gpu.process_gpu(
            case.cand,
            case.plan,
            resident=case.resident,
            resident_start=0,
            device=0,
            profile=None,
        )
        wall_s = perf_counter() - started
    finally:
        gpu._compute_binned_products = original_compute
        gpu._dmt_shared_capacity = original_capacity
        gpu._kernel_functions = original_kernel_functions
    return products, wall_s, guard_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("--rounds must be at least 3")

    import cupy as cp

    from your.utils import gpu

    baseline = geometry_bench.KERNEL_PATH.read_text()
    source_hash = hashlib.sha256(baseline.encode()).hexdigest()
    if source_hash != geometry_bench.EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"unexpected production CUDA source hash {source_hash}")
    stage_sources = staging_bench.source_variants(baseline)
    fast_geometry = next(
        item for item in geometry_bench.GEOMETRIES if item.name == FAST_GEOMETRY_NAME
    )
    stage_by_case = dict(CASES)
    module_sources = {"baseline": baseline}
    if "endpoints_guarded" in stage_by_case.values():
        module_sources["endpoints_guarded"] = stage_sources["endpoints_guarded"]
    modules = {}
    variant_hashes = {}
    for source_name, source in module_sources.items():
        for kernel_geometry in (geometry_bench.BASELINE, fast_geometry):
            code = geometry_bench.geometry_source(source, kernel_geometry)
            modules[(source_name, kernel_geometry.name)] = make_module(cp, code)
            variant_hashes[f"{source_name}:{kernel_geometry.name}"] = hashlib.sha256(
                code.encode()
            ).hexdigest()

    props = cp.cuda.runtime.getDeviceProperties(0)
    old_state = {
        name: getattr(gpu, name)
        for name in (
            "_TIME_TILE",
            "_DM_TILE",
            "_CHANNEL_TILE",
            "_kernel_functions",
            "_dmt_shared_capacity",
        )
    }
    sentinel = object()
    old_ept = getattr(gpu, "_EPT", sentinel)
    original_compute = gpu._compute_binned_products
    gpu._EPT = 1
    report = {
        "gpu": props["name"].decode(),
        "source_sha256": source_hash,
        "rounds": args.rounds,
        "conditions": list(CONDITIONS),
        "stage_strategy_by_case": dict(CASES),
        "geometry": {
            "name": fast_geometry.name,
            "time_tile": fast_geometry.time_tile,
            "dm_tile": fast_geometry.dm_tile,
            "channel_tile": fast_geometry.channel_tile,
            "ept": fast_geometry.ept,
        },
        "source_variant_sha256": variant_hashes,
        "timing_protocol": (
            "randomized paired full process_gpu wall; inputs resident before timing; "
            "profile=None, production events retained; endpoint monotonicity guard "
            "runs inside the timed call and falls back to baseline source when false"
        ),
        "cases": {},
    }
    source_modules = {
        "modules": modules,
        "fast_geometry": fast_geometry,
        "stage_by_case": stage_by_case,
    }
    rng = np.random.default_rng(20261018)
    try:
        for case_index, (case_name, _stage) in enumerate(CASES):
            specs = {item["name"]: item for item in geometry_bench.case_specs(700.0)}
            case = geometry_bench.make_case(specs[case_name])
            case.resident = cp.ascontiguousarray(
                cp.asarray(np.ascontiguousarray(case.data.T))
            )
            cp.cuda.get_current_stream().synchronize()
            if case.plan.pad_before or case.plan.pad_after:
                raise AssertionError(f"{case_name} must cover the complete DMT output")

            reference, _, _ = run_case(
                cp, gpu, case, "baseline", source_modules, original_compute
            )
            results = {
                condition: {"wall_s": [], "max_differences": {}}
                for condition in CONDITIONS
            }
            for condition in CONDITIONS:
                products, _, _ = run_case(
                    cp, gpu, case, condition, source_modules, original_compute
                )
                results[condition]["max_differences"] = geometry_bench.compare_products(
                    products,
                    reference,
                    f"{case_name}/{condition}/warmup",
                    case.cand.data.dtype,
                )

            order = list(CONDITIONS)
            for round_index in range(args.rounds):
                rng.shuffle(order)
                for condition in order:
                    products, wall_s, guard_passed = run_case(
                        cp, gpu, case, condition, source_modules, original_compute
                    )
                    results[condition]["max_differences"] = (
                        geometry_bench.compare_products(
                            products,
                            reference,
                            f"{case_name}/{condition}/round{round_index}",
                            case.cand.data.dtype,
                        )
                    )
                    results[condition]["wall_s"].append(wall_s)
                    if guard_passed is not None:
                        results[condition].setdefault(
                            "guard_passed_per_round", []
                        ).append(bool(guard_passed))
                print(
                    f"{case_name}: paired round {round_index + 1}/{args.rounds}",
                    flush=True,
                )

            baseline_s = geometry_bench.median(results["baseline"]["wall_s"])
            for condition in CONDITIONS:
                samples = results[condition].pop("wall_s")
                results[condition]["wall"] = geometry_bench.summarize_samples(samples)
                results[condition]["ratio_to_baseline"] = (
                    results[condition]["wall"]["median_s"] / baseline_s
                )
            report["cases"][case_name] = {
                "input": {
                    "shape": list(case.data.shape),
                    "dtype": str(case.data.dtype),
                    "resident_bytes": int(case.resident.nbytes),
                    "delay_rows_monotone": staging_bench.delay_rows_monotone(case.plan),
                },
                **results,
            }
            del case
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        gpu._compute_binned_products = original_compute
        for name, value in old_state.items():
            setattr(gpu, name, value)
        if old_ept is sentinel:
            if hasattr(gpu, "_EPT"):
                del gpu._EPT
        else:
            gpu._EPT = old_ept

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for case_name in report["cases"]:
        print(case_name)
        for condition in CONDITIONS:
            item = report["cases"][case_name][condition]
            print(
                f"  {condition:13s} {item['wall']['median_s'] * 1e3:8.2f} ms "
                f"{item['ratio_to_baseline']:.3f}x"
            )
    print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
