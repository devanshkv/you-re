#!/usr/bin/env python3
"""Compare read-only CUDA load variants for the direct GPU product kernels.

Run with:
    CUDA_PATH=/usr/local/cuda-12.9 /tmp/your-cupy-env/bin/python checks/benchmark_ldg.py

The production .cu file is only read. Variant source is built in memory and the
JSON report defaults to /tmp/benchmark_ldg_results.json.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "your/utils/gpu_kernels.cu"
VARIANTS = ("baseline", "restrict", "ldg")
EXPECTED_BASELINE_SHA256 = (
    "ef0c47434f622a613e7c281483d7ff65e248dddfa031558e8ffb31e6eb03cd49"
)
POINTERS = (
    "source",
    "medians",
    "enabled",
    "ft_delays",
    "delays",
    "output",
    "tile",
    "native",
)
DTYPES = (
    np.uint8,
    np.uint16,
    np.int16,
    np.int32,
    np.float16,
    np.float32,
    np.float64,
)
SUFFIXES = ("u8", "u16", "i16", "i32", "f16", "f32", "f64")


def variant_source(baseline, variant):
    code = baseline
    if variant == "restrict":
        for name in POINTERS:
            code = code.replace("*" + name, "* __restrict__ " + name)
    elif variant == "ldg":
        code = re.sub(
            r"\b(source|medians|enabled|ft_delays|delays|native)\s*\[([^\]]+)\]",
            lambda m: f"__ldg({m.group(1)} + ({m.group(2)}))",
            code,
        )
    elif variant != "baseline":
        raise ValueError(f"unknown variant {variant}")
    return code


def make_module(cp, code):
    module = cp.RawModule(code=code, options=("--std=c++11",), backend="nvrtc")
    names = ["bin_ft", "bin_dmt"]
    names += [f"dedisp_ft_{suffix}" for suffix in SUFFIXES]
    names += [f"dmt_tiled_{suffix}" for suffix in SUFFIXES]
    return module, {name: module.get_function(name) for name in names}


def ptx_load_summary(code, cc):
    from cupy.cuda.compiler import _NVRTCProgram

    program = _NVRTCProgram(code, "gpu_kernels.cu", (), (), method="ptx")
    options = (
        "--std=c++11",
        f"--gpu-architecture=compute_{cc}",
        "--device-as-default-execution-space",
        "-I/usr/local/cuda-12.9/include",
    )
    ptx, _ = program.compile(options)
    ptx = ptx.decode()
    ops = re.findall(r"\bld\.global(?:\.[a-z0-9]+)*", ptx)
    return {
        "ptx_bytes": len(ptx),
        "global_load_count": len(ops),
        "global_load_modes": {op: ops.count(op) for op in sorted(set(ops))},
    }


@dataclass
class Case:
    name: str
    data: np.ndarray
    cand: object
    plan: object
    resident: object = None


def make_case(
    name, nchans, rows, dtype, time_size, width, dmsteps, tsamp, tcand, dm=80.0
):
    from your.utils.candidate_pipeline import make_plan

    freqs = np.linspace(400.0, 800.0, nchans, dtype=np.float64)
    killed = np.zeros(nchans, dtype=bool)
    killed[[0, nchans // 2]] = True
    cand = SimpleNamespace(
        data=None,
        dm=dm,
        tcand=tcand,
        width=width,
        native_tsamp=tsamp,
        chan_freqs=freqs,
        your_header=SimpleNamespace(
            nspectra=0 if rows is None else rows,
            dtype=np.dtype(dtype),
            tsamp=tsamp,
            foff=-0.1,
        ),
        kill_mask=killed,
        flag_rfi=False,
    )
    plan = make_plan(
        cand,
        time_size=time_size,
        frequency_size=256,
        dmsteps=dmsteps,
        output_dtype=np.float32,
    )
    if rows is None:
        if plan.required_start < 0:
            raise ValueError("full-coverage cases must start at a nonnegative sample")
        rows = plan.required_stop
        cand.your_header.nspectra = rows
        plan = make_plan(
            cand,
            time_size=time_size,
            frequency_size=256,
            dmsteps=dmsteps,
            output_dtype=np.float32,
        )

    rng = np.random.default_rng(4159 + nchans + rows)
    if np.dtype(dtype).kind in "iu":
        data = rng.integers(0, 256, size=(rows, nchans), dtype=np.uint8)
        if np.dtype(dtype) != np.dtype(np.uint8):
            data = data.astype(dtype)
    else:
        data = rng.normal(14.0, 7.0, size=(rows, nchans)).astype(dtype)
        if rows > 4 and nchans > 4:
            data[1, 2] = np.nan
            data[3, 1] = np.inf
    cand.data = data
    return Case(name=name, data=data, cand=cand, plan=plan)


def compare_products(actual, expected, context):
    if expected is None:
        return
    for label, got, want in zip(("FT", "DMT"), actual, expected):
        np.testing.assert_allclose(
            got,
            want,
            rtol=2e-5,
            atol=2e-5,
            err_msg=f"{context} {label} differs",
        )


def summarize(samples):
    values = [float(value) for value in samples]
    return {
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
        "samples_s": values,
    }


def run_process(gpu, cp, modules, variant, case, *, profile=True):
    module, kernels = modules[variant]
    gpu._kernel_functions = lambda _cp: (module, kernels)
    measurements = {} if profile else None
    started = perf_counter()
    products = gpu.process_gpu(
        case.cand,
        case.plan,
        resident=case.resident,
        resident_start=0,
        device=0,
        profile=measurements,
    )
    wall_s = perf_counter() - started
    return products, wall_s, measurements or {}


def benchmark_case(gpu, cp, modules, case, repeats, seed):
    rng = np.random.default_rng(seed)
    reference = None
    for variant in rng.permutation(VARIANTS):
        products, _, _ = run_process(gpu, cp, modules, str(variant), case)
        compare_products(products, reference, f"{case.name}/{variant}")
        if variant == "baseline":
            reference = tuple(np.array(value, copy=True) for value in products)
    if reference is None:
        raise AssertionError("baseline warmup did not run")

    results = {variant: {"wall_s": [], "profile": {}} for variant in VARIANTS}
    for repeat in range(repeats):
        for variant in rng.permutation(VARIANTS):
            variant = str(variant)
            products, wall_s, profile = run_process(gpu, cp, modules, variant, case)
            compare_products(
                products, reference, f"{case.name}/{variant}/repeat{repeat}"
            )
            results[variant]["wall_s"].append(wall_s)
            for name, seconds in profile.items():
                results[variant]["profile"].setdefault(name, []).append(float(seconds))
        print(f"completed {case.name} paired round {repeat + 1}/{repeats}", flush=True)

    summary = {}
    for variant, values in results.items():
        summary[variant] = {"wall": summarize(values["wall_s"]), "profile": {}}
        for name, samples in values["profile"].items():
            summary[variant]["profile"][name] = summarize(samples)
    return summary


def measure_kernel_kinds(gpu, cp, modules, case, seed):
    """Time each production launch with per-call CUDA events in a separate run."""

    class TimedKernel:
        def __init__(self, function, name):
            self.function = function
            self.name = name
            self.events = []

        def __call__(self, *args, **kwargs):
            stream = kwargs.get("stream") or cp.cuda.get_current_stream()
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            result = self.function(*args, **kwargs)
            stop.record(stream)
            self.events.append((start, stop))
            return result

    timings = {}
    for variant in np.random.default_rng(seed).permutation(VARIANTS):
        module, kernels = modules[str(variant)]
        wrapped = {name: TimedKernel(fn, name) for name, fn in kernels.items()}
        gpu._kernel_functions = lambda _cp, m=module, f=wrapped: (m, f)
        gpu.process_gpu(
            case.cand,
            case.plan,
            resident=case.resident,
            resident_start=0,
            device=0,
        )
        cp.cuda.Device().synchronize()
        values = {}
        for name, function in wrapped.items():
            events = function.events
            if events:
                values[name] = (
                    sum(cp.cuda.get_elapsed_time(a, b) for a, b in events) / 1000
                )
        timings[str(variant)] = values
    return timings


def dtype_smoke(gpu, cp, modules):
    results = {}
    for dtype in DTYPES:
        case = make_case(
            f"dtype_{np.dtype(dtype).name}",
            nchans=16,
            rows=512,
            dtype=dtype,
            time_size=8,
            width=2,
            dmsteps=8,
            tsamp=1e-4,
            tcand=4e-4,
        )
        case.resident = cp.ascontiguousarray(
            cp.asarray(np.ascontiguousarray(case.data.T))
        )
        cp.cuda.get_current_stream().synchronize()
        baseline, _, _ = run_process(gpu, cp, modules, "baseline", case)
        for variant in VARIANTS[1:]:
            products, _, _ = run_process(gpu, cp, modules, variant, case)
            compare_products(
                products, baseline, f"dtype_{np.dtype(dtype).name}/{variant}"
            )
        results[np.dtype(dtype).name] = "passed"
        del case.resident
        cp.get_default_memory_pool().free_all_blocks()
    return results


def dmt_shared_details(gpu, cp, plan):
    limit_bytes = gpu._shared_limit(cp)
    limit_floats = limit_bytes // np.dtype(np.float32).itemsize
    required_floats = 0
    capacities = []
    for channel_start in range(0, plan.nchans, gpu._CHANNEL_TILE):
        channel_stop = min(plan.nchans, channel_start + gpu._CHANNEL_TILE)
        capacity = gpu._dmt_shared_capacity(
            plan.delays, channel_start, channel_stop, limit_bytes
        )
        capacities.append(capacity)
        for dm_start in range(0, plan.delays.shape[0], gpu._DM_TILE):
            delay_tile = plan.delays[
                dm_start : dm_start + gpu._DM_TILE, channel_start:channel_stop
            ]
            spread = int(np.max(np.ptp(delay_tile, axis=0)))
            required_floats = max(
                required_floats, min(gpu._TIME_TILE, plan.native_count) + spread
            )
    return {
        "limit_bytes": limit_bytes,
        "limit_floats": limit_floats,
        "required_floats": required_floats,
        "capacities_floats": sorted(set(capacities)),
        "fallback_expected": required_floats > limit_floats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dm", type=float, default=80.0)
    parser.add_argument("--large-only", action="store_true")
    parser.add_argument("--output", default="/tmp/benchmark_ldg_results.json")
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if not np.isfinite(args.dm) or args.dm < 0:
        parser.error("--dm must be finite and nonnegative")

    sys.path.insert(0, str(ROOT))
    import cupy as cp

    from your.utils import gpu

    baseline = KERNEL_PATH.read_text()
    source_hash = hashlib.sha256(baseline.encode()).hexdigest()
    props = cp.cuda.runtime.getDeviceProperties(0)
    cc = int(props["major"]) * 10 + int(props["minor"])
    modules = {}
    ptx = {}
    for variant in VARIANTS:
        code = variant_source(baseline, variant)
        modules[variant] = make_module(cp, code)
        ptx[variant] = ptx_load_summary(code, cc)
        print(f"compiled {variant}", flush=True)

    cases = []
    if not args.large_only:
        cases.append(
            make_case(
                "336ch_f32_bin4_dm256",
                nchans=336,
                rows=8192,
                dtype=np.float32,
                time_size=128,
                width=8,
                dmsteps=256,
                tsamp=1e-4,
                tcand=0.01,
                dm=args.dm,
            )
        )
    dm_label = f"{args.dm:g}"
    large_narrow = make_case(
        f"4096ch_u8_bin1_dm{dm_label}",
        nchans=4096,
        rows=None,
        dtype=np.uint8,
        time_size=256,
        width=2,
        dmsteps=256,
        tsamp=1e-4,
        tcand=0.0128,
        dm=args.dm,
    )
    large_wide = make_case(
        f"4096ch_u8_bin128_dm{dm_label}",
        nchans=4096,
        rows=None,
        dtype=np.uint8,
        time_size=256,
        width=256,
        dmsteps=256,
        tsamp=1e-4,
        tcand=1.6384,
        dm=args.dm,
    )
    cases.extend((large_narrow, large_wide))
    for case in cases[-2:]:
        if (
            case.plan.output_start != 0
            or case.plan.pad_before != 0
            or case.plan.pad_after != 0
        ):
            raise AssertionError(f"{case.name} does not have full unpadded coverage")

    old_kernel_functions = gpu._kernel_functions
    try:
        correctness = dtype_smoke(gpu, cp, modules)
        results = {}
        case_storage = {}
        individual_kernels = {}
        for idx, case in enumerate(cases):
            case.resident = cp.ascontiguousarray(
                cp.asarray(np.ascontiguousarray(case.data.T))
            )
            cp.cuda.get_current_stream().synchronize()
            shared = dmt_shared_details(gpu, cp, case.plan)
            case_storage[case.name] = {
                "host_source_bytes": int(case.data.nbytes),
                "resident_source_bytes": int(case.resident.nbytes),
                "planned_valid_rows": int(case.plan.read_stop - case.plan.read_start),
                "planned_valid_source_bytes": int(
                    (case.plan.read_stop - case.plan.read_start)
                    * case.plan.nchans
                    * case.data.dtype.itemsize
                ),
                "resident_start": 0,
                "plan_read_start": int(case.plan.read_start),
                "plan_read_stop": int(case.plan.read_stop),
                "required_start": int(case.plan.required_start),
                "required_stop": int(case.plan.required_stop),
                "output_start": int(case.plan.output_start),
                "pad_before": int(case.plan.pad_before),
                "pad_after": int(case.plan.pad_after),
                "time_bin": int(case.plan.time_bin),
                "native_count": int(case.plan.native_count),
                "dmsteps": int(case.plan.dmsteps),
                "dmt_shared": shared,
            }
            print(
                f"{case.name}: host/resident={case.data.nbytes / 2**20:.1f} MiB, "
                f"rows={case.plan.read_stop - case.plan.read_start}, "
                f"shared required={shared['required_floats']} floats, "
                f"capacity={shared['capacities_floats']} floats, "
                f"fallback={shared['fallback_expected']}",
                flush=True,
            )
            results[case.name] = benchmark_case(
                gpu, cp, modules, case, args.repeats, seed=20261017 + idx
            )
            individual_kernels[case.name] = measure_kernel_kinds(
                gpu, cp, modules, case, seed=20262017 + idx
            )
            case.resident = None
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        gpu._kernel_functions = old_kernel_functions

    report = {
        "gpu": props["name"].decode(),
        "compute_capability": f"{cc // 10}.{cc % 10}",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "source_sha256": source_hash,
        "baseline_sha256_expected": EXPECTED_BASELINE_SHA256,
        "repeats": args.repeats,
        "ptx_loads": ptx,
        "dtype_correctness": correctness,
        "cases": results,
        "dm": args.dm,
        "case_storage": case_storage,
        "individual_kernel_seconds_one_instrumented_run": individual_kernels,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for case_name, variants in results.items():
        print(f"\n{case_name}")
        base_wall = variants["baseline"]["wall"]["median_s"]
        for variant in VARIANTS:
            values = variants[variant]
            wall = values["wall"]["median_s"]
            dedisp = values["profile"].get("dedisp_s", {}).get("median_s", math.nan)
            bins = values["profile"].get("bin_s", {}).get("median_s", math.nan)
            print(
                f"{variant:8s} wall={wall * 1e3:8.2f} ms "
                f"({wall / base_wall:5.3f}x) dedisp={dedisp * 1e3:8.2f} ms "
                f"bin={bins * 1e3:7.3f} ms"
            )
    print(f"\nJSON: {output_path}")
    print(json.dumps({"ptx_loads": ptx, "dtype_correctness": correctness}, indent=2))


if __name__ == "__main__":
    main()
