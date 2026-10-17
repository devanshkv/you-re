#!/usr/bin/env python3
"""Bounded A/B benchmark for DMT sample staging strategies.

This imports the existing candidate builders and compares isolated RawModule
variants. Production sources and the existing benchmark harness stay read-only.
Results are written to /tmp/benchmark_dmt_staging_results.json.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "your/utils/gpu_kernels.cu"
EXPECTED_BASELINE_SHA256 = (
    "ef0c47434f622a613e7c281483d7ff65e248dddfa031558e8ffb31e6eb03cd49"
)
VARIANTS = (
    "baseline",
    "endpoints_guarded",
    "direct_gather",
    "spread_aware_4096",
    "spread_aware_1024",
)
TIME_TILE = 128
SHARED_CAPS = {"spread_aware_4096": 4096, "spread_aware_1024": 1024}
DTYPES = (
    np.uint8,
    np.uint16,
    np.int16,
    np.int32,
    np.float16,
    np.float32,
    np.float64,
)

sys.path.insert(0, str(ROOT))
from checks import benchmark_ldg as common  # noqa: E402


def _replace_endpoint_scans(source: str) -> str:
    scan = re.compile(
        r"(?P<indent> +)long long minimum = delays\[\(long long\)dm_base \* nchans \+ global_channel\];\n"  # noqa: E501
        r"(?P=indent)long long maximum = minimum;\n"
        r"(?P=indent)for \(int d = 1; d < dm_count; \+\+d\) \{\n"
        r"(?P=indent)    long long delay = delays\[\(long long\)\(dm_base \+ d\) \* nchans \+ global_channel\];\n"  # noqa: E501
        r"(?P=indent)    minimum = min\(minimum, delay\);\n"
        r"(?P=indent)    maximum = max\(maximum, delay\);\n"
        r"(?P=indent)\}"
    )

    def endpoint(match):
        indent = match.group("indent")
        return (
            f"{indent}long long minimum = delays["
            f"(long long)dm_base * nchans + global_channel];\n"
            f"{indent}long long maximum = delays["
            f"(long long)(dm_base + dm_count - 1) * nchans + global_channel];"
        )

    _, count = scan.subn(endpoint, source)
    if count != 2:
        raise RuntimeError(f"expected two per-channel DM-bound scans, found {count}")
    return scan.sub(endpoint, source)


def _spread_aware_source(source: str) -> str:
    start_marker = "    bool use_shared = true;"
    end_marker = """    if (time < native_count) {
        for (int d = 0; d < dm_count; ++d) {
            long long index = (long long)(dm_base + d) * native_count + time;"""
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    # These few embedded CUDA lines stay byte-for-byte intact for source generation.
    replacement = (
        "    for (int local_channel = 0; local_channel < group_channels; ++local_channel) {\n"  # noqa: E501
        "        int global_channel = channel_start + local_channel;\n"
        "        long long minimum = delays[(long long)dm_base * nchans + global_channel];\n"  # noqa: E501
        "        long long maximum = minimum;\n"
        "        for (int d = 1; d < dm_count; ++d) {\n"
        "            long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];\n"  # noqa: E501
        "            minimum = min(minimum, delay);\n"
        "            maximum = max(maximum, delay);\n"
        "        }\n"
        "        int count = tile_times + (int)(maximum - minimum);\n"
        "        if (count <= shared_capacity) {\n"
        "            long long slab_start = output_start + time_base + minimum;\n"
        "            for (int i = lane; i < count; i += TIME_TILE) {\n"
        "                tile[i] = sample_at(\n"
        "                    source, medians, enabled, local_channel, slab_start + i,\n"
        "                    source_start, source_count, valid_start, valid_stop, nspectra,\n"  # noqa: E501
        "                    input_scale\n"
        "                );\n"
        "            }\n"
        "            __syncthreads();\n"
        "            if (time < native_count && enabled[local_channel]) {\n"
        "                for (int d = 0; d < dm_count; ++d) {\n"
        "                    long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];\n"  # noqa: E501
        "                    sums[d] += tile[lane + (int)(delay - minimum)];\n"
        "                }\n"
        "            }\n"
        "            __syncthreads();\n"
        "        } else if (time < native_count) {\n"
        "            for (int d = 0; d < dm_count; ++d) {\n"
        "                long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];\n"  # noqa: E501
        "                sums[d] += sample_at(\n"
        "                    source, medians, enabled, local_channel, output_start + time + delay,\n"  # noqa: E501
        "                    source_start, source_count, valid_start, valid_stop, nspectra,\n"  # noqa: E501
        "                    input_scale\n"
        "                );\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "\n"
    )
    return source[:start] + replacement + source[end:]


def source_variants(baseline: str) -> dict[str, str]:
    return {
        "baseline": baseline,
        "endpoints_guarded": _replace_endpoint_scans(baseline),
        "direct_gather": baseline,
        "spread_aware_4096": _spread_aware_source(baseline),
        "spread_aware_1024": _spread_aware_source(baseline),
    }


def delay_rows_monotone(plan) -> bool:
    delays = np.asarray(plan.delays)
    return bool(delays.shape[0] < 2 or np.all(delays[1:] >= delays[:-1]))


def _effective_variant(variant: str, case) -> tuple[str, bool]:
    if variant == "endpoints_guarded" and not delay_rows_monotone(case.plan):
        return "baseline", True
    return variant, False


class _TimedKernel:
    def __init__(self, cp, function):
        self.cp = cp
        self.function = function
        self.events = []

    def __call__(self, *args, **kwargs):
        stream = kwargs.get("stream") or self.cp.cuda.get_current_stream()
        start, stop = self.cp.cuda.Event(), self.cp.cuda.Event()
        start.record(stream)
        result = self.function(*args, **kwargs)
        stop.record(stream)
        self.events.append((start, stop))
        return result

    def elapsed_seconds(self) -> float:
        return sum(self.cp.cuda.get_elapsed_time(a, b) for a, b in self.events) / 1000.0


def run_process(gpu, cp, modules, variant, case, *, profile=True, measure_dmt=False):
    started = perf_counter()
    effective, guard_applied = _effective_variant(variant, case)
    module, kernels = modules[effective]
    timed = {}
    if measure_dmt:
        wrapped = dict(kernels)
        for name, function in kernels.items():
            if name.startswith("dmt_tiled_"):
                timer = _TimedKernel(cp, function)
                wrapped[name] = timer
                timed[name] = timer
        kernels = wrapped

    original_capacity = gpu._dmt_shared_capacities
    original_kernel_functions = gpu._kernel_functions
    if variant == "direct_gather":
        gpu._dmt_shared_capacities = lambda delays, _limit: np.zeros(
            (delays.shape[1] + gpu._CHANNEL_TILE - 1) // gpu._CHANNEL_TILE,
            dtype=np.int64,
        )
    elif variant in SHARED_CAPS:
        shared_cap = SHARED_CAPS[variant]
        gpu._dmt_shared_capacities = lambda delays, limit: np.minimum(
            original_capacity(delays, limit),
            min(shared_cap, int(limit) // np.dtype(np.float32).itemsize),
        )

    gpu._kernel_functions = lambda _cp, m=module, f=kernels: (m, f)
    measurements = {} if profile else None
    try:
        products = gpu.process_gpu(
            case.cand,
            case.plan,
            resident=case.resident,
            resident_start=0,
            device=0,
            profile=measurements,
        )
        wall_s = perf_counter() - started
        dmt_s = (
            sum(timer.elapsed_seconds() for timer in timed.values())
            if measure_dmt
            else None
        )
    finally:
        gpu._dmt_shared_capacities = original_capacity
        gpu._kernel_functions = original_kernel_functions
    return products, wall_s, measurements or {}, dmt_s, effective, guard_applied


def _case_record(case):
    return {
        "name": case.name,
        "dtype": np.dtype(case.data.dtype).name,
        "nchans": int(case.plan.nchans),
        "rows": int(case.data.shape[0]),
        "required_start": int(case.plan.required_start),
        "required_stop": int(case.plan.required_stop),
        "read_start": int(case.plan.read_start),
        "read_stop": int(case.plan.read_stop),
        "output_start": int(case.plan.output_start),
        "native_count": int(case.plan.native_count),
        "time_bin": int(case.plan.time_bin),
        "time_size": int(case.plan.time_size),
        "dmsteps": int(case.plan.dmsteps),
        "full_coverage": bool(
            case.plan.output_start == 0
            and case.plan.pad_before == 0
            and case.plan.pad_after == 0
            and case.plan.read_start == 0
            and case.plan.read_stop == case.data.shape[0]
        ),
        "delay_rows_monotone": delay_rows_monotone(case.plan),
        "host_source_bytes": int(case.data.nbytes),
    }


def _resident_case(cp, case):
    case.resident = cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(case.data.T)))
    cp.cuda.get_current_stream().synchronize()


def _release_case(cp, case):
    case.resident = None
    cp.get_default_memory_pool().free_all_blocks()


def compare_products(actual, expected, context):
    """Require finite, shape-identical products and return explicit errors."""
    if expected is None:
        return {"max_abs_error": 0.0, "max_rel_error": 0.0, "products": {}}
    errors = {}
    for label, got, want in zip(("FT", "DMT"), actual, expected):
        got = np.asarray(got)
        want = np.asarray(want)
        if got.shape != want.shape:
            raise AssertionError(
                f"{context} {label} shape differs: {got.shape} != {want.shape}"
            )
        if not np.all(np.isfinite(want)):
            raise AssertionError(
                f"{context} expected {label} contains nonfinite values"
            )
        if not np.all(np.isfinite(got)):
            raise AssertionError(f"{context} actual {label} contains nonfinite values")
        difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
        max_abs = float(np.max(difference)) if difference.size else 0.0
        scale = np.maximum(np.abs(want.astype(np.float64)), 2e-5)
        max_rel = float(np.max(difference / scale)) if difference.size else 0.0
        np.testing.assert_allclose(
            got,
            want,
            rtol=2e-5,
            atol=2e-5,
            equal_nan=False,
            err_msg=f"{context} {label} differs",
        )
        errors[label] = {"max_abs_error": max_abs, "max_rel_error": max_rel}
    return {
        "max_abs_error": max(item["max_abs_error"] for item in errors.values()),
        "max_rel_error": max(item["max_rel_error"] for item in errors.values()),
        "products": errors,
    }


def _record_correctness(report, stage, case, variant, errors):
    report.setdefault("correctness_errors", []).append(
        {
            "stage": stage,
            "case": case.name,
            "variant": variant,
            **errors,
        }
    )


def _record_timing(
    stage, case, variant, wall, profile, dmt_s, effective, guard, errors=None
):
    result = {
        "stage": stage,
        "case": case.name,
        "variant": variant,
        "effective_variant": effective,
        "endpoint_guard_fell_back": bool(guard),
        "wall_s": float(wall),
        "dmt_kernel_cuda_event_s": None if dmt_s is None else float(dmt_s),
        "profile_event_s": {key: float(value) for key, value in profile.items()},
    }
    if errors is not None:
        result["correctness_error"] = errors
    return result


def _make_large_case(dm: float, width: int, dtype=np.uint8):
    wide = width == 256
    tcand = 1.6384 if wide else 0.0128
    suffix = f"bin{width // 2}"
    return common.make_case(
        f"dm{dm:g}_4096ch_{np.dtype(dtype).name}_{suffix}",
        nchans=4096,
        rows=None,
        dtype=dtype,
        time_size=256,
        width=width,
        dmsteps=256,
        tsamp=1e-4,
        tcand=tcand,
        dm=dm,
    )


def _make_dtype_smoke(dtype):
    return common.make_case(
        f"dtype_smoke_{np.dtype(dtype).name}_dm7_time129",
        nchans=16,
        rows=None,
        dtype=dtype,
        time_size=129,
        width=2,
        dmsteps=7,
        tsamp=1e-4,
        tcand=0.0064,
        dm=80.0,
    )


def _make_float_correctness(dtype):
    return common.make_case(
        f"float_correctness_336ch_{np.dtype(dtype).name}_dm17_time129",
        nchans=336,
        rows=None,
        dtype=dtype,
        time_size=129,
        width=2,
        dmsteps=17,
        tsamp=1e-4,
        tcand=0.0064,
        dm=80.0,
    )


def _make_unordered_case():
    case = common.make_case(
        "unordered_dm_rows_u8_dm8_time129",
        nchans=64,
        rows=None,
        dtype=np.uint8,
        time_size=129,
        width=2,
        dmsteps=8,
        tsamp=1e-4,
        tcand=0.0064,
        dm=80.0,
    )
    order = np.array([0, 1, 7, 2, 3, 4, 5, 6], dtype=np.intp)
    case.plan = dataclasses.replace(
        case.plan,
        dm_values=case.plan.dm_values[order].copy(),
        delays=case.plan.delays[order].copy(),
    )
    return case


def _correctness_suite(gpu, cp, modules, report):
    cases = [_make_dtype_smoke(dtype) for dtype in DTYPES]
    cases += [_make_float_correctness(np.float16), _make_float_correctness(np.float32)]
    cases.append(_make_unordered_case())
    status = {}
    for case in cases:
        case_info = _case_record(case)
        report["correctness_cases"][case.name] = case_info
        _resident_case(cp, case)
        try:
            baseline, wall, profile, dmt, effective, guard = run_process(
                gpu, cp, modules, "baseline", case, profile=True
            )
            baseline_error = compare_products(
                baseline, baseline, f"{case.name}/baseline"
            )
            _record_correctness(report, "correctness", case, "baseline", baseline_error)
            report["timings"].append(
                _record_timing(
                    "correctness_baseline",
                    case,
                    "baseline",
                    wall,
                    profile,
                    None,
                    effective,
                    guard,
                )
            )
            case_status = {"baseline": {"passed": True, **baseline_error}}
            for variant in VARIANTS[1:]:
                products, wall, profile, dmt, effective, guard = run_process(
                    gpu, cp, modules, variant, case, profile=True
                )
                errors = compare_products(products, baseline, f"{case.name}/{variant}")
                _record_correctness(report, "correctness", case, variant, errors)
                report["timings"].append(
                    _record_timing(
                        "correctness_variant",
                        case,
                        variant,
                        wall,
                        profile,
                        None,
                        effective,
                        guard,
                    )
                )
                case_status[variant] = {
                    "passed": True,
                    "effective_variant": effective,
                    "endpoint_guard_fell_back": bool(guard),
                    **errors,
                }
            status[case.name] = case_status
            print(f"correctness passed: {case.name}", flush=True)
        finally:
            _release_case(cp, case)
    report["correctness"] = status


def _measure_suite_case(gpu, cp, modules, case, report, rng):
    report["primary_cases"][case.name] = _case_record(case)
    _resident_case(cp, case)
    try:
        # Compare each full staging path on the exact production-shaped input.
        baseline, wall, profile, _, effective, guard = run_process(
            gpu, cp, modules, "baseline", case, profile=True
        )
        baseline_error = compare_products(baseline, baseline, f"{case.name}/baseline")
        _record_correctness(
            report, "primary_correctness", case, "baseline", baseline_error
        )
        report["timings"].append(
            _record_timing(
                "primary_correctness",
                case,
                "baseline",
                wall,
                profile,
                None,
                effective,
                guard,
            )
        )
        for variant in VARIANTS[1:]:
            products, wall, profile, _, effective, guard = run_process(
                gpu, cp, modules, variant, case, profile=True
            )
            errors = compare_products(products, baseline, f"{case.name}/{variant}")
            _record_correctness(report, "primary_correctness", case, variant, errors)
            report["timings"].append(
                _record_timing(
                    "primary_correctness",
                    case,
                    variant,
                    wall,
                    profile,
                    None,
                    effective,
                    guard,
                )
            )
        report["primary_correctness"][case.name] = {
            variant: {
                "passed": True,
                **next(
                    item
                    for item in reversed(report["correctness_errors"])
                    if item["case"] == case.name
                    and item["stage"] == "primary_correctness"
                    and item["variant"] == variant
                ),
            }
            for variant in VARIANTS
        }

        # Randomized warmup, then three cheap interleaved screening rounds.
        for variant in rng.permutation(VARIANTS):
            variant = str(variant)
            products, wall, profile, dmt, effective, guard = run_process(
                gpu, cp, modules, variant, case, profile=True
            )
            errors = compare_products(
                products, baseline, f"{case.name}/{variant}/warmup"
            )
            _record_correctness(report, "screen_warmup", case, variant, errors)
            report["timings"].append(
                _record_timing(
                    "screen_warmup", case, variant, wall, profile, dmt, effective, guard
                )
            )
        for round_index in range(3):
            for variant in rng.permutation(VARIANTS):
                variant = str(variant)
                products, wall, profile, dmt, effective, guard = run_process(
                    gpu, cp, modules, variant, case, profile=True
                )
                errors = compare_products(
                    products, baseline, f"{case.name}/{variant}/screen{round_index}"
                )
                _record_correctness(
                    report, f"screen_round_{round_index + 1}", case, variant, errors
                )
                report["timings"].append(
                    _record_timing(
                        f"screen_round_{round_index + 1}",
                        case,
                        variant,
                        wall,
                        profile,
                        dmt,
                        effective,
                        guard,
                    )
                )
            print(f"screened {case.name} round {round_index + 1}/3", flush=True)
    finally:
        _release_case(cp, case)


def _screen_scores(report):
    primary = list(report["primary_cases"])
    timings = report["timings"]
    score = {}
    for variant in VARIANTS:
        case_ratios = []
        case_wall_ratios = []
        for case in primary:
            selected = [
                item
                for item in timings
                if item["case"] == case
                and item["stage"].startswith("screen_round_")
                and item["variant"] == variant
            ]
            baseline = [
                item
                for item in timings
                if item["case"] == case
                and item["stage"].startswith("screen_round_")
                and item["variant"] == "baseline"
            ]
            variant_dedisp = statistics.median(
                item["profile_event_s"]["dedisp_s"] for item in selected
            )
            baseline_dedisp = statistics.median(
                item["profile_event_s"]["dedisp_s"] for item in baseline
            )
            variant_wall = statistics.median(item["wall_s"] for item in selected)
            baseline_wall = statistics.median(item["wall_s"] for item in baseline)
            case_ratios.append(variant_dedisp / baseline_dedisp)
            case_wall_ratios.append(variant_wall / baseline_wall)
        score[variant] = {
            "dedisp_geometric_mean_ratio": float(
                math.exp(sum(math.log(x) for x in case_ratios) / len(case_ratios))
            ),
            "wall_geometric_mean_ratio": float(
                math.exp(
                    sum(math.log(x) for x in case_wall_ratios) / len(case_wall_ratios)
                )
            ),
            "per_case_dedisp_ratio": dict(zip(primary, map(float, case_ratios))),
            "per_case_wall_ratio": dict(zip(primary, map(float, case_wall_ratios))),
        }
    return score


def _probe_dmt_events(gpu, cp, modules, case, report, variants, rng, stage):
    _resident_case(cp, case)
    try:
        for variant in rng.permutation(variants):
            variant = str(variant)
            _, wall, profile, dmt, effective, guard = run_process(
                gpu, cp, modules, variant, case, profile=False, measure_dmt=True
            )
            report["timings"].append(
                _record_timing(
                    stage, case, variant, wall, profile, dmt, effective, guard
                )
            )
    finally:
        _release_case(cp, case)


def _dmt_event_scores(report):
    primary = list(report["primary_cases"])
    timings = report["timings"]
    score = {}
    for variant in VARIANTS:
        ratios = []
        for case in primary:
            values = [
                item["dmt_kernel_cuda_event_s"]
                for item in timings
                if item["case"] == case
                and item["stage"] == "primary_dmt_event_probe"
                and item["variant"] == variant
            ]
            baseline = [
                item["dmt_kernel_cuda_event_s"]
                for item in timings
                if item["case"] == case
                and item["stage"] == "primary_dmt_event_probe"
                and item["variant"] == "baseline"
            ]
            ratios.append(float(np.median(values)) / float(np.median(baseline)))
        score[variant] = {
            "dmt_event_geometric_mean_ratio": float(
                math.exp(sum(math.log(x) for x in ratios) / len(ratios))
            ),
            "per_case_dmt_event_ratio": dict(zip(primary, map(float, ratios))),
        }
    return score


def _final_confirm(gpu, cp, modules, case, winner, report, rng, rounds=5):
    report["final_cases"][case.name] = _case_record(case)
    _resident_case(cp, case)
    variants = ("baseline", winner)
    try:
        # Correctness also supplies one randomized warmup per finalist.
        baseline_products = None
        finalist_products = None
        for variant in rng.permutation(variants):
            variant = str(variant)
            products, wall, profile, dmt, effective, guard = run_process(
                gpu, cp, modules, variant, case, profile=False
            )
            if variant == "baseline":
                baseline_products = tuple(
                    np.array(value, copy=True) for value in products
                )
            else:
                finalist_products = products
            report["timings"].append(
                _record_timing(
                    "final_warmup", case, variant, wall, profile, dmt, effective, guard
                )
            )
        errors = compare_products(
            finalist_products, baseline_products, f"{case.name}/{winner}/final-warmup"
        )
        _record_correctness(report, "final_warmup", case, winner, errors)
        for round_index in range(rounds):
            for variant in rng.permutation(variants):
                variant = str(variant)
                products, wall, profile, dmt, effective, guard = run_process(
                    gpu, cp, modules, variant, case, profile=False
                )
                errors = compare_products(
                    products,
                    baseline_products,
                    f"{case.name}/{variant}/final{round_index}",
                )
                _record_correctness(
                    report, f"final_round_{round_index + 1}", case, variant, errors
                )
                report["timings"].append(
                    _record_timing(
                        f"final_round_{round_index + 1}",
                        case,
                        variant,
                        wall,
                        profile,
                        dmt,
                        effective,
                        guard,
                    )
                )
            print(f"confirmed {case.name} round {round_index + 1}/{rounds}", flush=True)
    finally:
        _release_case(cp, case)


def _float_final(gpu, cp, modules, case, winner, report, rng, rounds=3):
    report["float_final_cases"][case.name] = _case_record(case)
    _resident_case(cp, case)
    try:
        baseline, wall, profile, _, effective, guard = run_process(
            gpu, cp, modules, "baseline", case, profile=False
        )
        baseline_error = compare_products(baseline, baseline, f"{case.name}/baseline")
        _record_correctness(report, "float_final", case, "baseline", baseline_error)
        report["timings"].append(
            _record_timing(
                "float_correctness",
                case,
                "baseline",
                wall,
                profile,
                None,
                effective,
                guard,
            )
        )
        products, wall, profile, _, effective, guard = run_process(
            gpu, cp, modules, winner, case, profile=False
        )
        errors = compare_products(products, baseline, f"{case.name}/{winner}")
        _record_correctness(report, "float_final", case, winner, errors)
        report["timings"].append(
            _record_timing(
                "float_correctness", case, winner, wall, profile, None, effective, guard
            )
        )
        for round_index in range(rounds):
            for variant in rng.permutation(("baseline", winner)):
                variant = str(variant)
                products, wall, profile, dmt, effective, guard = run_process(
                    gpu, cp, modules, variant, case, profile=False
                )
                errors = compare_products(
                    products, baseline, f"{case.name}/{variant}/float{round_index}"
                )
                _record_correctness(
                    report,
                    f"float_final_round_{round_index + 1}",
                    case,
                    variant,
                    errors,
                )
                report["timings"].append(
                    _record_timing(
                        f"float_final_round_{round_index + 1}",
                        case,
                        variant,
                        wall,
                        profile,
                        dmt,
                        effective,
                        guard,
                    )
                )
            print(
                f"float-final {case.name} round {round_index + 1}/{rounds}", flush=True
            )
        _release_case(cp, case)
        _probe_dmt_events(
            gpu,
            cp,
            modules,
            case,
            report,
            ("baseline", winner),
            rng,
            "float_final_dmt_event_probe",
        )
    finally:
        _release_case(cp, case)


def _summarize_final(report, stages, variants):
    summary = {}
    for case in stages:
        summary[case] = {}
        for variant in variants:
            samples = [
                item
                for item in report["timings"]
                if item["case"] == case
                and item["variant"] == variant
                and item["stage"].startswith("final_round_")
            ]
            if not samples:
                continue
            summary[case][variant] = {
                "wall": common.summarize([item["wall_s"] for item in samples]),
                "dmt_kernel_cuda_event": common.summarize(
                    [
                        item["dmt_kernel_cuda_event_s"]
                        for item in report["timings"]
                        if item["case"] == case
                        and item["variant"] == variant
                        and item["stage"] == "primary_dmt_event_probe"
                    ]
                ),
                "profile_event_medians_s": {
                    name: statistics.median(
                        item["profile_event_s"][name]
                        for item in samples
                        if name in item["profile_event_s"]
                    )
                    for name in set().union(
                        *(item["profile_event_s"].keys() for item in samples)
                    )
                },
            }
    return summary


def _summarize_float_final(report, cases, variants):
    summary = {}
    for case in cases:
        summary[case] = {}
        for variant in variants:
            samples = [
                item
                for item in report["timings"]
                if item["case"] == case
                and item["variant"] == variant
                and item["stage"].startswith("float_final_round_")
            ]
            if samples:
                summary[case][variant] = {
                    "wall": common.summarize([item["wall_s"] for item in samples]),
                    "dmt_kernel_cuda_event": common.summarize(
                        [
                            item["dmt_kernel_cuda_event_s"]
                            for item in report["timings"]
                            if item["case"] == case
                            and item["variant"] == variant
                            and item["stage"] == "float_final_dmt_event_probe"
                        ]
                    ),
                }
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/benchmark_dmt_staging_results.json")
    args = parser.parse_args()

    import cupy as cp

    from your.utils import gpu

    baseline_source = KERNEL_PATH.read_text()
    source_hash = hashlib.sha256(baseline_source.encode()).hexdigest()
    if source_hash != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            "unexpected production kernel source hash "
            f"{source_hash}; refusing to benchmark"
        )
    sources = source_variants(baseline_source)
    props = cp.cuda.runtime.getDeviceProperties(0)
    cc = int(props["major"]) * 10 + int(props["minor"])
    modules = {
        variant: common.make_module(cp, source) for variant, source in sources.items()
    }
    for variant in VARIANTS:
        print(f"compiled {variant}", flush=True)

    report = {
        "gpu": props["name"].decode(),
        "compute_capability": f"{cc // 10}.{cc % 10}",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "source_sha256": source_hash,
        "expected_baseline_sha256": EXPECTED_BASELINE_SHA256,
        "source_variant_sha256": {
            name: hashlib.sha256(source.encode()).hexdigest()
            for name, source in sources.items()
        },
        "configuration": {
            "time_tile": TIME_TILE,
            "shared_caps_floats": SHARED_CAPS,
            "screen_rounds": 3,
            "final_u8_rounds": 5,
            "float_final_rounds": 3,
            "primary_dm_values": [80.0, 700.0],
            "primary_channels": 4096,
            "primary_dtype": "uint8",
            "primary_time_size": 256,
            "primary_widths": {"2": "bin1", "256": "bin128"},
            "primary_dmsteps": 256,
            "resident_input": True,
            "endpoint_condition": (
                "actual delay rows nondecreasing; otherwise baseline scan"
            ),
            "spread_aware_capacity": (
                "min(existing required capacity, per-variant cap, device limit)"
            ),
        },
        "correctness_cases": {},
        "correctness": {},
        "primary_cases": {},
        "primary_correctness": {},
        "final_cases": {},
        "float_final_cases": {},
        "timings": [],
    }
    output_path = Path(args.output)

    def save():
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    old_kernel_functions = gpu._kernel_functions
    rng = np.random.default_rng(20261017)
    try:
        _correctness_suite(gpu, cp, modules, report)
        save()

        primary_specs = [
            (80.0, 2),
            (80.0, 256),
            (700.0, 2),
            (700.0, 256),
        ]
        for dm, width in primary_specs:
            case = _make_large_case(dm, width, np.uint8)
            _measure_suite_case(gpu, cp, modules, case, report, rng)
            save()
            del case

        # Measure DMT kernel launches in a separate pass so wall/profile runs
        # do not include per-launch event-wrapper overhead.
        for dm, width in primary_specs:
            case = _make_large_case(dm, width, np.uint8)
            _probe_dmt_events(
                gpu,
                cp,
                modules,
                case,
                report,
                VARIANTS,
                rng,
                "primary_dmt_event_probe",
            )
            save()
            del case

        scores = _screen_scores(report)
        dmt_event_scores = _dmt_event_scores(report)
        nonbaseline = [name for name in VARIANTS if name != "baseline"]
        best_nonbaseline = min(
            nonbaseline,
            key=lambda name: dmt_event_scores[name]["dmt_event_geometric_mean_ratio"],
        )
        winner = min(
            VARIANTS,
            key=lambda name: dmt_event_scores[name]["dmt_event_geometric_mean_ratio"],
        )
        finalist = winner if winner != "baseline" else best_nonbaseline
        report["screen_scores"] = scores
        report["dmt_event_scores"] = dmt_event_scores
        report["screen_winner"] = winner
        report["finalist"] = finalist
        save()
        print("screen scores:", json.dumps(scores, indent=2), flush=True)
        print("DMT event scores:", json.dumps(dmt_event_scores, indent=2), flush=True)

        for dm, width in primary_specs:
            case = _make_large_case(dm, width, np.uint8)
            _final_confirm(gpu, cp, modules, case, finalist, report, rng, rounds=5)
            save()
            del case

        # Input-float confirmation covers real channel count, nonfinite values,
        # kill masks, full source coverage, and the widest production tile.
        for dtype in (np.float16, np.float32):
            case = _make_large_case(80.0, 256, dtype)
            _float_final(gpu, cp, modules, case, finalist, report, rng, rounds=3)
            save()
            del case

        report["final_summary"] = _summarize_final(
            report, list(report["final_cases"]), ("baseline", finalist)
        )
        report["float_final_summary"] = _summarize_float_final(
            report, list(report["float_final_cases"]), ("baseline", finalist)
        )
        report["notes"] = [
            (
                "Dedicated DMT event timing sums CUDA-event durations around each "
                "dmt_tiled launch in separate resident process passes."
            ),
            (
                "Process wall timing includes output transfer and CPU product "
                "finishing while source arrays are resident. Screen rounds enable "
                "built-in stage profiling; final confirmation and float wall rounds "
                "use profile=False, with DMT CUDA-event probes recorded separately."
            ),
            (
                "Endpoint bounds require nondecreasing delay rows. gpu_dmt(dms=...) "
                "accepts arbitrary order, so unordered rows use the baseline full scan."
            ),
            (
                "The spread-aware kernel evaluates each channel's full delay span "
                "uniformly across the block; all threads take the same shared/direct "
                "branch and barriers remain block-uniform on partial time tiles."
            ),
            (
                "No production source file was modified; candidate source variants "
                "are generated in memory by this new benchmark check."
            ),
        ]
        save()
        print(f"JSON: {output_path}", flush=True)
        print(
            json.dumps(
                {
                    "screen_winner": winner,
                    "finalist": finalist,
                    "screen_scores": scores,
                    "dmt_event_scores": dmt_event_scores,
                    "final_summary": report["final_summary"],
                    "float_final_summary": report["float_final_summary"],
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        gpu._kernel_functions = old_kernel_functions
        try:
            cp.cuda.Device().synchronize()
        finally:
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
