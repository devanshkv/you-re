#!/usr/bin/env python3
"""Screen bounded DMT CUDA tile and channel-group geometries.

Compile without creating a CUDA context:
    CUDA_PATH=/usr/local/cuda-12.9 /tmp/your-cupy-env/bin/python \
        checks/benchmark_dmt_geometry.py --compile-only

Run only after the GPU is reserved for this benchmark:
    CUDA_PATH=/usr/local/cuda-12.9 /tmp/your-cupy-env/bin/python \
        checks/benchmark_dmt_geometry.py

The production .cu file is read only. Generated sources and compile artifacts
live in memory or temporary directories; the JSON report defaults to /tmp.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "your/utils/gpu_kernels.cu"
EXPECTED_BASELINE_SHA256 = (
    "ef0c47434f622a613e7c281483d7ff65e248dddfa031558e8ffb31e6eb03cd49"
)
DEFAULT_OUTPUT = Path("/tmp/benchmark_dmt_geometry_results.json")


@dataclass(frozen=True)
class Geometry:
    name: str
    time_tile: int
    dm_tile: int
    channel_tile: int
    ept: int = 1


GEOMETRIES = (
    Geometry("base_t128_d8_c32", 128, 8, 32),
    Geometry("t512_d4_c32", 512, 4, 32),
    Geometry("t512_d8_c32", 512, 8, 32),
    Geometry("t1024_d4_c32", 1024, 4, 32),
    Geometry("t1024_d8_c32", 1024, 8, 32),
    Geometry("t128_d8_c128", 128, 8, 128),
    Geometry("t128_d8_c256", 128, 8, 256),
    Geometry("t512_d4_c128", 512, 4, 128),
    Geometry("t512_d8_c128", 512, 8, 128),
    Geometry("t512_d4_c256", 512, 4, 256),
    Geometry("t512_d8_c256", 512, 8, 256),
    Geometry("t512_d4_ept2_c32", 512, 4, 32, ept=2),
)
BASELINE = GEOMETRIES[0]


# The EPT=2 body maps each thread to two coalesced time samples. Other
# geometries preserve the source body and change only its two tile constants.
_EPT_REWRITES = (
    (
        (
            "int lane = threadIdx.x;\n"
            "    int time_base = blockIdx.x * TIME_TILE;\n"
            "    int time = time_base + lane;"
        ),
        "int lane = threadIdx.x;\n    int time_base = blockIdx.x * TIME_TILE;",
    ),
    ("float sums[DM_TILE] = {0.0f};", "float sums[EPT][DM_TILE] = {};"),
    (
        "for (int i = lane; i < count; i += TIME_TILE)",
        "for (int i = lane; i < count; i += blockDim.x)",
    ),
    (
        (
            "            if (time < native_count && enabled[local_channel]) {\n"
            "                for (int d = 0; d < dm_count; ++d) {\n"
            "                    long long delay = delays[(long long)(dm_base +"
            " d) * nchans + global_channel];\n"
            "                    sums[d] += tile[lane + (int)(delay - minimum)];\n"
            "                }\n"
            "            }"
        ),
        (
            "            for (int e = 0; e < EPT; ++e) {\n"
            "                int time = time_base + lane + e * blockDim.x;\n"
            "                if (time < native_count && enabled[local_channel]) {\n"
            "                    for (int d = 0; d < dm_count; ++d) {\n"
            "                        long long delay = delays[(long"
            " long)(dm_base + d) * nchans + global_channel];\n"
            "                        sums[e][d] += tile[lane + e * blockDim.x +"
            " (int)(delay - minimum)];\n"
            "                    }\n"
            "                }\n"
            "            }"
        ),
    ),
    (
        (
            "    } else if (time < native_count) {\n"
            "        for (int local_channel = 0; local_channel < group_channels;"
            " ++local_channel) {\n"
            "            int global_channel = channel_start + local_channel;\n"
            "            for (int d = 0; d < dm_count; ++d) {\n"
            "                long long delay = delays[(long long)(dm_base + d) *"
            " nchans + global_channel];\n"
            "                sums[d] += sample_at(\n"
            "                    source, medians, enabled, local_channel,"
            " output_start + time + delay,\n"
            "                    source_start, source_count, valid_start,"
            " valid_stop, nspectra,\n"
            "                    input_scale\n"
            "                );\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "\n"
            "    if (time < native_count) {\n"
            "        for (int d = 0; d < dm_count; ++d) {\n"
            "            long long index = (long long)(dm_base + d) *"
            " native_count + time;\n"
            "            if (!first_group) sums[d] += output[index];\n"
            "            output[index] = sums[d];\n"
            "        }\n"
            "    }"
        ),
        (
            "    } else {\n"
            "        for (int e = 0; e < EPT; ++e) {\n"
            "            int time = time_base + lane + e * blockDim.x;\n"
            "            if (time < native_count) {\n"
            "                for (int local_channel = 0; local_channel <"
            " group_channels; ++local_channel) {\n"
            "                    int global_channel = channel_start +"
            " local_channel;\n"
            "                    for (int d = 0; d < dm_count; ++d) {\n"
            "                        long long delay = delays[(long"
            " long)(dm_base + d) * nchans + global_channel];\n"
            "                        sums[e][d] += sample_at(\n"
            "                            source, medians, enabled,"
            " local_channel, output_start + time + delay,\n"
            "                            source_start, source_count,"
            " valid_start, valid_stop, nspectra,\n"
            "                            input_scale\n"
            "                        );\n"
            "                    }\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "\n"
            "    for (int e = 0; e < EPT; ++e) {\n"
            "        int time = time_base + lane + e * blockDim.x;\n"
            "        if (time < native_count) {\n"
            "            for (int d = 0; d < dm_count; ++d) {\n"
            "                long long index = (long long)(dm_base + d) *"
            " native_count + time;\n"
            "                if (!first_group) sums[e][d] += output[index];\n"
            "                output[index] = sums[e][d];\n"
            "            }\n"
            "        }\n"
            "    }"
        ),
    ),
)


def geometry_source(baseline, geometry):
    """Build a geometry from any compatible baseline or staging-winner source."""
    code, count_time = re.subn(
        r"static const int TIME_TILE = \d+;",
        f"static const int TIME_TILE = {geometry.time_tile};",
        baseline,
        count=1,
    )
    code, count_dm = re.subn(
        r"static const int DM_TILE = \d+;",
        f"static const int DM_TILE = {geometry.dm_tile};",
        code,
        count=1,
    )
    if count_time != 1 or count_dm != 1:
        raise ValueError("source must define TIME_TILE and DM_TILE exactly once")
    if geometry.ept == 1:
        return code
    if geometry.ept != 2:
        raise ValueError(f"unsupported EPT {geometry.ept}")

    marker = f"static const int DM_TILE = {geometry.dm_tile};"
    code, count = (
        code.replace(marker, marker + "\nstatic const int EPT = 2;"),
        code.count(marker),
    )
    if count != 1:
        raise ValueError("could not insert EPT constant")
    for old, new in _EPT_REWRITES:
        count = code.count(old)
        if count != 1:
            raise ValueError(
                f"EPT source pattern should occur once, found {count}: {old[:72]!r}"
            )
        code = code.replace(old, new, 1)
    return code


def compile_only(source, geometries, output_dir=None):
    """Compile with nvcc/PTX only; this does not create a CUDA context."""
    cuda_path = os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_path) / "bin/nvcc") if cuda_path else shutil.which("nvcc")
    if not nvcc or (cuda_path and not Path(nvcc).is_file()):
        raise RuntimeError("nvcc not found; set CUDA_PATH or add nvcc to PATH")
    temp_context = tempfile.TemporaryDirectory(prefix="dmt-geometry-ptx-")
    workdir = Path(output_dir) if output_dir else Path(temp_context.name)
    workdir.mkdir(parents=True, exist_ok=True)
    include = Path(cuda_path) / "include" if cuda_path else None
    results = {}
    try:
        for geometry in geometries:
            source_path = workdir / f"{geometry.name}.cu"
            ptx_path = workdir / f"{geometry.name}.ptx"
            source_path.write_text(geometry_source(source, geometry))
            command = [
                nvcc,
                "--std=c++11",
                "--ptx",
                "--gpu-architecture=compute_70",
                str(source_path),
                "-o",
                str(ptx_path),
            ]
            if include and include.is_dir():
                command.extend(("-I", str(include)))
            subprocess.run(command, check=True, capture_output=True, text=True)
            results[geometry.name] = {
                "ptx_bytes": ptx_path.stat().st_size,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
            ptx_bytes = results[geometry.name]["ptx_bytes"]
            print(
                f"compiled {geometry.name}: {ptx_bytes} PTX bytes",
                flush=True,
            )
        return results
    finally:
        temp_context.cleanup()


def make_case(spec):
    sys.path.insert(0, str(ROOT))
    from checks.benchmark_ldg import make_case as make_ldg_case

    return make_ldg_case(
        spec["name"],
        nchans=spec["nchans"],
        rows=None,
        dtype=np.dtype(spec["dtype"]),
        time_size=256,
        width=spec["width"],
        dmsteps=256,
        tsamp=1e-4,
        tcand=0.0128 if spec["width"] == 2 else 1.6384,
        dm=spec["dm"],
    )


def case_specs(dm):
    return [
        {
            "name": "4096u8_dm80_bin1",
            "nchans": 4096,
            "dtype": "uint8",
            "width": 2,
            "dm": 80.0,
        },
        {
            "name": "4096u8_dm80_bin128",
            "nchans": 4096,
            "dtype": "uint8",
            "width": 256,
            "dm": 80.0,
        },
        {
            "name": f"4096u8_dm{dm:g}_bin1",
            "nchans": 4096,
            "dtype": "uint8",
            "width": 2,
            "dm": dm,
        },
        {
            "name": f"4096u8_dm{dm:g}_bin128",
            "nchans": 4096,
            "dtype": "uint8",
            "width": 256,
            "dm": dm,
        },
    ]


def float_case_specs():
    return [
        {
            "name": "4096f16_dm80_bin128",
            "nchans": 4096,
            "dtype": "float16",
            "width": 256,
            "dm": 80.0,
        },
        {
            "name": "4096f32_dm80_bin128",
            "nchans": 4096,
            "dtype": "float32",
            "width": 256,
            "dm": 80.0,
        },
    ]


def set_geometry(gpu, geometry, modules, original_compute=None):
    gpu._TIME_TILE = geometry.time_tile
    gpu._DM_TILE = geometry.dm_tile
    gpu._CHANNEL_TILE = geometry.channel_tile
    gpu._EPT = geometry.ept
    gpu._kernel_functions = lambda _cp: modules[geometry.name]
    if original_compute is not None:
        install_ept_launch(gpu, original_compute, geometry.ept)


def install_ept_launch(gpu, original_compute, ept):
    gpu._compute_binned_products = original_compute
    if ept == 1:
        return
    code = inspect.getsource(original_compute)
    old = "(_TIME_TILE,),"
    if code.count(old) != 1:
        raise ValueError("could not patch the DMT thread count for EPT")
    code = code.replace(old, "(_TIME_TILE // _EPT,),", 1)
    exec(compile(code, "<dmt-geometry-launch>", "exec"), gpu.__dict__)


def median(values):
    return float(statistics.median(float(v) for v in values))


def is_launch_resource_error(error):
    message = str(error)
    return (
        "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES" in message
        or "too many resources requested for launch" in message
    )


def compare_products(actual, expected, context, input_dtype):
    acceptance_tolerance = 3e-3
    differences = {}
    for label, got, want in zip(("FT", "DMT"), actual, expected):
        if want is None:
            continue
        if got is None or got.shape != want.shape:
            raise AssertionError(f"{context} {label} shape mismatch")
        if not np.all(np.isfinite(got)) or not np.all(np.isfinite(want)):
            raise AssertionError(f"{context} {label} contains a non-finite value")
        delta = np.abs(got.astype(np.float64) - want.astype(np.float64))
        max_abs = float(delta.max(initial=0.0))
        max_rel = float((delta / np.maximum(np.abs(want), 1e-6)).max(initial=0.0))
        np.testing.assert_allclose(
            got,
            want,
            rtol=acceptance_tolerance,
            atol=acceptance_tolerance,
            err_msg=(
                f"{context} {label} differs (max_abs={max_abs:g}, max_rel={max_rel:g})"
            ),
        )
        differences[label] = {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "within_2e-5_rtol_atol": bool(np.allclose(got, want, rtol=2e-5, atol=2e-5)),
        }
    return differences


def raw_dmt(gpu, cp, case, *, profile=None):
    plan = case.plan
    resident = case.resident
    if resident is None:
        resident = cp.ascontiguousarray(cp.asarray(np.ascontiguousarray(case.data.T)))
    return gpu._compute_binned_products(
        cp=cp,
        data=None,
        resident=resident,
        resident_start=0,
        source_start=0,
        valid_start=plan.read_start,
        valid_stop=plan.read_stop,
        median_start=plan.read_start,
        median_stop=plan.read_stop,
        nspectra=plan.nspectra,
        nchans=plan.nchans,
        delays=plan.delays,
        ft_delays=None,
        output_start=plan.output_start,
        native_count=plan.native_count,
        time_bin=plan.time_bin,
        time_size=plan.time_size,
        enabled=gpu._wrapper_enabled(case.cand, plan.nchans),
        device=0,
        stream=None,
        profile=profile,
        compute_ft=False,
        compute_dmt=True,
        input_scale=1.0,
    )


def screen_case(gpu, cp, modules, case, geometries, repeats, seed, original_compute):
    rng = np.random.default_rng(seed)
    set_geometry(gpu, BASELINE, modules, original_compute)
    base_products = raw_dmt(gpu, cp, case)
    reference = (None, np.array(base_products[1], copy=True))
    compare_products(
        base_products,
        reference,
        f"{case.cand.data.dtype}/baseline",
        case.cand.data.dtype,
    )

    result = {
        geometry.name: {"kernel_s": [], "wall_s": [], "profile": {}}
        for geometry in geometries
    }
    order = list(geometries)
    rng.shuffle(order)
    for geometry in order:
        set_geometry(gpu, geometry, modules, original_compute)
        # Compile/warmup launches are outside measured rounds.
        try:
            warmup = raw_dmt(gpu, cp, case)
        except Exception as error:
            if not is_launch_resource_error(error):
                raise
            result[geometry.name]["unsupported"] = str(error)
            print(
                "resource-limited raw DMT "
                f"{case.cand.data.dtype}/{geometry.name}: {error}",
                flush=True,
            )
            continue
        compare_products(
            warmup,
            reference,
            f"{case.cand.data.dtype}/{geometry.name}/warmup",
            case.cand.data.dtype,
        )
    for repeat in range(repeats):
        rng.shuffle(order)
        for geometry in order:
            entry = result[geometry.name]
            if "unsupported" in entry:
                continue
            set_geometry(gpu, geometry, modules, original_compute)
            profile = {}
            started = perf_counter()
            try:
                products = raw_dmt(gpu, cp, case, profile=profile)
            except Exception as error:
                if not is_launch_resource_error(error):
                    raise
                entry["unsupported"] = str(error)
                print(
                    "resource-limited raw DMT "
                    f"{case.cand.data.dtype}/{geometry.name}: {error}",
                    flush=True,
                )
                continue
            wall_s = perf_counter() - started
            differences = compare_products(
                products,
                reference,
                f"{case.cand.data.dtype}/{geometry.name}/screen{repeat}",
                case.cand.data.dtype,
            )
            entry = result[geometry.name]
            entry["wall_s"].append(wall_s)
            entry["kernel_s"].append(float(profile.get("dedisp_s", math.nan)))
            for label, values in profile.items():
                entry["profile"].setdefault(label, []).append(float(values))
            entry["max_differences"] = differences
        print(
            f"completed raw DMT screen {case.cand.data.dtype} {repeat + 1}/{repeats}",
            flush=True,
        )
    return result


def summarize_samples(samples):
    values = [float(value) for value in samples]
    return {
        "median_s": median(values),
        "min_s": min(values),
        "max_s": max(values),
        "samples_s": values,
    }


def choose_finalists(screen, geometries):
    """Keep a measured leader per workload plus a near-tie EPT check."""
    baseline = next(geometry for geometry in geometries if geometry == BASELINE)
    winners = {}
    finalists_by_case = {}
    ratios = {}
    ept_candidates = [geometry for geometry in geometries if geometry.ept > 1]
    for case_name, per_case in screen.items():
        valid = [
            geometry
            for geometry in geometries
            if len(per_case[geometry.name].get("kernel_s", []))
            and "unsupported" not in per_case[geometry.name]
        ]
        if baseline not in valid:
            raise RuntimeError(f"baseline could not launch for {case_name}")
        times = {
            geometry.name: median(per_case[geometry.name]["kernel_s"])
            for geometry in valid
        }
        baseline_time = times[baseline.name]
        ratios[case_name] = {
            name: value / baseline_time for name, value in times.items()
        }
        winner = min(valid, key=lambda geometry: (times[geometry.name], geometry.name))
        winners[case_name] = winner.name
        finalists = [baseline]
        if winner != baseline:
            finalists.append(winner)
        valid_ept_candidates = [
            geometry for geometry in ept_candidates if geometry in valid
        ]
        if valid_ept_candidates:
            ept = min(
                valid_ept_candidates,
                key=lambda geometry: (times[geometry.name], geometry.name),
            )
            if ept != winner and times[ept.name] <= times[winner.name] * 1.05:
                finalists.append(ept)
        finalists_by_case[case_name] = finalists
    return winners, finalists_by_case, ratios


def choose_float_group_geometries(screen, geometries, case_name):
    """Pick one promising time/DM tile, then compare groups 32/128/256 on floats."""
    per_case = screen[case_name]
    pairs = {}
    for geometry in geometries:
        if geometry.ept == 1 and geometry.channel_tile in (32, 128, 256):
            pairs.setdefault((geometry.time_tile, geometry.dm_tile), {})[
                geometry.channel_tile
            ] = geometry
    complete = {
        key: groups
        for key, groups in pairs.items()
        if {32, 128, 256} <= groups.keys()
        and all(
            len(per_case[groups[channel_tile].name].get("kernel_s", [])) > 0
            and "unsupported" not in per_case[groups[channel_tile].name]
            for channel_tile in (32, 128, 256)
        )
    }
    if not complete:
        raise AssertionError("no matched C32/C128/C256 geometry is available")
    time_tile, dm_tile = min(
        complete,
        key=lambda key: math.sqrt(
            median(per_case[complete[key][128].name]["kernel_s"])
            * median(per_case[complete[key][256].name]["kernel_s"])
        ),
    )
    groups = complete[(time_tile, dm_tile)]
    return [groups[32], groups[128], groups[256]], {
        "time_tile": time_tile,
        "dm_tile": dm_tile,
    }


def full_process(
    gpu, cp, modules, case, geometry, original_compute, *, instrument=False
):
    set_geometry(gpu, geometry, modules, original_compute)
    profile = {} if instrument else None
    started = perf_counter()
    products = gpu.process_gpu(
        case.cand,
        case.plan,
        resident=case.resident,
        resident_start=0,
        device=0,
        profile=profile,
    )
    return products, perf_counter() - started, profile or {}


def benchmark_full_case(
    gpu,
    cp,
    modules,
    case,
    finalists,
    repeats,
    seed,
    original_compute,
    *,
    reference_geometry=BASELINE,
):
    rng = np.random.default_rng(seed)
    reference, _, _ = full_process(
        gpu, cp, modules, case, reference_geometry, original_compute
    )
    for geometry in finalists:
        products, _, _ = full_process(
            gpu, cp, modules, case, geometry, original_compute
        )
        compare_products(
            products,
            reference,
            f"{case.cand.data.dtype}/{geometry.name}/warmup",
            case.cand.data.dtype,
        )

    result = {geometry.name: {"wall_s": [], "profile": {}} for geometry in finalists}
    order = list(finalists)
    for repeat in range(repeats):
        rng.shuffle(order)
        for geometry in order:
            products, wall_s, _ = full_process(
                gpu, cp, modules, case, geometry, original_compute
            )
            differences = compare_products(
                products,
                reference,
                f"{case.cand.data.dtype}/{geometry.name}/round{repeat}",
                case.cand.data.dtype,
            )
            entry = result[geometry.name]
            entry["wall_s"].append(wall_s)
            entry["max_differences"] = differences
        print(
            f"completed full process {case.cand.data.dtype} {repeat + 1}/{repeats}",
            flush=True,
        )

    for geometry in finalists:
        entry = result[geometry.name]
        products, _, profile = full_process(
            gpu, cp, modules, case, geometry, original_compute, instrument=True
        )
        differences = compare_products(
            products,
            reference,
            f"{case.cand.data.dtype}/{geometry.name}/profile",
            case.cand.data.dtype,
        )
        entry["max_differences"] = differences
        entry["wall"] = summarize_samples(entry.pop("wall_s"))
        entry["profile"] = {label: float(seconds) for label, seconds in profile.items()}
    return result


def dtype_tail_smoke(gpu, cp, modules, original_compute):
    """Check every input dtype with channel, time, and DM tails on each variant."""
    from checks.benchmark_ldg import make_case as make_ldg_case

    results = {}
    for dtype in (
        np.uint8,
        np.uint16,
        np.int16,
        np.int32,
        np.float16,
        np.float32,
        np.float64,
    ):
        case = make_ldg_case(
            f"tail_{np.dtype(dtype).name}",
            nchans=37,
            rows=512,
            dtype=dtype,
            time_size=129,
            width=2,
            dmsteps=7,
            tsamp=1e-4,
            tcand=0.0064,
            dm=80.0,
        )
        case.resident = cp.ascontiguousarray(
            cp.asarray(np.ascontiguousarray(case.data.T))
        )
        cp.cuda.get_current_stream().synchronize()
        reference, _, _ = full_process(
            gpu, cp, modules, case, BASELINE, original_compute
        )
        passed = []
        unsupported = {}
        for geometry in GEOMETRIES:
            try:
                products, _, _ = full_process(
                    gpu, cp, modules, case, geometry, original_compute
                )
            except Exception as error:
                if not is_launch_resource_error(error):
                    raise
                unsupported[geometry.name] = str(error)
                print(
                    "resource-limited dtype tail "
                    f"{np.dtype(dtype).name}/{geometry.name}: {error}",
                    flush=True,
                )
                continue
            compare_products(
                products,
                reference,
                f"tail/{np.dtype(dtype).name}/{geometry.name}",
                dtype,
            )
            passed.append(geometry.name)
        results[np.dtype(dtype).name] = {
            "shape": list(case.data.shape),
            "native_count": int(case.plan.native_count),
            "dmsteps": int(case.plan.dmsteps),
            "plan_padding": [int(case.plan.pad_before), int(case.plan.pad_after)],
            "passed_variants": passed,
            "unsupported_variants": unsupported,
        }
        del case
        cp.get_default_memory_pool().free_all_blocks()
    return results


def runtime_benchmark(source, source_hash, args):
    sys.path.insert(0, str(ROOT))
    import cupy as cp

    from checks.benchmark_ldg import dmt_shared_details, make_module
    from your.utils import gpu

    props = cp.cuda.runtime.getDeviceProperties(0)
    cc = int(props["major"]) * 10 + int(props["minor"])
    modules = {}
    variant_hashes = {}
    for geometry in GEOMETRIES:
        code = geometry_source(source, geometry)
        modules[geometry.name] = make_module(cp, code)
        variant_hashes[geometry.name] = hashlib.sha256(code.encode()).hexdigest()
        print(f"compiled {geometry.name}", flush=True)

    old_state = {
        name: getattr(gpu, name)
        for name in ("_TIME_TILE", "_DM_TILE", "_CHANNEL_TILE", "_kernel_functions")
    }
    missing = object()
    old_ept = getattr(gpu, "_EPT", missing)
    original_compute = gpu._compute_binned_products
    gpu._EPT = 1
    screen = {}
    storage = {}
    try:
        dtype_correctness = dtype_tail_smoke(gpu, cp, modules, original_compute)
        for idx, spec in enumerate(case_specs(args.dm)):
            case = make_case(spec)
            case.resident = cp.ascontiguousarray(
                cp.asarray(np.ascontiguousarray(case.data.T))
            )
            cp.cuda.get_current_stream().synchronize()
            set_geometry(gpu, BASELINE, modules, original_compute)
            details = dmt_shared_details(gpu, cp, case.plan)
            storage[spec["name"]] = {
                "shape": list(case.data.shape),
                "dtype": str(case.data.dtype),
                "source_bytes": int(case.data.nbytes),
                "time_bin": int(case.plan.time_bin),
                "native_count": int(case.plan.native_count),
                "rows": int(case.data.shape[0]),
                "pad_before": int(case.plan.pad_before),
                "pad_after": int(case.plan.pad_after),
                "dmt_shared_baseline": details,
            }
            if case.plan.pad_before or case.plan.pad_after:
                raise AssertionError(f"{spec['name']} is not full-coverage")
            print(
                f"{spec['name']}: {case.data.shape}, "
                f"{case.data.nbytes / 2**30:.2f} GiB, "
                f"shared capacity {details['capacities_floats']} floats",
                flush=True,
            )
            screen[spec["name"]] = screen_case(
                gpu,
                cp,
                modules,
                case,
                GEOMETRIES,
                args.screen_repeats,
                20261017 + idx,
                original_compute,
            )
            case.resident = None
            del case
            cp.get_default_memory_pool().free_all_blocks()

        winners, finalists_by_case, ratios = choose_finalists(screen, GEOMETRIES)
        for case_name, winner in winners.items():
            print(f"{case_name} leader: {winner}", flush=True)
        full_results = {}
        full_setup = {}
        float_group_geometries, float_group_geometry = choose_float_group_geometries(
            screen, GEOMETRIES, "4096u8_dm80_bin128"
        )
        float_groups = [
            BASELINE,
            *[geometry for geometry in float_group_geometries if geometry != BASELINE],
        ]
        print(
            "float group comparison geometry: "
            f"T{float_group_geometry['time_tile']}/DM{float_group_geometry['dm_tile']} "
            "at C32/C128/C256",
            flush=True,
        )
        full_specs = [*case_specs(args.dm)[:4], *float_case_specs()]
        for idx, spec in enumerate(full_specs):
            case = make_case(spec)
            device_free_before, device_total = cp.cuda.runtime.memGetInfo()
            case.resident = cp.ascontiguousarray(
                cp.asarray(np.ascontiguousarray(case.data.T))
            )
            cp.cuda.get_current_stream().synchronize()
            device_free_after, _ = cp.cuda.runtime.memGetInfo()
            is_float = np.dtype(spec["dtype"]).kind == "f"
            screen_key = (
                spec["name"]
                if spec["name"] in finalists_by_case
                else "4096u8_dm80_bin128"
            )
            finalists = float_groups if is_float else finalists_by_case[screen_key]
            memory = {
                "host_source_bytes": int(case.data.nbytes),
                "resident_source_bytes": int(case.resident.nbytes),
                "device_free_before_resident_bytes": int(device_free_before),
                "device_free_after_resident_bytes": int(device_free_after),
                "device_total_bytes": int(device_total),
            }
            if is_float:
                memory["dmt_shared_by_geometry"] = {}
                for geometry in float_groups:
                    set_geometry(gpu, geometry, modules, original_compute)
                    memory["dmt_shared_by_geometry"][geometry.name] = (
                        dmt_shared_details(gpu, cp, case.plan)
                    )
            full_results[spec["name"]] = benchmark_full_case(
                gpu,
                cp,
                modules,
                case,
                finalists,
                args.full_repeats,
                20262017 + idx,
                original_compute,
                reference_geometry=BASELINE,
            )
            full_setup[spec["name"]] = {
                "timed_geometries": [geometry.name for geometry in finalists],
                "timing_baseline": (
                    float_groups[0].name if is_float else BASELINE.name
                ),
                "correctness_reference": BASELINE.name,
                "selection_screen_case": screen_key,
                "memory": memory,
            }
            del case
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        gpu._compute_binned_products = original_compute
        for name, value in old_state.items():
            setattr(gpu, name, value)
        if old_ept is missing:
            del gpu._EPT
        else:
            gpu._EPT = old_ept

    report = {
        "gpu": props["name"].decode(),
        "compute_capability": f"{cc // 10}.{cc % 10}",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "source_sha256": source_hash,
        "expected_baseline_sha256": EXPECTED_BASELINE_SHA256,
        "geometries": [asdict(g) for g in GEOMETRIES],
        "variant_source_sha256": variant_hashes,
        "screen_cases": storage,
        "dtype_tail_correctness": dtype_correctness,
        "screen_kernel_speedup_ratio_to_baseline": ratios,
        "screen_case_winners": winners,
        "finalists_by_screen_case": {
            name: [geometry.name for geometry in finalists]
            for name, finalists in finalists_by_case.items()
        },
        "float_group_comparison_geometry": float_group_geometry,
        "full_validation_setup": full_setup,
        "raw_dmt_screen": screen,
        "full_process_repeats": args.full_repeats,
        "full_process": full_results,
        "full_wall_timing": (
            "unprofiled process_gpu wall with preuploaded resident input "
            "(production events retained; no added wrappers)"
        ),
        "tolerance": {
            "acceptance_rtol_atol": 3e-3,
            "stricter_fp32_comparison_reported_at": 2e-5,
        },
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for case_name, variants in full_results.items():
        print(f"\n{case_name}")
        timing_baseline = full_setup[case_name]["timing_baseline"]
        base = variants[timing_baseline]["wall"]["median_s"]
        for name, result in variants.items():
            print(
                f"{name:22s} wall={result['wall']['median_s'] * 1e3:8.2f} ms "
                f"({result['wall']['median_s'] / base:5.3f}x) "
                f"dmt={result['profile'].get('dedisp_s', math.nan) * 1e3:8.2f} ms"
            )
    print(f"JSON: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=KERNEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--compile-dir", type=Path)
    parser.add_argument("--dm", type=float, default=700.0)
    parser.add_argument("--screen-repeats", type=int, default=2)
    parser.add_argument("--full-repeats", type=int, default=3)
    args = parser.parse_args()
    if not np.isfinite(args.dm) or args.dm <= 0:
        parser.error("--dm must be finite and positive")
    if args.screen_repeats < 1:
        parser.error("--screen-repeats must be positive")
    if not 3 <= args.full_repeats <= 5:
        parser.error("--full-repeats must be from 3 through 5")

    source = args.source.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if (
        args.source.resolve() == KERNEL_PATH.resolve()
        and source_hash != EXPECTED_BASELINE_SHA256
    ):
        raise SystemExit(
            "production CUDA source does not match the expected baseline SHA256"
        )
    if args.compile_only:
        compiled = compile_only(source, GEOMETRIES, args.compile_dir)
        print(
            json.dumps({"source_sha256": source_hash, "compiled": compiled}, indent=2)
        )
        return
    runtime_benchmark(source, source_hash, args)


if __name__ == "__main__":
    main()
