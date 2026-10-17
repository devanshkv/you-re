"""Bounded GPU arithmetic for candidates whose complete input does not fit VRAM.

Raw samples are staged on disk once. Exact padding/RFI statistics stay on the
host on this exceptional path; dedispersion and time reductions remain on GPU.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from your.utils.candidate_contract import decoded_sample_dtype, parse_kill_mask
from your.utils.candidate_pipeline import finish_products
from your.utils.rfi import calc_N, savgol_filter, sk_rejection_mask


def _enabled(cand, nchans):
    return ~parse_kill_mask(getattr(cand, "kill_mask", [False]), nchans)


def _rfi_enabled(cand, plan, raw, medians, enabled, rows):
    """The existing SK/SG decisions, using bounded passes over logical samples."""
    if not getattr(cand, "flag_rfi", False):
        return enabled
    logical_count = plan.required_stop - plan.required_start
    padding = logical_count - len(raw)
    fill = medians * enabled
    s1 = padding * fill
    s2 = padding * fill**2
    count = padding * np.count_nonzero(fill)
    nonzero_channels = (padding > 0) & (fill != 0)

    def blocks():
        for start in range(0, len(raw), rows):
            block = np.asarray(raw[start : start + rows], dtype=np.float64)
            block = np.where(np.isfinite(block), block, medians[None, :])
            yield block * enabled

    for block in blocks():
        s1 += block.sum(axis=0)
        s2 += np.square(block).sum(axis=0)
        count += np.count_nonzero(block)
        nonzero_channels |= np.any(block != 0, axis=0)
    sk_sigma = float(getattr(cand, "spectral_kurtosis_sigma", 4))
    sg_sigma = float(getattr(cand, "savgol_sigma", 4))
    if sk_sigma < 0 or sg_sigma < 0 or (sk_sigma == 0 and sg_sigma == 0):
        raise ValueError("RFI sigmas must be nonnegative and at least one positive")
    masked = np.zeros(plan.nchans, dtype=bool)
    if sk_sigma > 0:
        mean = s1.sum() / count if count else 0.0
        variance = padding * np.square(fill[fill != 0] - mean).sum()
        for block in blocks():
            nonzero = block != 0
            variance += np.square(block[nonzero] - mean).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = mean**2 / (variance / count) if count else np.nan
            n = calc_N(cand.your_header.foff, plan.native_tsamp)
            channel_mask = ~nonzero_channels
            sum_d = np.ma.array(s1, mask=channel_mask)[()]
            sum2_d = np.ma.array(s2, mask=channel_mask)[()]
            sk = (
                ((logical_count * shape * n) + 1)
                * (logical_count * sum2_d / sum_d**2 - 1)
                / (logical_count - 1)
            )
        masked = np.asarray(sk_rejection_mask(sk, sk_sigma), dtype=bool)
    if sg_sigma > 0 and np.any(~masked):
        selected = np.flatnonzero(~masked)
        rejected = savgol_filter(
            s1[selected],
            channel_bandwidth=cand.your_header.foff,
            frequency_window=getattr(cand, "savgol_frequency_window", 15),
            sigma=sg_sigma,
        )
        masked[selected[rejected]] = True
    cand.rfi_mask = masked
    return enabled & ~masked


def process_gpu_streaming(cand, plan, *, memory_budget_bytes, device=0, profile=None):
    """Return finished images without placing the complete input on the GPU."""
    import cupy as cp

    from your.utils.gpu import _kernel_functions

    stats = {} if profile is None else profile
    started = perf_counter()
    budget = int(memory_budget_bytes)
    if budget < 64 * 1024:
        raise ValueError("GPU budget must allow at least 64 KiB of streaming workspace")
    dtype = np.dtype(decoded_sample_dtype(cand))
    suffixes = {
        "uint8": "u8",
        "uint16": "u16",
        "int16": "i16",
        "int32": "i32",
        "float16": "f16",
        "float32": "f32",
        "float64": "f64",
    }
    upload_dtype = dtype if dtype.name in suffixes else np.dtype("float64")
    suffix = suffixes[upload_dtype.name]
    rows = max(1, min(65536, budget // max(1, plan.nchans * dtype.itemsize * 4)))
    nread = plan.read_stop - plan.read_start
    stats.update(read_bytes=0, upload_bytes=0, streamed=True)
    with TemporaryDirectory(prefix="your-candidate-stage-") as directory:
        if nread:
            raw = np.memmap(
                Path(directory) / "samples.bin",
                dtype=dtype,
                mode="w+",
                shape=(nread, plan.nchans),
            )
            stage_start = perf_counter()
            for offset in range(0, nread, rows):
                count = min(rows, nread - offset)
                block = np.asarray(cand.get_data(plan.read_start + offset, count))
                if block.shape != (count, plan.nchans):
                    raise ValueError("reader returned incomplete streaming input")
                raw[offset : offset + count] = block
                stats["read_bytes"] += int(block.nbytes)
            raw.flush()
            stats["read_seconds"] = perf_counter() - stage_start
        else:
            raw = np.empty((0, plan.nchans), dtype=dtype)
            stats["read_seconds"] = 0.0
        prep_start = perf_counter()
        medians = np.zeros(plan.nchans, dtype=np.float64)
        peaks = np.zeros(plan.nchans, dtype=np.float64)
        for channel in range(plan.nchans):
            values = np.asarray(raw[:, channel], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values):
                medians[channel] = np.median(values)
                peaks[channel] = np.max(np.abs(values))
        enabled = _rfi_enabled(
            cand, plan, raw, medians, _enabled(cand, plan.nchans), rows
        )
        peak = float(peaks[enabled].max()) if np.any(enabled) else 0.0
        input_scale = peak if peak > 1e10 else 1.0
        stats["input_scale"] = input_scale
        stats["padding_rfi_cpu_seconds"] = perf_counter() - prep_start
        with cp.cuda.Device(device):
            _, functions = _kernel_functions(cp)
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                ft = np.zeros((plan.time_size, plan.nchans), dtype=np.float32)
                dmt = np.zeros((plan.dmsteps, plan.time_size), dtype=np.float32)
                # Bound native output, input, and reduction temporaries together.
                max_native = max(1, min(4096, budget // 256))
                time_block = max(1, min(plan.time_size, budget // 1024))
                stats["streaming_max_time_bins"] = time_block
                shared_capacity = min(4096, max(128, budget // 32))
                kernel_ms = 0.0
                upload_ms = 0.0
                download_ms = 0.0

                def timed(call, category):
                    nonlocal kernel_ms, upload_ms, download_ms
                    begin, end = cp.cuda.Event(), cp.cuda.Event()
                    begin.record(stream)
                    result = call()
                    end.record(stream)
                    end.synchronize()
                    elapsed = cp.cuda.get_elapsed_time(begin, end)
                    if category == "upload":
                        upload_ms += elapsed
                    elif category == "download":
                        download_ms += elapsed
                    else:
                        kernel_ms += elapsed
                    return result

                def upload(channel, low, high):
                    lo = max(plan.read_start, min(plan.read_stop, low))
                    hi = max(lo, min(plan.read_stop, high))
                    host = np.ascontiguousarray(
                        raw[lo - plan.read_start : hi - plan.read_start, channel],
                        dtype=upload_dtype,
                    ).reshape(1, -1)
                    stats["upload_bytes"] += int(host.nbytes)
                    return timed(lambda: cp.asarray(host), "upload"), lo, hi - lo

                def time_tiles(block_start, block_stop):
                    if plan.time_bin <= max_native:
                        bins_per_tile = max(1, max_native // plan.time_bin)
                        for bin_start in range(block_start, block_stop, bins_per_tile):
                            bins = min(bins_per_tile, block_stop - bin_start)
                            yield (
                                bin_start,
                                bins,
                                bin_start * plan.time_bin,
                                bins * plan.time_bin,
                            )
                    else:
                        for bin_start in range(block_start, block_stop):
                            for offset in range(0, plan.time_bin, max_native):
                                yield (
                                    bin_start,
                                    1,
                                    bin_start * plan.time_bin + offset,
                                    min(max_native, plan.time_bin - offset),
                                )

                # Channel-local intervals remove the full-band dispersion halo.
                for channel in range(plan.nchans):
                    if not enabled[channel]:
                        continue
                    median_d = cp.asarray(
                        [medians[channel] / input_scale], dtype=cp.float32
                    )
                    enabled_d = cp.ones(1, dtype=cp.uint8)
                    ft_delay = cp.asarray([plan.ft_delays[channel]], dtype=cp.int64)
                    ft_kernel = functions[f"dedisp_ft_{suffix}"]
                    for block_start in range(0, plan.time_size, time_block):
                        block_stop = min(plan.time_size, block_start + time_block)
                        accumulator = cp.zeros(
                            block_stop - block_start, dtype=cp.float32
                        )
                        for bin_start, bins, native_offset, count in time_tiles(
                            block_start, block_stop
                        ):
                            output_start = plan.output_start + native_offset
                            delay = int(plan.ft_delays[channel])
                            source, origin, nsource = upload(
                                channel,
                                output_start + delay,
                                output_start + delay + count,
                            )
                            native = cp.empty((count, 1), dtype=cp.float32)
                            args = (
                                source,
                                median_d,
                                enabled_d,
                                ft_delay,
                                native,
                                np.int32(1),
                                np.int32(0),
                                np.int32(1),
                                np.int64(origin),
                                np.int64(nsource),
                                np.int64(plan.read_start),
                                np.int64(plan.read_stop),
                                np.int64(plan.nspectra),
                                np.int64(output_start),
                                np.int32(count),
                                np.float64(input_scale),
                            )
                            timed(
                                lambda: ft_kernel(
                                    ((count + 31) // 32, 1), (32, 8), args
                                ),
                                "compute",
                            )
                            target = accumulator[
                                bin_start - block_start : bin_start - block_start + bins
                            ]
                            timed(
                                lambda: cp.add(
                                    target,
                                    native.reshape(bins, -1).sum(
                                        axis=1, dtype=cp.float32
                                    ),
                                    out=target,
                                ),
                                "compute",
                            )
                        timed(
                            lambda: cp.divide(
                                accumulator, plan.time_bin, out=accumulator
                            ),
                            "compute",
                        )
                        ft[block_start:block_stop, channel] = timed(
                            lambda: accumulator.get(), "download"
                        )
                # Accumulate all channels before dividing or normalizing each DM tile.
                for dm_start in range(0, plan.dmsteps, 8):
                    dm_stop = min(plan.dmsteps, dm_start + 8)
                    for block_start in range(0, plan.time_size, time_block):
                        block_stop = min(plan.time_size, block_start + time_block)
                        accumulator = cp.zeros(
                            (dm_stop - dm_start, block_stop - block_start),
                            dtype=cp.float32,
                        )
                        for channel in range(plan.nchans):
                            if not enabled[channel]:
                                continue
                            median_d = cp.asarray(
                                [medians[channel] / input_scale], dtype=cp.float32
                            )
                            enabled_d = cp.ones(1, dtype=cp.uint8)
                            tasks = [(dm_start, dm_stop)]
                            while tasks:
                                d0, d1 = tasks.pop()
                                offsets = plan.delays[d0:d1, channel]
                                span = (
                                    int(offsets.max()) - int(offsets.min()) + max_native
                                )
                                if (
                                    span * upload_dtype.itemsize > budget // 4
                                    and d1 - d0 > 1
                                ):
                                    middle = (d0 + d1) // 2
                                    tasks.extend(((middle, d1), (d0, middle)))
                                    continue
                                delays_d = cp.asarray(offsets[:, None], dtype=cp.int64)
                                kernel = functions[f"dmt_tiled_{suffix}"]
                                for bin_start, bins, native_offset, count in time_tiles(
                                    block_start, block_stop
                                ):
                                    output_start = plan.output_start + native_offset
                                    source, origin, nsource = upload(
                                        channel,
                                        output_start + int(offsets.min()),
                                        output_start + int(offsets.max()) + count,
                                    )
                                    native = cp.empty(
                                        (d1 - d0, count), dtype=cp.float32
                                    )
                                    args = (
                                        source,
                                        median_d,
                                        enabled_d,
                                        delays_d,
                                        native,
                                        np.int32(1),
                                        np.int32(0),
                                        np.int32(1),
                                        np.int32(d1 - d0),
                                        np.int64(origin),
                                        np.int64(nsource),
                                        np.int64(plan.read_start),
                                        np.int64(plan.read_stop),
                                        np.int64(plan.nspectra),
                                        np.int64(output_start),
                                        np.int32(count),
                                        np.int32(shared_capacity),
                                        np.int32(1),
                                        np.float64(input_scale),
                                    )
                                    timed(
                                        lambda: kernel(
                                            ((count + 127) // 128, 1),
                                            (128,),
                                            args,
                                            shared_mem=shared_capacity * 4,
                                        ),
                                        "compute",
                                    )
                                    target = accumulator[
                                        d0 - dm_start : d1 - dm_start,
                                        bin_start - block_start : bin_start
                                        - block_start
                                        + bins,
                                    ]
                                    timed(
                                        lambda: cp.add(
                                            target,
                                            native.reshape(d1 - d0, bins, -1).sum(
                                                axis=2, dtype=cp.float32
                                            ),
                                            out=target,
                                        ),
                                        "compute",
                                    )
                        timed(
                            lambda: cp.divide(
                                accumulator,
                                plan.nchans * plan.time_bin,
                                out=accumulator,
                            ),
                            "compute",
                        )
                        dmt[dm_start:dm_stop, block_start:block_stop] = timed(
                            lambda: accumulator.get(), "download"
                        )
                stream.synchronize()
                stats.update(
                    dedisp_bin_seconds=kernel_ms / 1000,
                    upload_seconds=upload_ms / 1000,
                    download_seconds=download_ms / 1000,
                )
    result = finish_products(ft, dmt, plan, profile=stats)
    stats["total_seconds"] = perf_counter() - started
    return result
