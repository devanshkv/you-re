"""Optional CuPy kernels for direct, non-circular candidate products."""

import logging
import math
import os
import subprocess
import threading
from operator import index
from time import perf_counter

import numpy as np

from your.utils.candidate_contract import parse_kill_mask

logger = logging.getLogger(__name__)

_CHANNEL_TILE = 32
_TIME_TILE = 128
_DM_TILE = 8
_KERNELS = {}
_KERNELS_LOCK = threading.Lock()
_DTYPE_SUFFIX = {
    np.dtype(np.uint8): "u8",
    np.dtype(np.uint16): "u16",
    np.dtype(np.int16): "i16",
    np.dtype(np.int32): "i32",
    np.dtype(np.float16): "f16",
    np.dtype(np.float32): "f32",
    np.dtype(np.float64): "f64",
}
_BOUNDED_SCALE_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float16),
}


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy with CUDA 12 support is required for GPU processing"
        ) from exc
    return cp


def gpu_available(device=0):
    """Return whether CuPy can see the requested CUDA device."""
    try:
        cp = _cupy()
        return 0 <= int(device) < cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False


def _kernel_functions(cp):
    device_id = cp.cuda.runtime.getDevice()
    with _KERNELS_LOCK:
        cached = _KERNELS.get(device_id)
        if cached is not None:
            return cached
        path = os.path.join(os.path.dirname(__file__), "gpu_kernels.cu")
        with open(path, "r", encoding="utf8") as source_file:
            source = source_file.read()
        module = cp.RawModule(code=source, options=("--std=c++11",), backend="nvrtc")
        names = ("dedisp_ft_", "dmt_tiled_")
        functions = {
            key: module.get_function(key)
            for key in (
                "bin_ft",
                "bin_dmt",
                *("corner_turn_b" + str(size) for size in (1, 2, 4, 8)),
                *(name + suffix for name in names for suffix in _DTYPE_SUFFIX.values()),
            )
        }
        cached = (module, functions)
        _KERNELS[device_id] = cached
        return cached


def _corner_turn(cp, source, destination, nsamples, time_offset, stream):
    """Transpose raw time-major samples into a pitched resident channel window."""
    _, kernels = _kernel_functions(cp)
    channel_tiles = (source.shape[1] + 31) // 32
    kernels["corner_turn_b" + str(source.dtype.itemsize)](
        (((nsamples + 31) // 32) * channel_tiles,),
        (32, 8),
        (
            source,
            destination,
            np.int64(nsamples),
            np.int32(source.shape[1]),
            np.int64(destination.strides[0] // destination.dtype.itemsize),
            np.int64(time_offset),
        ),
        stream=stream,
    )


def _elapsed_events(cp, events):
    return {
        name: sum(cp.cuda.get_elapsed_time(start, stop) for start, stop in pairs)
        / 1000.0
        for name, pairs in events.items()
    }


def _event_start(cp, stream, events, name):
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record(stream)
    events.setdefault(name, []).append((start, stop))
    return stop


def _kill_channels(cand, nchans):
    return parse_kill_mask(getattr(cand, "kill_mask", np.array([False])), nchans)


def _host_coverage(data, resident, resident_start, read_start, read_stop, cp, stream):
    if data is not None:
        if hasattr(data, "get"):
            return data.get(stream=stream)
        return np.asarray(data)
    count = read_stop - read_start
    if count == 0:
        return np.empty((0, resident.shape[0]), dtype=resident.dtype)
    start = read_start - resident_start
    with stream:
        return resident[:, start : start + count].T.get(stream=stream)


def _channel_enabled(cand, plan, data, resident, resident_start, cp, stream, profile):
    killed = _kill_channels(cand, plan.nchans)
    if not getattr(cand, "flag_rfi", False):
        return (~killed).astype(np.uint8)

    started = perf_counter()
    coverage = _host_coverage(
        data, resident, resident_start, plan.read_start, plan.read_stop, cp, stream
    )
    coverage = np.asarray(coverage)
    finite = np.isfinite(coverage)
    medians = np.zeros(plan.nchans, dtype=np.float64)
    for channel in range(plan.nchans):
        values = coverage[finite[:, channel], channel]
        if values.size:
            medians[channel] = np.median(values)

    logical = np.empty(
        (plan.required_stop - plan.required_start, plan.nchans), dtype=np.float64
    )
    logical[:] = medians
    finite_coverage = np.where(finite, coverage, medians[None, :])
    overlap_start = max(plan.required_start, plan.read_start)
    overlap_stop = min(plan.required_stop, plan.read_stop)
    if overlap_stop > overlap_start:
        dst = overlap_start - plan.required_start
        src = overlap_start - plan.read_start
        logical[dst : dst + overlap_stop - overlap_start] = finite_coverage[
            src : src + overlap_stop - overlap_start
        ]
    logical[:, killed] = 0

    from your.utils.rfi import sk_sg_filter

    header = getattr(cand, "your_header", None)
    foff = getattr(header, "foff", None)
    if foff is None:
        raise ValueError("your_header.foff is required when flag_rfi is enabled")
    rfi_mask = np.asarray(
        sk_sg_filter(
            data=logical,
            foff=foff,
            tsamp=plan.native_tsamp,
            spectral_kurtosis_sigma=getattr(cand, "spectral_kurtosis_sigma", 4),
            savgol_frequency_window=getattr(cand, "savgol_frequency_window", 15),
            savgol_sigma=getattr(cand, "savgol_sigma", 4),
        )
    )
    if rfi_mask.shape != (plan.nchans,):
        raise ValueError("RFI filter returned an invalid channel mask")
    cand.rfi_mask = rfi_mask.copy()
    if profile is not None:
        profile["rfi_mask_s"] = perf_counter() - started
    return (~(killed | rfi_mask)).astype(np.uint8)


def _median(cp, source, median_start, median_stop, source_start, input_scale=1.0):
    if median_stop <= median_start:
        return cp.zeros(source.shape[0], dtype=cp.float32)
    lo = median_start - source_start
    hi = median_stop - source_start
    values = source[:, lo:hi].astype(cp.float64, copy=False)
    if input_scale != 1.0:
        values = values / input_scale
    values = cp.where(cp.isfinite(values), values, cp.nan)
    medians = cp.nanmedian(values, axis=1)
    return cp.nan_to_num(medians, nan=0.0).astype(cp.float32)


def _candidate_input_scale(
    cp,
    data,
    resident,
    resident_start,
    read_start,
    read_stop,
    enabled,
    device,
    stream,
    profile,
):
    """Scan finite values in bounded channel/time tiles for one shared divisor."""
    started = perf_counter()
    dtype = np.dtype(resident.dtype if resident is not None else data.dtype)
    if dtype in _BOUNDED_SCALE_DTYPES:
        if profile is not None:
            profile["scale_s"] = perf_counter() - started
        return 1.0

    from your.utils.candidate_pipeline import product_input_scale

    peak = 0.0
    read_count = int(read_stop) - int(read_start)
    if read_count > 0 and np.any(enabled):
        if resident is None and not hasattr(data, "get"):
            for time_start in range(0, read_count, 65536):
                time_stop = min(read_count, time_start + 65536)
                for channel_start in range(0, len(enabled), _CHANNEL_TILE):
                    channel_stop = min(len(enabled), channel_start + _CHANNEL_TILE)
                    active = np.flatnonzero(enabled[channel_start:channel_stop])
                    if not active.size:
                        continue
                    block = np.asarray(
                        data[time_start:time_stop, channel_start:channel_stop]
                    ).T[active]
                    values = block.astype(np.float64, copy=False)
                    finite = np.isfinite(values)
                    if np.any(finite):
                        peak = max(peak, float(np.max(np.abs(values[finite]))))
        else:
            with cp.cuda.Device(device):
                stream = stream or cp.cuda.Stream.null
                with stream:
                    peak_device = cp.zeros((), dtype=cp.float64)
                    for time_offset in range(0, read_count, 65536):
                        count = min(65536, read_count - time_offset)
                        for channel_start in range(0, len(enabled), _CHANNEL_TILE):
                            channel_stop = min(
                                len(enabled), channel_start + _CHANNEL_TILE
                            )
                            active = np.flatnonzero(enabled[channel_start:channel_stop])
                            if not active.size:
                                continue
                            active_device = cp.asarray(active, dtype=cp.int32)
                            if resident is not None:
                                lo = read_start + time_offset - resident_start
                                block = resident[
                                    channel_start:channel_stop, lo : lo + count
                                ][active_device]
                            else:
                                block = data[
                                    time_offset : time_offset + count,
                                    channel_start:channel_stop,
                                ].T[active_device]
                                block = cp.ascontiguousarray(block)
                            values = block.astype(cp.float64, copy=False)
                            finite = cp.isfinite(values)
                            magnitude = cp.abs(values)
                            cp.copyto(magnitude, 0.0, where=~finite)
                            cp.maximum(peak_device, cp.max(magnitude), out=peak_device)
                    peak = float(peak_device.get(stream=stream))
                    stream.synchronize()

    input_scale = product_input_scale(peak)
    if profile is not None:
        profile["scale_s"] = perf_counter() - started
    return input_scale


def _shared_limit(cp):
    return int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))


def _dmt_shared_capacities(delays, max_shared):
    """Size all channel groups from the widest per-channel DM_TILE spread."""
    dmsteps, nchans = delays.shape
    if not nchans:
        return np.empty(0, dtype=np.int64)
    spreads = np.zeros(nchans, dtype=np.int64)
    full_stop = dmsteps // _DM_TILE * _DM_TILE
    if full_stop:
        tiles = delays[:full_stop].reshape(-1, _DM_TILE, nchans)
        spreads = np.ptp(tiles, axis=1).max(axis=0)
    if full_stop < dmsteps:
        np.maximum(spreads, np.ptp(delays[full_stop:], axis=0), out=spreads)
    capacities = np.maximum.reduceat(spreads, np.arange(0, nchans, _CHANNEL_TILE))
    # Preserve the scalar zero floor, including signed ptp overflow.
    np.maximum(capacities, 0, out=capacities)
    # Clamp before addition so valid int64 delay spreads cannot overflow.
    np.minimum(
        capacities,
        max(0, int(max_shared) // np.dtype(np.float32).itemsize - _TIME_TILE),
        out=capacities,
    )
    capacities += _TIME_TILE
    return capacities


def _dmt_shared_capacity(delays, channel_start, channel_stop, max_shared):
    """Size one shared tile for an arbitrary, possibly empty channel range."""
    capacities = _dmt_shared_capacities(
        delays[:, channel_start:channel_stop], max_shared
    )
    return int(capacities.max(initial=_TIME_TILE))


def _compute_binned_products(
    *,
    cp,
    data,
    resident,
    resident_start,
    source_start,
    valid_start,
    valid_stop,
    median_start,
    median_stop,
    nspectra,
    nchans,
    delays,
    ft_delays,
    output_start,
    native_count,
    time_bin,
    time_size,
    enabled,
    device,
    stream,
    profile=None,
    compute_ft=True,
    compute_dmt=True,
    input_scale=1.0,
):
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("input_scale must be finite and positive")
    if native_count != time_size * time_bin:
        raise ValueError("native_count must equal time_size * time_bin")
    if not (compute_ft or compute_dmt):
        return None, None
    if native_count == 0:
        dmsteps = int(np.asarray(delays).shape[0])
        return (
            np.empty((time_size, nchans), dtype=np.float32) if compute_ft else None,
            np.empty((dmsteps, time_size), dtype=np.float32) if compute_dmt else None,
        )
    dtype = np.dtype(resident.dtype if resident is not None else data.dtype)
    try:
        suffix = _DTYPE_SUFFIX[dtype]
    except KeyError as exc:
        raise TypeError(f"unsupported GPU input dtype: {dtype}") from exc

    with cp.cuda.Device(device):
        stream = stream or cp.cuda.Stream(non_blocking=True)
        module, kernels = _kernel_functions(cp)
        events = {}
        with stream:
            upload_stop = _event_start(cp, stream, events, "upload")
            delays_device = cp.asarray(delays, dtype=cp.int64)
            ft_delays_device = (
                None if ft_delays is None else cp.asarray(ft_delays, dtype=cp.int64)
            )
            enabled_device = cp.asarray(enabled, dtype=cp.uint8)
            upload_stop.record(stream)

            if resident is not None:
                if resident.ndim != 2 or resident.shape[0] != nchans:
                    raise ValueError(
                        "resident must be channel-major with shape (nchans, N)"
                    )
                if int(resident.device.id) != int(device):
                    raise ValueError("resident must be on the selected CUDA device")
                storage_start = int(resident_start)
                source_count = int(resident.shape[1])
            else:
                if data.ndim != 2 or data.shape[1] != nchans:
                    raise ValueError("data must be time-major with shape (N, nchans)")
                storage_start = int(source_start)
                source_count = int(data.shape[0])

            if compute_ft:
                ft_native = cp.empty((native_count, nchans), dtype=cp.float32)
                ft_binned = cp.empty((time_size, nchans), dtype=cp.float32)
            else:
                ft_native = ft_binned = None
            if compute_dmt:
                dmsteps = int(np.asarray(delays).shape[0])
                shared_capacities = _dmt_shared_capacities(
                    np.asarray(delays), _shared_limit(cp)
                )
                dmt_native = cp.empty((dmsteps, native_count), dtype=cp.float32)
                dmt_binned = cp.empty((dmsteps, time_size), dtype=cp.float32)
            else:
                dmsteps = 0
                dmt_native = dmt_binned = None

            all_delays = np.asarray(delays).ravel()
            if compute_ft:
                all_delays = np.concatenate((all_delays, np.asarray(ft_delays).ravel()))
            source_first = int(output_start) + int(all_delays.min())
            source_stop = int(output_start) + int(native_count) + int(all_delays.max())
            medians_needed = not (
                dtype.kind in "iu"
                and source_first >= int(valid_start)
                and source_stop <= int(valid_stop)
            )

            if data is not None and source_count == 0:
                source_count_device = 0
            else:
                source_count_device = source_count

            for channel_start in range(0, nchans, _CHANNEL_TILE):
                channel_stop = min(nchans, channel_start + _CHANNEL_TILE)
                group_channels = channel_stop - channel_start
                upload_stop = _event_start(cp, stream, events, "upload")
                if resident is not None:
                    source_group = resident[channel_start:channel_stop]
                    median_source = source_group
                    median_origin = storage_start
                else:
                    host_group = data[:, channel_start:channel_stop].T
                    if hasattr(host_group, "get"):
                        source_group = cp.ascontiguousarray(host_group)
                    else:
                        source_group = cp.asarray(np.ascontiguousarray(host_group))
                    median_source = source_group
                    median_origin = storage_start
                upload_stop.record(stream)

                median_stop_event = _event_start(cp, stream, events, "median")
                if medians_needed:
                    medians_device = _median(
                        cp,
                        median_source,
                        median_start,
                        median_stop,
                        median_origin,
                        input_scale,
                    )
                else:
                    medians_device = cp.zeros(group_channels, dtype=cp.float32)
                median_stop_event.record(stream)

                kernel_args = (
                    source_group,
                    medians_device,
                    enabled_device[channel_start:channel_stop],
                )
                dedisp_stop = _event_start(cp, stream, events, "dedisp")
                if compute_ft:
                    kernels["dedisp_ft_" + suffix](
                        (
                            math.ceil(native_count / 32),
                            math.ceil(group_channels / 8),
                        ),
                        (32, 8),
                        kernel_args
                        + (
                            ft_delays_device,
                            ft_native,
                            nchans,
                            channel_start,
                            group_channels,
                            storage_start,
                            source_count_device,
                            int(valid_start),
                            int(valid_stop),
                            int(nspectra),
                            int(output_start),
                            native_count,
                            np.float64(input_scale),
                        ),
                        stream=stream,
                    )
                if compute_dmt:
                    shared_capacity = int(
                        shared_capacities[channel_start // _CHANNEL_TILE]
                    )
                    shared_bytes = shared_capacity * np.dtype(np.float32).itemsize
                    kernels["dmt_tiled_" + suffix](
                        (
                            math.ceil(native_count / _TIME_TILE),
                            math.ceil(dmsteps / _DM_TILE),
                        ),
                        (_TIME_TILE,),
                        kernel_args
                        + (
                            delays_device,
                            dmt_native,
                            nchans,
                            channel_start,
                            group_channels,
                            dmsteps,
                            storage_start,
                            source_count_device,
                            int(valid_start),
                            int(valid_stop),
                            int(nspectra),
                            int(output_start),
                            native_count,
                            shared_capacity,
                            int(channel_start == 0),
                            np.float64(input_scale),
                        ),
                        shared_mem=shared_bytes,
                        stream=stream,
                    )
                dedisp_stop.record(stream)

            bin_stop = _event_start(cp, stream, events, "bin")
            if compute_ft:
                kernels["bin_ft"](
                    (math.ceil(time_size / 32), math.ceil(nchans / 8)),
                    (32, 8),
                    (
                        ft_native,
                        ft_binned,
                        nchans,
                        native_count,
                        time_size,
                        time_bin,
                    ),
                    stream=stream,
                )
            if compute_dmt:
                kernels["bin_dmt"](
                    (math.ceil(time_size / 128), math.ceil(dmsteps / 4)),
                    (128, 4),
                    (
                        dmt_native,
                        dmt_binned,
                        nchans,
                        dmsteps,
                        native_count,
                        time_size,
                        time_bin,
                    ),
                    stream=stream,
                )
            bin_stop.record(stream)

            download_stop = _event_start(cp, stream, events, "download")
            ft_host = (
                None if not compute_ft else ft_binned.get(stream=stream, blocking=False)
            )
            dmt_host = (
                None
                if not compute_dmt
                else dmt_binned.get(stream=stream, blocking=False)
            )
            download_stop.record(stream)
            stream.synchronize()

        if profile is not None:
            for name, elapsed in _elapsed_events(cp, events).items():
                profile[name + "_s"] = elapsed
        return ft_host, dmt_host


def estimate_process_gpu_bytes(plan, input_dtype):
    """Estimate per-job GPU workspace, excluding caller-owned resident data."""
    dtype = np.dtype(input_dtype)
    read_count = max(0, int(plan.read_stop) - int(plan.read_start))
    channels = int(plan.nchans)
    dmsteps = int(plan.dmsteps)
    group_channels = min(channels, _CHANNEL_TILE)
    staged = group_channels * read_count * dtype.itemsize
    scale_workspace = (
        0
        if dtype in _BOUNDED_SCALE_DTYPES
        else group_channels * min(read_count, 65536) * (dtype.itemsize + 24)
    )
    needs_medians = not (
        dtype.kind in "iu" and int(plan.pad_before) == 0 and int(plan.pad_after) == 0
    )
    median_workspace = group_channels * read_count * 32 if needs_medians else 0
    native_products = int(plan.native_count) * (channels + dmsteps) * 4
    binned_products = int(plan.time_size) * (channels + dmsteps) * 4
    delay_tables = (channels + dmsteps * channels) * 8
    return int(
        staged
        + scale_workspace
        + median_workspace
        + native_products
        + binned_products
        + delay_tables
        + 64 * 1024
    )


def process_gpu(
    cand,
    plan,
    data=None,
    device=0,
    *,
    resident=None,
    resident_start=None,
    stream=None,
    profile=None,
):
    """Build finished CuPy products with direct, non-wrapping time indices."""
    cp = _cupy()
    if profile is not None and not hasattr(profile, "__setitem__"):
        raise TypeError("profile must be a mutable mapping")
    if resident is None and data is None:
        raise ValueError("provide plan data or a resident CuPy array")
    if resident is None:
        if hasattr(data, "get"):
            shape = data.shape
        else:
            data = np.asarray(data)
            shape = data.shape
        expected = (plan.read_stop - plan.read_start, plan.nchans)
        if shape != expected:
            raise ValueError("data must match the planned clipped read interval")
        source_start = plan.read_start
        valid_start, valid_stop = plan.read_start, plan.read_stop
    else:
        if resident_start is None:
            raise ValueError("resident_start is required with resident data")
        if not hasattr(resident, "device"):
            raise ValueError("resident must be a CuPy array")
        if resident.ndim != 2 or resident.shape[0] != plan.nchans:
            raise ValueError("resident must be channel-major with shape (nchans, N)")
        if not resident.flags.c_contiguous:
            raise ValueError("resident must be C-contiguous")
        if int(resident.device.id) != int(device):
            raise ValueError("resident must be on the selected CUDA device")
        source_start = int(resident_start)
        valid_start, valid_stop = plan.read_start, plan.read_stop
        if (
            resident_start > plan.read_start
            or resident_start + resident.shape[1] < plan.read_stop
        ):
            raise ValueError("resident must cover the planned clipped read interval")
        if data is not None:
            raise ValueError("data and resident are mutually exclusive")

    started = perf_counter()
    with cp.cuda.Device(device):
        active_stream = stream or cp.cuda.Stream.null
        enabled = _channel_enabled(
            cand,
            plan,
            data,
            resident,
            resident_start,
            cp,
            active_stream,
            profile,
        )
        input_scale = _candidate_input_scale(
            cp,
            data,
            resident,
            resident_start,
            plan.read_start,
            plan.read_stop,
            enabled,
            device,
            stream,
            profile,
        )
    if profile is not None:
        profile.setdefault("mask_s", perf_counter() - started)
    ft, dmt = _compute_binned_products(
        cp=cp,
        data=data,
        resident=resident,
        resident_start=resident_start,
        source_start=source_start,
        valid_start=valid_start,
        valid_stop=valid_stop,
        median_start=plan.read_start,
        median_stop=plan.read_stop,
        nspectra=plan.nspectra,
        nchans=plan.nchans,
        delays=plan.delays,
        ft_delays=plan.ft_delays,
        output_start=plan.output_start,
        native_count=plan.native_count,
        time_bin=plan.time_bin,
        time_size=plan.time_size,
        enabled=enabled,
        device=device,
        stream=stream,
        profile=profile,
        input_scale=input_scale,
    )
    from your.utils.candidate_pipeline import finish_products

    started = perf_counter()
    result = finish_products(ft, dmt, plan, profile=profile)
    if profile is not None:
        profile["finish_s"] = perf_counter() - started
    return result


def _time_bounds(nt, time_range):
    if time_range is None:
        return 0, nt
    try:
        start, stop = map(index, time_range)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_range must contain start and stop integer samples"
        ) from exc
    if not 0 <= start <= stop <= nt:
        raise ValueError("expected 0 <= start <= stop <= data length")
    return start, stop


def _dispersion_delays(cand, dm_values):
    frequencies = np.asarray(cand.chan_freqs, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.all(np.isfinite(frequencies))
        or np.any(frequencies <= 0)
    ):
        raise ValueError("expected one finite positive frequency per channel")
    tsamp = getattr(cand, "native_tsamp", None)
    if tsamp is None:
        tsamp = cand.your_header.tsamp
    tsamp = float(tsamp)
    if not np.isfinite(tsamp) or tsamp <= 0:
        raise ValueError("native_tsamp must be finite and positive")
    dm_values = np.asarray(dm_values, dtype=np.float64)
    if not np.all(np.isfinite(dm_values)):
        raise ValueError("DM values must be finite")
    delay_samples = (
        4148808.0
        * dm_values[:, None]
        * (1.0 / frequencies[None, :] ** 2 - 1.0 / np.max(frequencies) ** 2)
        / 1000.0
        / tsamp
    )
    if not np.all(np.isfinite(delay_samples)) or np.any(np.abs(delay_samples) >= 2**63):
        raise ValueError("dispersion delay exceeds int64 sample range")
    return np.rint(delay_samples).astype(np.int64)


def _wrapper_source(cand, dm_values, start, stop):
    if cand.data is None:
        raise ValueError("candidate data is required for GPU processing")
    helper = getattr(cand, "_direct_input", None)
    if callable(helper):
        source, delays, origin = helper(dm_values, start, stop)
        chunk_start = getattr(cand, "_chunk_start", None)
        from_file = (
            chunk_start is not None
            and getattr(cand, "_chunk_shape", None) == cand.data.shape
        )
        base = int(chunk_start) if from_file else 0
        source_start = base + int(origin)
        output_start = base + start
        header = getattr(cand, "your_header", None)
        nspectra = int(getattr(header, "nspectra", len(cand.data)))
        valid_start = max(0, source_start)
        valid_stop = min(nspectra, source_start + len(source))
        return (
            source,
            delays,
            source_start,
            valid_start,
            valid_stop,
            nspectra,
            output_start,
        )

    source = np.asarray(cand.data)
    delays = _dispersion_delays(cand, dm_values)
    return source, delays, 0, 0, len(source), len(source), start


def _wrapper_enabled(cand, nchans):
    enabled = ~_kill_channels(cand, nchans)
    rfi_mask = np.asarray(getattr(cand, "rfi_mask", np.zeros(nchans, dtype=bool)))
    if rfi_mask.dtype.kind == "b" and rfi_mask.shape == (nchans,):
        enabled &= ~rfi_mask
    return enabled.astype(np.uint8)


def gpu_dedisperse(
    cand, device=0, dms=None, *, time_range=None, stream=None, profile=None
):
    """Direct GPU dedispersion; missing samples use per-channel medians, never wrap."""
    if cand.data is None:
        logger.warning("No data in self.data, run self.get_chunk() first")
        cand.dedispersed = None
        return cand
    start, stop = _time_bounds(len(cand.data), time_range)
    dm = float(cand.dm if dms is None else dms)
    source, delays, source_start, valid_start, valid_stop, nspectra, output_start = (
        _wrapper_source(cand, np.array([dm]), start, stop)
    )
    ft, _ = _compute_binned_products(
        cp=_cupy(),
        data=source,
        resident=None,
        resident_start=None,
        source_start=source_start,
        valid_start=valid_start,
        valid_stop=valid_stop,
        median_start=valid_start,
        median_stop=valid_stop,
        nspectra=nspectra,
        nchans=len(cand.chan_freqs),
        delays=delays,
        ft_delays=delays[0],
        output_start=output_start,
        native_count=stop - start,
        time_bin=1,
        time_size=stop - start,
        enabled=_wrapper_enabled(cand, len(cand.chan_freqs)),
        device=device,
        stream=stream,
        profile=profile,
        compute_dmt=False,
    )
    cand.dedispersed = ft
    return cand


def gpu_dmt(
    cand, device=0, dmsteps=256, *, dms=None, time_range=None, stream=None, profile=None
):
    """Direct GPU DM-time products with optional explicit DM trials and time range."""
    if cand.data is None:
        logger.warning("No data in self.data, run self.get_chunk() first")
        cand.dmt = None
        return cand
    start, stop = _time_bounds(len(cand.data), time_range)
    if dms is None:
        dmsteps = index(dmsteps)
        if dmsteps <= 0:
            raise ValueError("dmsteps must be positive")
        dm = float(cand.dm)
        dm_values = np.linspace(0.0, 2.0 * dm, dmsteps, dtype=np.float64)
    else:
        dm_values = np.atleast_1d(np.asarray(dms, dtype=np.float64))
        if dm_values.ndim != 1 or dm_values.size == 0:
            raise ValueError("dms must be a nonempty one-dimensional sequence")
        dmsteps = len(dm_values)
    source, delays, source_start, valid_start, valid_stop, nspectra, output_start = (
        _wrapper_source(cand, dm_values, start, stop)
    )
    _, dmt = _compute_binned_products(
        cp=_cupy(),
        data=source,
        resident=None,
        resident_start=None,
        source_start=source_start,
        valid_start=valid_start,
        valid_stop=valid_stop,
        median_start=valid_start,
        median_stop=valid_stop,
        nspectra=nspectra,
        nchans=len(cand.chan_freqs),
        delays=delays,
        ft_delays=None,
        output_start=output_start,
        native_count=stop - start,
        time_bin=1,
        time_size=stop - start,
        enabled=_wrapper_enabled(cand, len(cand.chan_freqs)),
        device=device,
        stream=stream,
        profile=profile,
        compute_ft=False,
    )
    cand.dmt = dmt
    return cand


def gpu_dedisp_and_dmt_crop(
    cand, device=0, dms=None, dmsteps=256, *, time_range=None, stream=None, profile=None
):
    """Build the legacy centered, time-cropped FT and DM-time GPU products."""
    if cand.data is None:
        raise ValueError("candidate data is required for GPU processing")
    nchans = len(cand.chan_freqs)
    if nchans < 256:
        raise IndexError("GPU candmaker will not work if nchans < 256.")
    time_bin = max(1, int(cand.width) // 2)
    start, stop = _time_bounds(len(cand.data), time_range)
    native_count = ((stop - start) // time_bin) * time_bin
    time_size = native_count // time_bin
    if time_size < 256:
        raise IndexError("GPU candmaker requires at least 256 binned time samples.")
    if dms is None:
        dmsteps = index(dmsteps)
        if dmsteps <= 0:
            raise ValueError("dmsteps must be positive")
        dm = float(cand.dm)
        dm_values = np.linspace(0.0, 2.0 * dm, dmsteps, dtype=np.float64)
    else:
        dm_values = np.atleast_1d(np.asarray(dms, dtype=np.float64))
        if dm_values.ndim != 1 or dm_values.size == 0:
            raise ValueError("dms must be a nonempty one-dimensional sequence")
        dmsteps = len(dm_values)
    dm_values = np.concatenate(([float(cand.dm)], dm_values))
    source, delays, source_start, valid_start, valid_stop, nspectra, output_start = (
        _wrapper_source(cand, dm_values, start, start + native_count)
    )
    ft, dmt = _compute_binned_products(
        cp=_cupy(),
        data=source,
        resident=None,
        resident_start=None,
        source_start=source_start,
        valid_start=valid_start,
        valid_stop=valid_stop,
        median_start=valid_start,
        median_stop=valid_stop,
        nspectra=nspectra,
        nchans=nchans,
        delays=delays[1:],
        ft_delays=delays[0],
        output_start=output_start,
        native_count=native_count,
        time_bin=time_bin,
        time_size=time_size,
        enabled=_wrapper_enabled(cand, nchans),
        device=device,
        stream=stream,
        profile=profile,
    )
    frequency_bin = max(1, nchans // 256)
    output_channels = nchans // frequency_bin
    ft = (
        ft[:, : output_channels * frequency_bin]
        .reshape(time_size, output_channels, frequency_bin)
        .sum(axis=2, dtype=np.float32)
    )
    time_start = time_size // 2 - 128
    cand.dedispersed = ft[time_start : time_start + 256]
    cand.dmt = dmt[:, time_start : time_start + 256]
    return cand


def get_gpu_memory_map(gpu_id):
    """Return free GPU memory in MiB using nvidia-smi."""
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=memory.free",
            "--format=csv,nounits,noheader",
        ]
    )
    return int(result.decode())
