"""Direct, non-circular candidate products and their host-side finishing steps."""

from dataclasses import dataclass
from math import floor
from operator import index
from time import perf_counter

import numpy as np
from scipy.signal import detrend
from skimage.transform import resize

from your._rust import dedisperse as _rust_dedisperse
from your._rust import dedispersets as _rust_dedispersets
from your.utils.candidate_contract import parse_kill_mask
from your.utils.rfi import sk_sg_filter


@dataclass(frozen=True)
class CandidatePlan:
    """Absolute sample ranges and integer delays shared by CPU and GPU paths."""

    time_size: int
    frequency_size: int
    dmsteps: int
    output_dtype: np.dtype
    time_bin: int
    output_start: int
    native_count: int
    dm_values: np.ndarray
    delays: np.ndarray
    ft_delays: np.ndarray
    required_start: int
    required_stop: int
    read_start: int
    read_stop: int
    nspectra: int
    native_tsamp: float
    nchans: int
    pad_before: int
    pad_after: int


def _positive_index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_delay_bins(dm_values, frequencies, tsamp):
    frequency_term = 1.0 / frequencies**2 - 1.0 / np.max(frequencies) ** 2
    delay_samples = (
        4148808.0 * dm_values[:, None] * frequency_term[None, :] / 1000.0 / tsamp
    )
    if not np.all(np.isfinite(delay_samples)) or np.any(delay_samples < 0):
        raise ValueError("dispersion delays must be finite nonnegative sample offsets")
    rounded = np.rint(delay_samples)
    if np.any(rounded >= float(2**63)):
        raise ValueError("dispersion delay exceeds int64 sample range")
    return rounded.astype(np.int64)


def make_plan(
    cand,
    time_size=256,
    frequency_size=256,
    dmsteps=256,
    output_dtype="float16",
):
    """Create a fixed-shape plan without reading or mutating candidate data."""
    time_size = _positive_index(time_size, "time_size")
    frequency_size = _positive_index(frequency_size, "frequency_size")
    dmsteps = _positive_index(dmsteps, "dmsteps")

    try:
        dtype = np.dtype(output_dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError("output_dtype must be float16 or float32") from exc
    if dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise ValueError("output_dtype must be float16 or float32")

    try:
        dm = float(cand.dm)
        tcand = float(cand.tcand)
        tsamp = float(cand.native_tsamp)
        width = index(cand.width)
        nspectra = index(cand.your_header.nspectra)
        raw_frequencies = np.asarray(cand.chan_freqs)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("candidate is missing valid planning metadata") from exc
    if isinstance(cand.width, (bool, np.bool_)):
        raise ValueError("candidate width must be a nonnegative integer")
    if isinstance(cand.your_header.nspectra, (bool, np.bool_)) or nspectra < 0:
        raise ValueError("candidate nspectra must be a nonnegative integer")
    if width < 0:
        raise ValueError("candidate width must be a nonnegative integer")
    if not np.isfinite(dm) or dm < 0:
        raise ValueError("candidate DM must be finite and nonnegative")
    if not np.isfinite(tcand):
        raise ValueError("candidate time must be finite")
    if not np.isfinite(tsamp) or tsamp <= 0:
        raise ValueError("native_tsamp must be finite and positive")
    if raw_frequencies.ndim != 1 or raw_frequencies.size == 0:
        raise ValueError(
            "candidate chan_freqs must be a nonempty one-dimensional array"
        )
    if raw_frequencies.dtype.kind not in "biuf":
        raise ValueError("candidate channel frequencies must be numeric")
    frequencies = raw_frequencies.astype(np.float64)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("candidate channel frequencies must be finite and positive")

    time_bin = max(1, width // 2)
    native_count = time_size * time_bin
    if native_count > np.iinfo(np.int64).max:
        raise ValueError("native output window exceeds int64 sample range")
    sample_at_candidate = tcand / tsamp
    if not np.isfinite(sample_at_candidate):
        raise ValueError("candidate sample index exceeds integer range")
    center_sample = floor(sample_at_candidate)
    output_start = center_sample - (time_size // 2) * time_bin

    dm_values = np.linspace(0.0, 2.0 * dm, dmsteps, dtype=np.float64)
    if not np.all(np.isfinite(dm_values)):
        raise ValueError("DM trial values must be finite")
    delays = _finite_delay_bins(dm_values, frequencies, tsamp)
    ft_delays = _finite_delay_bins(
        np.array([dm], dtype=np.float64), frequencies, tsamp
    )[0]

    min_delay = int(min(np.min(delays), np.min(ft_delays)))
    max_delay = int(max(np.max(delays), np.max(ft_delays)))
    required_start = output_start + min_delay
    required_stop = output_start + native_count + max_delay
    int64 = np.iinfo(np.int64)
    if not int64.min <= required_start <= int64.max:
        raise ValueError("required sample range exceeds int64 range")
    if not int64.min <= required_stop <= int64.max:
        raise ValueError("required sample range exceeds int64 range")
    if required_stop - required_start > int64.max:
        raise ValueError("required sample window exceeds int64 range")

    read_start = max(0, min(nspectra, required_start))
    read_stop = max(read_start, min(nspectra, required_stop))
    pad_before = max(0, min(required_stop, 0) - required_start)
    pad_after = max(0, required_stop - max(required_start, nspectra))

    return CandidatePlan(
        time_size=time_size,
        frequency_size=frequency_size,
        dmsteps=dmsteps,
        output_dtype=dtype,
        time_bin=time_bin,
        output_start=output_start,
        native_count=native_count,
        dm_values=dm_values,
        delays=delays,
        ft_delays=ft_delays,
        required_start=required_start,
        required_stop=required_stop,
        read_start=read_start,
        read_stop=read_stop,
        nspectra=nspectra,
        native_tsamp=tsamp,
        nchans=int(frequencies.size),
        pad_before=pad_before,
        pad_after=pad_after,
    )


def _numeric_array(data, name):
    array = np.asarray(data)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    return array


def read_plan_data(reader, plan):
    """Read exactly the real, clipped interval required by a candidate plan."""
    count = plan.read_stop - plan.read_start
    if count == 0:
        try:
            dtype = np.dtype(reader.your_header.dtype)
        except (AttributeError, TypeError, ValueError):
            dtype = np.dtype(np.float32)
        if dtype.kind not in "biuf":
            dtype = np.dtype(np.float32)
        return np.empty((0, plan.nchans), dtype=dtype)

    data = _numeric_array(
        reader.get_data(nstart=plan.read_start, nsamp=count), "reader data"
    )
    if data.shape != (count, plan.nchans):
        raise ValueError(
            "reader data must be time-major and match the planned clipped interval"
        )
    return data


def _channel_medians(data, nchans):
    data = np.asarray(data, dtype=np.float64)
    medians = np.zeros(nchans, dtype=np.float64)
    for channel in range(nchans):
        values = data[:, channel]
        values = values[np.isfinite(values)]
        if values.size:
            medians[channel] = np.median(values)
    return medians


def _apply_candidate_masks(data, cand, plan):
    kill_mask = parse_kill_mask(
        getattr(cand, "kill_mask", np.array([False])), plan.nchans
    )
    data[:, kill_mask] = 0

    if getattr(cand, "flag_rfi", False):
        mask = sk_sg_filter(
            data=data,
            your_object=cand,
            tsamp=plan.native_tsamp,
            spectral_kurtosis_sigma=getattr(cand, "spectral_kurtosis_sigma", 4),
            savgol_frequency_window=getattr(cand, "savgol_frequency_window", 15),
            savgol_sigma=getattr(cand, "savgol_sigma", 4),
        )
        if np.asarray(mask).shape != (plan.nchans,):
            raise ValueError("RFI filter returned an invalid channel mask")
        data[:, mask] = 0
        cand.rfi_mask = np.asarray(mask, dtype=bool)


def _finite_float32(data, name):
    array = _numeric_array(data, name).astype(np.float64, copy=False)
    limit = np.finfo(np.float32).max
    array = np.nan_to_num(array, copy=True, nan=0.0, posinf=limit, neginf=-limit)
    np.clip(array, -limit, limit, out=array)
    return array.astype(np.float32)


def product_input_scale(max_abs):
    """Return a common divisor that keeps FP32 product accumulation well scaled."""
    maximum = float(max_abs)
    if np.isfinite(maximum) and maximum > 1e10:
        return maximum
    return 1.0


def _normalize(data, dtype):
    values = np.asarray(data, dtype=np.float64)
    median = np.median(values)
    std = np.std(values)
    if not np.isfinite(std) or std == 0:
        normalized = np.zeros(values.shape, dtype=np.float64)
    else:
        normalized = (values - median) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    limit = np.finfo(dtype).max
    np.clip(normalized, -limit, limit, out=normalized)
    return normalized.astype(dtype)


def finish_products(ft, dmt, plan, profile=None):
    """Resize, detrend, normalize, and cast time-binned CPU/GPU products."""
    ft = _finite_float32(ft, "FT products")
    dmt = _finite_float32(dmt, "DMT products")
    if ft.shape != (plan.time_size, plan.nchans):
        raise ValueError("FT products must have shape (time_size, nchans)")
    if dmt.shape != (plan.dmsteps, plan.time_size):
        raise ValueError("DMT products must have shape (dmsteps, time_size)")

    if profile is not None:
        started = perf_counter()
    ft_frequency_time = ft.T
    if plan.frequency_size == plan.nchans:
        ft_frequency_time = np.array(ft_frequency_time, dtype=np.float32, copy=True)
    else:
        ft_frequency_time = resize(
            ft_frequency_time,
            (plan.frequency_size, plan.time_size),
            anti_aliasing=True,
            mode="constant",
            preserve_range=True,
        ).astype(np.float32, copy=False)
    if profile is not None:
        profile["resize_seconds"] = perf_counter() - started

    if profile is not None:
        started = perf_counter()
    # Center first so constant nonzero bandpasses do not leave float32 detrend noise.
    row_means = np.mean(ft_frequency_time, axis=1, keepdims=True, dtype=np.float64)
    ft_frequency_time = (ft_frequency_time.astype(np.float64) - row_means).astype(
        np.float32
    )
    peak = np.max(np.abs(ft_frequency_time))
    if peak > 1e10:
        # Normalization cancels this scale and avoids float32 least-squares overflow.
        ft_frequency_time = ft_frequency_time / peak
    ft_frequency_time = detrend(ft_frequency_time, axis=1, type="linear").astype(
        np.float32, copy=False
    )
    if profile is not None:
        profile["detrend_seconds"] = perf_counter() - started

    if profile is not None:
        started = perf_counter()
    ft_finished = _normalize(ft_frequency_time.T, plan.output_dtype)
    dmt_finished = _normalize(dmt, plan.output_dtype)
    if profile is not None:
        profile["normalize_seconds"] = perf_counter() - started
    return ft_finished, dmt_finished


def process_cpu(cand, plan, data=None, profile=None):
    """Build finished products with direct indexing and native FP32 accumulation."""
    if profile is not None:
        started = perf_counter()
    if data is None:
        data = read_plan_data(cand, plan)
    data = _numeric_array(data, "candidate data")
    expected_shape = (plan.read_stop - plan.read_start, plan.nchans)
    if data.shape != expected_shape:
        raise ValueError("candidate data must cover exactly the planned clipped range")
    if profile is not None:
        profile["read_seconds"] = perf_counter() - started

    if profile is not None:
        started = perf_counter()
    real = data.astype(np.float64, copy=False)
    medians = _channel_medians(real, plan.nchans)
    logical = np.empty(
        (plan.required_stop - plan.required_start, plan.nchans), dtype=np.float64
    )
    logical[:] = medians
    finite_real = np.where(np.isfinite(real), real, medians[None, :])
    overlap_start = max(plan.required_start, plan.read_start)
    overlap_stop = min(plan.required_stop, plan.read_stop)
    if overlap_stop > overlap_start:
        dst_start = overlap_start - plan.required_start
        src_start = overlap_start - plan.read_start
        logical[dst_start : dst_start + overlap_stop - overlap_start] = finite_real[
            src_start : src_start + overlap_stop - overlap_start
        ]
    _apply_candidate_masks(logical, cand, plan)

    scale_started = perf_counter() if profile is not None else None
    maximum = float(np.max(np.abs(logical))) if logical.size else 0.0
    input_scale = product_input_scale(maximum)
    if input_scale != 1.0:
        logical /= input_scale
    if profile is not None:
        profile["scale_seconds"] = perf_counter() - scale_started

    source_offset = plan.output_start - plan.required_start
    all_delays = np.concatenate((plan.delays.ravel(), plan.ft_delays))
    min_delay = int(all_delays.min())
    max_delay = int(all_delays.max())
    source_start = source_offset + min_delay
    source_stop = source_offset + plan.native_count + max_delay
    if source_offset < 0 or source_start < 0 or source_stop > len(logical):
        raise ValueError("planned source samples escape the padded logical input")
    if profile is not None:
        profile["prepare_seconds"] = perf_counter() - started

    if profile is not None:
        started = perf_counter()
    ft_native = _rust_dedisperse(
        logical,
        np.asarray(-plan.ft_delays, dtype=np.int64),
        source_offset,
        source_offset + plan.native_count,
    )
    dmt_native = np.empty((plan.dmsteps, plan.native_count), dtype=np.float32)
    for dm_index, dm_delays in enumerate(plan.delays):
        dmt_native[dm_index] = _rust_dedispersets(
            logical,
            np.asarray(-dm_delays, dtype=np.int64),
            source_offset,
            source_offset + plan.native_count,
        )
    if profile is not None:
        profile["dedisp_seconds"] = perf_counter() - started

    if profile is not None:
        started = perf_counter()
    ft = ft_native.reshape(plan.time_size, plan.time_bin, plan.nchans).mean(
        axis=1, dtype=np.float32
    )
    dmt_native /= np.float32(plan.nchans)
    dmt = dmt_native.reshape(plan.dmsteps, plan.time_size, plan.time_bin).mean(
        axis=2, dtype=np.float32
    )
    ft = _finite_float32(ft, "FT products")
    dmt = _finite_float32(dmt, "DMT products")
    if profile is not None:
        profile["bin_seconds"] = perf_counter() - started

    return finish_products(ft, dmt, plan, profile=profile)
