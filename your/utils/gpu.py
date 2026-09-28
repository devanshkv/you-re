import logging
import math
import subprocess

import numba
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError

logger = logging.getLogger(__name__)


def gpu_dedisperse(cand, device=0):
    """

    GPU dedispersion (by rolling the array)

    Args:
        cand: Candidate instance
        device (int): GPU ID

    Returns:
        candidate object

    """
    cuda.select_device(device)
    chan_freqs = cuda.to_device(np.array(cand.chan_freqs, dtype=np.float32))
    cand_data_in = cuda.to_device(np.array(cand.data.T, dtype=cand.your_header.dtype))
    cand_data_out = cuda.to_device(
        np.zeros_like(cand.data.T, dtype=cand.your_header.dtype)
    )

    @cuda.jit
    def gpu_dedisp(cand_data_in, chan_freqs, dm, cand_data_out, tsamp):
        ii, jj = cuda.grid(2)
        if ii < cand_data_in.shape[0] and jj < cand_data_in.shape[1]:
            disp_time = int(
                round(
                    -4148808.0
                    * dm
                    * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2)
                    / 1000
                    / tsamp
                )
            )
            cand_data_out[ii, jj] = cand_data_in[
                ii, (jj + disp_time) % cand_data_in.shape[1]
            ]

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(cand_data_in.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(cand_data_in.shape[1] / threadsperblock[1])

    blockspergrid = (blockspergrid_x, blockspergrid_y)

    gpu_dedisp[blockspergrid, threadsperblock](
        cand_data_in,
        chan_freqs,
        float(cand.dm),
        cand_data_out,
        float(cand.your_header.tsamp),
    )

    cand.dedispersed = cand_data_out.copy_to_host().T

    return cand


@cuda.jit
def dmt_channels(cand_data_in, delays, cand_data_out):
    """
    One thread per (DM, sample), summing the channels in a register.

    Args:
        cand_data_in: (nchans, nsamples) chunk, channel major
        delays: (ndm, nchans) sample shift per DM and channel
        cand_data_out: (ndm, nsamples) DM-time plane

    """
    jj, kk = cuda.grid(2)
    nsamples = cand_data_in.shape[1]
    if jj < nsamples and kk < cand_data_out.shape[0]:
        acc = 0
        for ii in range(cand_data_in.shape[0]):
            # a negative shift wraps, Python modulo semantics, as numba gives
            acc += cand_data_in[ii, (jj + delays[kk, ii]) % nsamples]
        cand_data_out[kk, jj] = acc


@cuda.jit
def dmt_runs(cumsum, starts, stops, delays, nruns, cand_data_out):
    """
    As `dmt_channels`, but each span of channels sharing a delay bin is one
    subtraction of the band cumulative sum rather than a loop over its channels.

    Args:
        cumsum: (nchans + 1, nsamples) cumulative sum down the channel axis
        starts: (ndm, nruns) first channel of each span
        stops: (ndm, nruns) one past the last channel of each span
        delays: (ndm, nruns) sample shift of each span
        nruns: (ndm,) spans this DM actually has, the rest being padding
        cand_data_out: (ndm, nsamples) DM-time plane

    """
    jj, kk = cuda.grid(2)
    nsamples = cumsum.shape[1]
    if jj < nsamples and kk < cand_data_out.shape[0]:
        acc = 0
        for rr in range(nruns[kk]):
            idx = (jj + delays[kk, rr]) % nsamples
            acc += cumsum[stops[kk, rr], idx] - cumsum[starts[kk, rr], idx]
        cand_data_out[kk, jj] = acc


@cuda.jit
def band_cumsum(cand_data_in, cumsum):
    """
    Cumulative sum down the channel axis, one thread per sample, with a leading
    zero row so any contiguous span of channels sums as one subtraction.

    Args:
        cand_data_in: (nchans, nsamples) chunk, channel major
        cumsum: (nchans + 1, nsamples) output, row 0 already zeroed

    """
    jj = cuda.grid(1)
    if jj < cand_data_in.shape[1]:
        acc = 0
        for ii in range(cand_data_in.shape[0]):
            acc += cand_data_in[ii, jj]
            cumsum[ii + 1, jj] = acc


def delay_table(chan_freqs, tsamp, dms):
    """
    Sample shift per DM and channel. float64, so it rounds the way the CPU path
    does; the delays used to be worked out in the kernel in float32, where a few
    channels per plane land in a neighbouring bin.

    Args:
        chan_freqs (numpy.ndarray): channel frequencies, MHz
        tsamp (float): sampling time, seconds
        dms (numpy.ndarray): DMs to make the plane over

    Returns:
        numpy.ndarray: (ndm, nchans) shift in samples

    """
    delay = (
        4148808.0
        * dms[:, None]
        * (1 / chan_freqs[0] ** 2 - 1 / chan_freqs[None, :] ** 2)
        / 1000
        / tsamp
    )
    return -np.round(delay).astype(np.int32)


def run_edges(delays):
    """
    Where each span of equal delay starts, and how many spans each DM has.

    Args:
        delays (numpy.ndarray): (ndm, nchans) shift in samples

    Returns:
        tuple: (ndm, nchans) bool of span starts, (ndm,) span counts

    """
    edge = np.empty(delays.shape, dtype=bool)
    edge[:, 0] = True
    np.not_equal(delays[:, 1:], delays[:, :-1], out=edge[:, 1:])
    return edge, edge.sum(axis=1).astype(np.int32)


def run_table(delays, edge, nruns):
    """
    Lay the spans out per DM, padded to the DM with the most of them.

    Args:
        delays (numpy.ndarray): (ndm, nchans) shift in samples
        edge (numpy.ndarray): (ndm, nchans) bool of span starts
        nruns (numpy.ndarray): (ndm,) span counts

    Returns:
        tuple: starts, stops, span delays, each (ndm, max spans)

    """
    ndm, nchans = delays.shape
    width = int(nruns.max())
    kk, ii = np.nonzero(edge)
    rr = np.arange(kk.size) - np.repeat(np.cumsum(nruns) - nruns, nruns)

    starts = np.zeros((ndm, width), dtype=np.int32)
    run_delays = np.zeros((ndm, width), dtype=np.int32)
    starts[kk, rr] = ii
    run_delays[kk, rr] = delays[kk, ii]

    stops = np.zeros((ndm, width), dtype=np.int32)
    stops[:, :-1] = starts[:, 1:]
    stops[np.arange(ndm), nruns - 1] = nchans
    return starts, stops, run_delays


def gpu_dmt(cand, device=0, dmsteps=256, max_run_fraction=0.6):
    """

    GPU DM-Time bow-tie

    Two kernels, identical output. `dmt_runs` sums each span of channels sharing
    a delay bin as one subtraction of a band cumulative sum, so it wins while the
    band still collapses; once more than `max_run_fraction` of the channels earn
    their own bin it reads four bytes per span where `dmt_channels` reads one
    byte per channel, and loses. The default crossover was measured on a T4.

    Args:
        cand: Candidate instance
        device (int): GPU ID
        dmsteps (int): rows in the DM-time plane
        max_run_fraction (float): spans per channel above which to take
            `dmt_channels`

    Returns:
        candidate object

    """
    cuda.select_device(device)
    nsamples, nchans = cand.data.shape
    tsamp = float(cand.your_header.tsamp)
    chan_freqs = np.asarray(cand.chan_freqs, dtype=np.float64)
    # the same expression the CPU path uses, so the DM axes are bit identical
    dms = cand.dm + np.linspace(-cand.dm, cand.dm, dmsteps)

    delays = delay_table(chan_freqs, tsamp, dms)
    # counting the spans is cheap; only lay them out if we are going to use them
    edge, nruns = run_edges(delays)
    fraction = float(nruns.mean()) / nchans

    cand_data_in = cuda.to_device(np.ascontiguousarray(cand.data.T))
    dmt_return = cuda.device_array((dmsteps, nsamples), dtype=np.float32)
    threads = 128
    blocks = math.ceil(nsamples / threads)

    if fraction <= max_run_fraction:
        logger.debug(f"{fraction:.2f} spans per channel, summing runs")
        # int32 holds the whole band for any real filterbank; widen if it cannot
        info = np.iinfo(cand.data.dtype)
        bound = nchans * max(abs(int(info.min)), int(info.max))
        acc_dtype = np.int32 if bound <= np.iinfo(np.int32).max else np.int64
        starts, stops, run_delays = run_table(delays, edge, nruns)
        cumsum = cuda.device_array((nchans + 1, nsamples), dtype=acc_dtype)
        cumsum[0].copy_to_device(np.zeros(nsamples, dtype=acc_dtype))
        band_cumsum[blocks, threads](cand_data_in, cumsum)
        dmt_runs[(blocks, dmsteps), (threads, 1)](
            cumsum,
            cuda.to_device(starts),
            cuda.to_device(stops),
            cuda.to_device(run_delays),
            cuda.to_device(nruns),
            dmt_return,
        )
    else:
        logger.debug(f"{fraction:.2f} spans per channel, summing channels")
        dmt_channels[(blocks, dmsteps), (threads, 1)](
            cand_data_in, cuda.to_device(delays), dmt_return
        )

    cand.dmt = dmt_return.copy_to_host()

    return cand


@cuda.jit
def transpose_u8(src, dst, row0):
    """
    Rows `row0 .. row0 + len(src)` of a (nsamples, nchans) chunk into columns
    of the (nchans, nsamples) chunk, through a shared-memory tile so both the
    read and the write are coalesced. Launch with 32x8 threads.

    Args:
        src: (rows, nchans) uint8 slice of the chunk as read
        dst: (nchans, nsamples) uint8
        row0: sample the slice starts at

    """
    tile = cuda.shared.array((32, 33), dtype=numba.uint8)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    bx, by = cuda.blockIdx.x * 32, cuda.blockIdx.y * 32
    for k in range(0, 32, 8):
        r, c = by + ty + k, bx + tx
        if r < src.shape[0] and c < src.shape[1]:
            tile[ty + k, tx] = src[r, c]
    cuda.syncthreads()
    for k in range(0, 32, 8):
        r, c = bx + ty + k, by + tx
        if r < dst.shape[0] and c < src.shape[0]:
            dst[r, row0 + c] = tile[tx, ty + k]


def portable_pinned_array(size):
    """
    A uint8 host array page-locked for every CUDA context in the process, not
    only the current one, so a worker can upload from it to any GPU.

    Args:
        size (int): bytes

    Returns:
        numpy.ndarray: page-locked uint8 array
    """
    memory = cuda.current_context().memhostalloc(size, portable=True)
    return np.ndarray(size, dtype=np.uint8, buffer=memory)


class PinnedReadBuffer:
    """
    One page-locked host buffer, reused for every chunk read in this process
    and grown when a chunk is larger than any before it. Reading a chunk into
    it lets the upload to the GPU go straight over the bus, where from
    pageable memory the driver first copies each slice into its own staging
    buffer. It is locked for every GPU, so a worker whose candidates go to
    different GPUs keeps the one buffer.

    Each read overwrites the last, so a chunk read into it is only good until
    the next read: meant for a worker that makes one candidate at a time.

    Args:
        device (int): GPU whose context allocates the buffer
        granule (int): allocation size is rounded up to a multiple of this
    """

    def __init__(self, device=0, granule=64 * 2**20):
        self.device = device
        self.granule = granule
        self.buffer = None

    def __call__(self, nbytes):
        if self.buffer is None or self.buffer.size < nbytes:
            cuda.select_device(self.device)
            self.buffer = None
            size = -(-nbytes // self.granule) * self.granule
            try:
                self.buffer = portable_pinned_array(size)
            except CudaAPIError:
                # more than the host will lock: read this chunk into ordinary
                # memory, as without a buffer, and try again next time
                logger.warning(f"Could not pin {size} bytes, reading unpinned")
                return np.empty(nbytes, dtype=np.uint8)
        return self.buffer


def to_device_channel_major(data, stream, staging_bytes=128 * 2**20):
    """
    Put a (nsamples, nchans) uint8 chunk on the device as (nchans, nsamples),
    transposing there a slice at a time. The device holds the chunk once plus
    one `staging_bytes` slice, where a whole-chunk transpose would hold it
    twice: four candmaker workers on 1 GB chunks then fit in 8 GB.

    Args:
        data: (nsamples, nchans) uint8 chunk
        stream: CUDA stream to copy and transpose on
        staging_bytes (int): size of the reused slice buffer

    Returns:
        numba.cuda.cudadrv.devicearray.DeviceNDArray: (nchans, nsamples) chunk
    """
    nsamples, nchans = data.shape
    out = cuda.device_array((nchans, nsamples), dtype=np.uint8, stream=stream)
    step = max(32, staging_bytes // nchans // 32 * 32)
    staging = cuda.device_array(
        (min(step, nsamples), nchans), dtype=np.uint8, stream=stream
    )
    for row0 in range(0, nsamples, step):
        rows = min(step, nsamples - row0)
        part = staging[:rows]
        part.copy_to_device(
            np.ascontiguousarray(data[row0 : row0 + rows]), stream=stream
        )
        transpose_u8[(math.ceil(nchans / 32), math.ceil(rows / 32)), (32, 8), stream](
            part, out, row0
        )
    # stream order already keeps each slice's copy behind the previous
    # transpose; wait here so the staging buffer is free to release
    stream.synchronize()
    return out


@cuda.jit
def dedisp_window(data, delays, fdf, tdf, col0, zero, out):
    """
    Dedispersed frequency-time plane, decimated and cropped in one pass: one
    thread per kept output cell, summing its `fdf` channels by `tdf` samples in
    a register. Nothing is computed for the columns the crop throws away.

    Args:
        data: (nchans, nsamples) chunk, channel major
        delays: (nchans,) shift per channel, already wrapped into [0, nsamples)
        fdf: channels per output row
        tdf: samples per output column
        col0: first decimated column kept; negative wraps, as the crop did
        zero: one-element array whose dtype the sum accumulates in
        out: (nchans // fdf, width) output window

    """
    col, row = cuda.grid(2)
    nsamples = data.shape[1]
    ncols = nsamples // tdf
    if row < out.shape[0] and col < out.shape[1]:
        acc = zero[0]
        base = ((col0 + col) % ncols) * tdf
        for ch in range(row * fdf, row * fdf + fdf):
            d = delays[ch]
            for s in range(base, base + tdf):
                idx = s + d
                if idx >= nsamples:
                    idx -= nsamples
                acc += data[ch, idx]
        out[row, col] = acc


@cuda.jit
def dmt_window(data, delays, tdf, col0, zero, out):
    """
    DM-time plane, decimated and cropped in one pass: one thread per kept
    (DM, column) cell summing every channel over its `tdf` samples in a
    register, in place of one atomic add per channel and sample. Threads of a
    warp share a column across neighbouring DMs, whose delays barely differ,
    so their reads land on the same few cache lines.

    Args:
        data: (nchans, nsamples) chunk, channel major
        delays: (nchans, ndm) shift per channel and DM, wrapped into [0, nsamples)
        tdf: samples per output column
        col0: first decimated column kept; negative wraps, as the crop did
        zero: one-element array whose dtype the sum accumulates in
        out: (ndm, width) output window

    """
    kk, col = cuda.grid(2)
    nchans, nsamples = data.shape
    ncols = nsamples // tdf
    if kk < out.shape[0] and col < out.shape[1]:
        acc = zero[0]
        base = ((col0 + col) % ncols) * tdf
        for ch in range(nchans):
            d = delays[ch, kk]
            for s in range(base, base + tdf):
                idx = s + d
                if idx >= nsamples:
                    idx -= nsamples
                acc += data[ch, idx]
        out[kk, col] = acc


# samples per block of the block prefix sums
BLOCK = 32


@cuda.jit
def block_sums(data, csum):
    """
    Sum of each channel over each block of `BLOCK` samples: one thread per
    (channel, block), neighbouring threads on neighbouring blocks. Written one
    column right, so `block_prefix` can turn the rows into exclusive prefix sums.

    Args:
        data: (nchans, nsamples) chunk, channel major
        csum: (nchans, nsamples // BLOCK + 1) output; column 0 is not written

    """
    b, ch = cuda.grid(2)
    if ch < data.shape[0] and b < csum.shape[1] - 1:
        acc = csum[0, 0] - csum[0, 0]
        for s in range(b * BLOCK, b * BLOCK + BLOCK):
            acc += data[ch, s]
        csum[ch, b + 1] = acc


@cuda.jit
def block_prefix(csum):
    """
    Running sum along each channel's blocks, one thread per channel, so
    `csum[ch, b]` is the sum of the channel's first `b * BLOCK` samples.

    Args:
        csum: (nchans, nblocks + 1) block sums from `block_sums`

    """
    ch = cuda.grid(1)
    if ch < csum.shape[0]:
        acc = csum[ch, 0] - csum[ch, 0]
        csum[ch, 0] = acc
        for b in range(1, csum.shape[1]):
            acc += csum[ch, b]
            csum[ch, b] = acc


@cuda.jit(device=True, inline=True)
def span_sum(data, csum, ch, lo, hi, acc):
    """
    Add samples `lo .. hi` of one channel, 0 <= lo <= hi <= nsamples: the
    blocks wholly inside the span as one difference of the prefix sums, and
    the at most `2 * BLOCK - 2` samples either side of them one by one.
    """
    bl = (lo + BLOCK - 1) // BLOCK
    bh = hi // BLOCK
    if bl < bh:
        for s in range(lo, bl * BLOCK):
            acc += data[ch, s]
        acc += csum[ch, bh] - csum[ch, bl]
        for s in range(bh * BLOCK, hi):
            acc += data[ch, s]
    else:
        for s in range(lo, hi):
            acc += data[ch, s]
    return acc


@cuda.jit(device=True, inline=True)
def wrapped_span_sum(data, csum, ch, start, tdf, acc):
    """
    Add `tdf` samples of one channel from `start`, 0 <= start < 2 * nsamples,
    wrapping at the end of the chunk as the per-sample kernels do.
    """
    nsamples = data.shape[1]
    if start >= nsamples:
        start -= nsamples
    stop = start + tdf
    if stop <= nsamples:
        return span_sum(data, csum, ch, start, stop, acc)
    acc = span_sum(data, csum, ch, start, nsamples, acc)
    return span_sum(data, csum, ch, 0, stop - nsamples, acc)


@cuda.jit
def dedisp_window_blocks(data, csum, delays, fdf, tdf, col0, zero, out):
    """
    `dedisp_window`, adding each channel's `tdf` samples through the block
    prefix sums: the same integer sums in about `2 * BLOCK` reads, not `tdf`.
    """
    col, row = cuda.grid(2)
    ncols = data.shape[1] // tdf
    if row < out.shape[0] and col < out.shape[1]:
        acc = zero[0]
        base = ((col0 + col) % ncols) * tdf
        for ch in range(row * fdf, row * fdf + fdf):
            acc = wrapped_span_sum(data, csum, ch, base + delays[ch], tdf, acc)
        out[row, col] = acc


@cuda.jit
def dmt_window_blocks(data, csum, delays, tdf, col0, zero, out):
    """
    `dmt_window`, adding each channel's `tdf` samples through the block
    prefix sums: the same integer sums in about `2 * BLOCK` reads, not `tdf`.
    """
    kk, col = cuda.grid(2)
    nchans, nsamples = data.shape
    ncols = nsamples // tdf
    if kk < out.shape[0] and col < out.shape[1]:
        acc = zero[0]
        base = ((col0 + col) % ncols) * tdf
        for ch in range(nchans):
            acc = wrapped_span_sum(data, csum, ch, base + delays[ch, kk], tdf, acc)
        out[kk, col] = acc


@cuda.jit
def dedisp_delays(chan_freqs, dm, tsamp, delays):
    """
    Per-channel shift at the candidate's DM, written exactly as the atomic
    dedispersion kernel computed it in-line, so every channel lands in the same
    sample it did: float32 frequencies, float64 DM and sampling time.

    Args:
        chan_freqs: (nchans,) float32 channel frequencies, MHz
        dm (float): dispersion measure
        tsamp (float): sampling time, seconds
        delays: (nchans,) int64 output, unwrapped

    """
    ii = cuda.grid(1)
    if ii < chan_freqs.shape[0]:
        disp_time = int(
            round(
                -4148808.0
                * dm
                * (1 / (chan_freqs[0]) ** 2 - 1 / (chan_freqs[ii]) ** 2)
                / 1000
                / tsamp
            )
        )
        delays[ii] = disp_time


def dmt_delays(cand, dms):
    """
    Sample shift per channel for each DM of the bow-tie, rounded exactly as
    before (float64 frequencies, numpy rounding).

    Args:
        cand: Candidate instance
        dms (numpy.ndarray): DMs, float64

    Returns:
        numpy.ndarray: (nchans, ndm) int64 shift in samples, unwrapped
    """
    freqs = np.asarray(cand.chan_freqs)
    return np.round(
        -1
        * 4148808.0
        * dms[None, :]
        * (1 / (freqs[0]) ** 2 - 1 / (freqs[:, None]) ** 2)
        / 1000
        / cand.your_header.tsamp
    ).astype(np.int64)


def accumulator(dtype, count):
    """
    Narrowest exact accumulator for `count` samples of `dtype`: int32 while the
    sum cannot overflow it, int64 past that, float64 for float data.

    Args:
        dtype: data dtype
        count (int): samples summed into one output cell

    Returns:
        numpy.ndarray: one-element zero of the accumulator dtype
    """
    if not np.issubdtype(dtype, np.integer):
        return np.zeros(1, dtype=np.float64)
    info = np.iinfo(dtype)
    bound = count * max(abs(int(info.min)), int(info.max))
    return np.zeros(1, dtype=np.int32 if bound <= np.iinfo(np.int32).max else np.int64)


def gpu_dedisp_and_dmt_crop(cand, device=0, width=256, block_from=4 * BLOCK):
    """

    GPU dedispersion and DM-time bow-tie, decimated to 256 channels and by
    half the pulse width in time, and cropped to the central `width` columns.

    Each output cell is summed by one thread, so there are no atomic adds, and
    only the kept window is computed. The result is the same as decimating the
    whole chunk and then cropping it, as this function used to.

    From `block_from` samples per column, integer data is summed through
    per-channel prefix sums over blocks of `BLOCK` samples, so a cell costs
    about `2 * BLOCK` reads per channel rather than one per sample. Integer
    sums either way, so the planes are identical.

    Args:
        cand: Candidate instance
        device (int): GPU ID
        width (int): columns kept around the centre
        block_from (int): samples per column from which to use the block
            prefix sums

    Returns:
        candidate object

    """
    if cand.width < 3:
        time_decimation_factor = 1
    else:
        time_decimation_factor = cand.width // 2

    nsamples, nchans = cand.data.shape
    if nchans < 256:
        raise IndexError("GPU candmaker will not work if nchans < 256.")

    frequency_decimation_factor = nchans // 256
    fdf, tdf = int(frequency_decimation_factor), int(time_decimation_factor)
    rows = nchans // fdf
    col0 = nsamples // tdf // 2 - width // 2

    logger.debug(f"Freq decimation factor: {fdf}")
    logger.debug(f"Time decimation factor: {tdf}")

    cuda.select_device(device)
    stream = cuda.stream()

    # the chunk arrives (nsamples, nchans) in whatever dtype get_chunk left it
    # (float64 where it padded with the median); cast as before, then put it
    # channel major, on the device where that is quick
    data = np.asarray(cand.data).astype(cand.your_header.dtype, copy=False)
    if data.dtype == np.uint8:
        cand_data_in = to_device_channel_major(data, stream)
    else:
        cand_data_in = cuda.to_device(np.ascontiguousarray(data.T), stream=stream)

    ft_delays = cuda.device_array(nchans, dtype=np.int64, stream=stream)
    dedisp_delays[math.ceil(nchans / 128), 128, stream](
        cuda.to_device(np.array(cand.chan_freqs, dtype=np.float32), stream=stream),
        float(cand.dm),
        float(cand.your_header.tsamp),
        ft_delays,
    )
    ft_host = ft_delays.copy_to_host(stream=stream)
    stream.synchronize()
    # wrapped here so the kernels need one compare, not a 64-bit modulo
    ft_delays = cuda.to_device(
        (ft_host % nsamples).astype(np.int32),
        stream=stream,
    )
    delays = dmt_delays(cand, np.linspace(0, 2 * cand.dm, 256))
    dmt_delay_table = cuda.to_device(
        np.ascontiguousarray(delays % nsamples).astype(np.int32), stream=stream
    )

    ft_out = cuda.device_array((rows, width), dtype=np.float32, stream=stream)
    dmt_out = cuda.device_array((256, width), dtype=np.float32, stream=stream)
    ft_zero = cuda.to_device(accumulator(data.dtype, fdf * tdf), stream=stream)
    dmt_zero = cuda.to_device(accumulator(data.dtype, nchans * tdf), stream=stream)

    logger.debug("Allocated arrays on the GPU")

    ft_grid = (math.ceil(width / 32), math.ceil(rows / 8)), (32, 8), stream
    dmt_grid = (math.ceil(256 / 32), math.ceil(width / 4)), (32, 4), stream
    nblocks = nsamples // BLOCK
    if tdf >= block_from and nblocks and np.issubdtype(data.dtype, np.integer):
        logger.debug(f"Summing blocks of {BLOCK} samples")
        # int32 prefix sums while a whole channel cannot overflow them
        csum = cuda.device_array(
            (nchans, nblocks + 1),
            dtype=accumulator(data.dtype, nsamples).dtype,
            stream=stream,
        )
        block_sums[(math.ceil(nblocks / 32), math.ceil(nchans / 4)), (32, 4), stream](
            cand_data_in, csum
        )
        block_prefix[math.ceil(nchans / 128), 128, stream](csum)
        dedisp_window_blocks[ft_grid](
            cand_data_in, csum, ft_delays, fdf, tdf, col0, ft_zero, ft_out
        )
        dmt_window_blocks[dmt_grid](
            cand_data_in, csum, dmt_delay_table, tdf, col0, dmt_zero, dmt_out
        )
    else:
        csum = None
        dedisp_window[ft_grid](cand_data_in, ft_delays, fdf, tdf, col0, ft_zero, ft_out)
        dmt_window[dmt_grid](
            cand_data_in, dmt_delay_table, tdf, col0, dmt_zero, dmt_out
        )
    cand.dedispersed = ft_out.copy_to_host(stream=stream).T
    cand.dmt = dmt_out.copy_to_host(stream=stream)
    stream.synchronize()

    logger.debug("cand.dedispersed and cand.dmt set!")

    # numba defers frees until 10 arrays or 20% of the card are pending, and a
    # wide candidate's chunk is under that, so it would stay allocated into the
    # next candidate. One process retries after flushing, but candmaker's
    # workers cannot flush each other, so two chunks per worker ran the card
    # out of memory. Free them now.
    del (
        cand_data_in,
        csum,
        ft_delays,
        dmt_delay_table,
        ft_out,
        dmt_out,
        ft_zero,
        dmt_zero,
    )
    pending_frees().clear()
    return cand


def pending_frees():
    """
    The queue numba holds freed device arrays in until it flushes them. Since
    the external memory manager interface it belongs to the memory manager;
    `Context.deallocations` is a separate, unused queue there.

    Returns:
        numba's pending deallocations for the current context
    """
    ctx = cuda.current_context()
    return getattr(ctx.memory_manager, "deallocations", ctx.deallocations)


def get_gpu_memory_map(gpu_id):
    """
    Get the current gpu free memory

    Args:
        gpu_id (int): GPU id

    Returns:
        int: amount of free GPU RAM
    """
    cmd_list = [
        "nvidia-smi",
        "-i",
        f"{gpu_id}",
        "--query-gpu=memory.free",
        "--format=csv,nounits,noheader",
    ]
    result = subprocess.check_output(cmd_list)
    return int(result.decode())
