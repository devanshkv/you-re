# Candidate processing and profiling

Candidate creation uses direct sample indices, with `tcand` referenced to the
highest-frequency channel. The input includes real samples needed by every DM
trial and width bin. Only samples outside the observation are median-padded;
samples never wrap from the other end of the chunk. The default DM grid is 256
trials from zero to twice the candidate DM. `--dmsteps` changes the number of
trials without changing the range.

The CSV width remains log2(native samples). Time bins average
`max(1, width_in_native_samples // 2)` native samples. DM–time averages channels,
including masked channels as zeros in the original channel count. Frequency–time
is frequency-resized with anti-aliasing, then linearly detrended along the full
output time axis per channel. Each image is independently median-centered and
scaled by its standard deviation. Constant images become zeros. Finished arrays
and HDF5 datasets default to FP16; `--output_dtype float32` retains FP32 output.

## Install and run

The Rust extension is required for CPU processing. GPU processing is optional:

```bash
python -m pip install -e '.[test,gpu]'
CUDA_PATH=/usr/local/cuda-12.9 python bin/your_candmaker.py \
  -c /data/candidates.csv -o /data/products -g 0 \
  --profile_json /data/products/profile.json
```

The GPU extra selects CuPy 13 with CUDA 12. The development configuration is
Python 3.12, CuPy 13.6, CUDA 12.9 and a 16 GB V100. CUDA installation location and
CuPy wheel support must match the target interpreter; CPU-only installation does
not import CuPy or Numba. Numba is no longer a dependency.

Use `-g -1` for CPU. Multiple explicit IDs such as `-g 2 3` start one independent
spawned worker per device. The implementation allows duplicate observation
coverage on different GPUs; multi-GPU hardware validation is deferred.

## Scheduling and memory

The CLI retains compact planning metadata rather than a full DM-by-channel
delay table for every candidate. It closes each observation reader after planning;
workers build full delay tables only for active candidates. Jobs are grouped by
resolved observation file sets and sorted by required input interval. Each GPU retains immutable channel-major raw samples and uploads only
missing coverage as its resident window advances. Input uploads use contiguous
pinned time-major chunks followed by a bit-preserving GPU corner turn into the
channel-major cache. Candidate-specific medians,
kill masks and optional RFI masks do not change with batch composition.

`--gpu_inflight` bounds simultaneous candidate processing (default 2).
`--gpu_memory_mb` bounds the estimated working set; the default reserves a
fraction of currently free memory. Estimates include resident input, old plus
new buffers during overlap reuse, bounded GPU transpose staging, native
intermediates, delay tables, median
workspace and all in-flight candidates. HDF5 writes use a bounded CPU writer
queue and can overlap later GPU work. If only the old-plus-new cache replacement
exceeds the budget, the worker releases the old cache and reloads; this does not
force an otherwise fitting candidate into streaming. Input storage and memory
estimates use the decoded sample dtype, including calibrated PSRFITS floats.
Transpose staging is capped at 64 MiB, shrinks to fit the budget, and is released
before candidate workspaces are allocated. The `cache_upload_s` profile counter
includes both the raw transfer and GPU corner turn.

A singleton exceeding the memory budget uses disk-backed host staging, exact
host padding statistics, and bounded GPU channel/DM/time tiles. Channel partials
are accumulated in FP32 before averaging. Bin alignment is unchanged, and
normalization happens only after complete images are assembled. This path trades
repeated transfers for bounded GPU memory; it does not switch dedispersion to
CPU. Budgets below 64 KiB cannot run the streaming path. Host disk space must fit
the candidate's real input coverage.

GPU input sampling, shared-memory DM tiling and time binning use CuPy CUDA
kernels. Final resize, detrending and normalization currently stay on CPU, as
does existing optional RFI mask calculation. These stages are individually timed.
There is no cuCIM dependency: move another stage only when full-pipeline
measurements justify the dependency and transfer/synchronization costs.

## Benchmark against the unchanged checkout

Keep the baseline checkout unchanged and run both versions through the same
Python environment (the baseline additionally needs Numba and its compiled Rust
extension). Use distinct output directories so products cannot overwrite each
other:

```bash
CUDA_PATH=/usr/local/cuda-12.9 python checks/benchmark_candidates.py \
  --csv /data/candidates.csv --baseline /home/riftuser/you-re \
  --output /data/benchmarks/cupy --gpu-ids 0 --repeat 3
```

The harness alternates old/new execution order, runs them in separate processes,
and records end-to-end seconds, throughput, failures and stage timings. New runs
also record actual decoded and uploaded input bytes. `--limit` selects a prefix;
`--inflight` and `--memory-mb` exercise concurrency and streaming. Detailed new
profiles are in each run's `profile.json`; `summary.json` contains run summaries.
Wall throughput includes interpreter startup, imports, worker/context setup,
compilation, transfers, image processing and writing. Worker-stage sums can
overlap and must not be added to infer wall time. First-use CUDA timings can
include JIT and allocation effects. Do not infer a real-data speedup from tiny
fixture workloads.

## Output compatibility

Default filenames include a stable hash of the observation and full candidate
parameters so different widths, labels, and observations do not overwrite each
other. Scheduled repeated rows receive distinct names, and destination collisions
are checked before processing. HDF5 files are written to temporary files in the
destination directory and atomically published only after a successful close;
a failed save preserves any previous complete file.

Dataset names and axis order remain `data_freq_time[time, frequency]` and
`data_dm_time[DM, time]`. New files record processing version, native/output
sampling intervals, exact DM trials, required sample coverage and padding.
Plotting recognizes finished products and skips repeated detrending and
normalization. Legacy files retain the previous plotting behavior. Output
validation uses dataset shape, dtype and finite values rather than a minimum
compressed file size.

The new DM–time mean differs in scale from the old channel sum. Final normalized
images cancel that constant scaling. Public `dedispersets()` continues to return
a channel sum. CPU/GPU agreement is numerical at FP16 accuracy, not bitwise.
DM optimization is outside the GPU migration; its CLI flag is rejected on GPU.
