# CuPy pipeline performance

The initial comparison baseline is `cupy-candidates` commit `4db0d60`. A separate
end-to-end comparison against the Numba pipeline on `main` appears below. The
goal is faster execution of the direct, nonwrapping
candidate pipeline, keeping its real-data halo, padding, image finishing,
atomic output, resident cache and memory-bounded streaming.

## Lessons from main

- Compute the time bins that survive into the output, rather than building
  every native-time DM sample and discarding most of them during averaging.
  Prefix sums make the interval sum independent of pulse width after scanning
  the input. This is the largest CPU opportunity.
- Reuse input layout and avoid repeated full-sized copies. Channel-major data
  makes accesses along each channel contiguous; integer data need not always
  pay for median estimation and nonfinite-value replacement.
- Amortize kernel launches. The original CuPy pipeline divides 4096 channels
  into 128 groups and launches separate FT and DMT work for each group.
  Larger groups reduce Python dispatch and CUDA launch overhead.
- Measure the complete pipeline. CPU resizing, finishing and HDF5 compression
  can dominate once GPU arithmetic gets faster.

Main's GPU channel averaging is not a drop-in replacement for the CuPy
pipeline's anti-aliased frequency resize. This change keeps the latter. Main's
multiple-process GPU model also trades away some of the CuPy scheduler's input
cache reuse and memory accounting; it is not adopted without an end-to-end
comparison proving that trade worthwhile.

## CPU results

Measured on 2026-10-18 with Python 3.12.3, NumPy 2.3.5 and a Xeon E5-2680 v4
core, pinned to CPU 0. Each value is the median of three warmed processing
calls using the default synthetic workload below, including preparation and
image finishing but excluding file I/O and HDF5 output.

| Pulse width | Unchanged CuPy CPU path | Optimized CPU path | Speedup |
| --- | ---: | ---: | ---: |
| 4 samples | 1.887 s | 0.488 s | 3.9x |
| 8 samples | 4.807 s | 0.519 s | 9.3x |
| 32 samples | 22.406 s | 0.637 s | 35.1x |
| 128 samples | 88.514 s | 2.040 s | 43.4x |

The first three cases produced identical FP32 images. At width 128 the FT
image remained identical and the largest normalized DMT difference was
`5.42e-5`, due to the changed summation order. These are measurements of this
synthetic workload, not the original 380-candidate observation.

The fast path supports bounded integer input. Ordinary unpadded candidates
without RFI filtering use the raw integer array and an enabled-channel mask,
avoiding the padded float64 allocation and median calculation. Padded or
RFI-filtered integer candidates can use prepared-input prefix sums. Floating
input and magnitude/precision-sensitive cases retain the original computation.
Unaligned or non-native-endian input retains its preparation fallback.

## GPU results

Measured on the same date with CuPy 13.6, CUDA 12.9 and a Tesla V100-SXM2-16GB.
The regular GPU path now uses 256-channel groups, performs the anti-aliased
frequency resize on GPU, and downloads the smaller FT image. The resize uses
`cupyx.scipy.ndimage` with skimage's Gaussian sigma, grid coordinates, constant
boundary and clipping rules. Streaming retains its CPU resize. Memory estimates
include the extra image arrays and Gaussian coefficients.

| Pulse width | Unchanged GPU processing | Optimized GPU processing | Speedup |
| --- | ---: | ---: | ---: |
| 4 samples | 80.8 ms | 30.9 ms | 2.6x |
| 8 samples | 81.7 ms | 31.4 ms | 2.6x |
| 32 samples | 104.2 ms | 57.1 ms | 1.8x |
| 128 samples | 178.1 ms | 129.0 ms | 1.4x |

These are warmed processing calls under the same scope as the CPU table. The
resize-and-download stage alone fell from about 38.0 ms to 2.27 ms.

A separate full-CLI test used 120 uint8 candidates on a synthetic 4096-channel
filterbank with the same band/sampling/DM. The width distribution was 40 each
at widths 4 and 8, and 20 each at widths 32 and 128. Both versions used the same
five CPU cores, one GPU and default two candidates in flight. Runs alternated
baseline/current order; each completed all 120 HDF5 files with default FP16
products and gzip level 9.

| Full CLI | Three wall samples | Median |
| --- | --- | ---: |
| Unchanged | 20.124, 20.559, 20.318 s | 20.318 s |
| Optimized | 15.421, 15.676, 15.882 s | 15.676 s |

That is 23% less wall time, or about 30% more candidates per second, including
startup, input reading/upload and compressed output. All DMT datasets matched
exactly. The largest FT difference was 0.001953125 in normalized FP16 output
from different floating-point resize rounding. This is not the original pulsar
observation or the report's RTX A4000.

In one optimized run the writer performed about 5.36 seconds of HDF5 work;
GPU dedispersion stage times totaled about 4.33 seconds. These stages overlap,
so their sums do not explain wall time by simple addition. Wide-pulse native
sample processing and output writing remain useful targets.

Validation: 432 Python tests passed, including GPU and streaming cases; Rust
tests, Clippy and formatting checks passed. Final wheel/source distributions
built successfully, installed native checks passed, and both CUDA extras
resolved with `uv` in a clean environment. CUDA 13 runtime was not tested.

## Follow-up: CPU optimizations from main

This second comparison uses a snapshot of the optimized branch immediately
before the CPU port, including the GPU improvements above. It does not reuse
the original `4db0d60` baseline. The snapshot has its own compiled extension at
`/tmp/cupy-maincpu-before`; results below use the same machine, Python environment,
synthetic cases and three warmed calls per case as above.

The port brings main's 128-channel bands, tiled byte transpose, exact integer
prefix sums, reduction of DM prefix boundaries before differencing, and native
anti-aliased resize. Eight-channel residue tables turn strided boundary reads
into contiguous runs when their construction is cheaper than the repeated
queries. These tables are bounded to 32 MiB per worker and 128 MiB per candidate, with
direct prefix reads if ineligible or allocation fails. Rust workers own channel
bands and reduce partial DM planes after joining. The GIL is released during
native processing.

Main's wrapping source geometry was not ported. Every source remains
`output_start + native_time + delay[dm, channel]`; the shared planner still reads
the maximum required right-hand coverage for all trials. Fractional median
padding at observation boundaries retains its float64 preparation/fallback.
Native CPU resize matches skimage bit for bit for finite FP32 integer shrinks;
other inputs retain skimage. CuPy kernels and regular GPU resizing are unchanged.

| Width | Before, 1 core | After, 1 core | Speedup, 1 core | After, 4 threads | Speedup vs before |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 489.4 ms | 151.8 ms | 3.22x | 104.0 ms | 4.71x |
| 8 | 498.6 ms | 155.6 ms | 3.21x | 106.5 ms | 4.68x |
| 32 | 655.6 ms | 203.1 ms | 3.23x | 137.9 ms | 4.75x |
| 128 | 1471.2 ms | 355.7 ms | 4.14x | 191.8 ms | 7.67x |

The single-thread runs use CPU 0; four-thread runs use CPUs 0–3. This separates
algorithmic gain from additional parallelism. CPU resize alone fell from about
33–37 ms to 3.4 ms. In the same serial benchmark sequence, GPU before/after
medians were 33.4/33.9, 29.5/31.1, 60.0/54.8 and 130.6/129.7 ms for the four
widths: no consistent GPU speed change. All saved FT and DMT FP32 arrays from
both CPU thread counts and GPU were bitwise identical to their respective
pre-port outputs.

The CLI budgets Rust threads from CPU affinity divided by `-n` candidate
processes, with at least one thread per process. Direct Python calls default to
one thread; use `Candidate.make_products(cpu_threads=N)` or
`process_cpu(..., threads=N)` to select more. Prepared float64 and unsupported
integer-magnitude cases retain the single-threaded fallback.

The follow-up full-CLI benchmark reused the 120-candidate filterbank workload
above. Both checkouts used CPUs 0–5 and `-n 2`. Before the port there were two
single-threaded CPU candidate workers; afterwards each received three Rust
threads within the same six-core budget. GPU retained its existing scheduler
and one V100. Each version produced all 120 FP16/gzip-9 files in each run;
baseline/current order alternated across three repeats.

| Backend | Before wall samples | After wall samples | Before median | After median |
| --- | --- | --- | ---: | ---: |
| CPU | 49.904, 48.702, 52.824 s | 13.781, 13.825, 14.270 s | 49.904 s | 13.825 s |
| GPU | 15.878, 15.359, 14.861 s | 15.690, 15.755, 15.550 s | 15.359 s | 15.690 s |

CPU wall time improved 3.61x (72% less time). GPU was approximately unchanged:
its median was 2.2% slower, within the observed before/after run ranges.
These timings include startup, reading, upload where applicable, finishing and
compressed output. They are synthetic measurements, not results on the original
observation. GPU arithmetic remains faster than CPU processing in the warmed
per-candidate table, while full CLI throughput also depends on scheduling and
CPU output work.
All 120 FT and DMT HDF5 datasets were bitwise identical before/after for both
backends in every repeat.

Validation: 468 Python tests and 8 Rust tests passed, including real right-hand
pulses in both frequency orders, multi-thread agreement, masked channel tails,
fractional EOF padding, and skimage resize parity. Strict Clippy, Rust/Python
formatting, scoped Ruff lint and independent read-only review passed. Final
wheel and source packages built; installed native checks, missing-Rust checks,
new native product/resize smoke checks and source-module inclusion passed.

An audit follow-up removed duplicate CPU product copies and added an asynchronous
GPU resize guard for floating-input planes containing nonzero values below
`1e-20`. CUDA's FP32 weighted operations can flush these values to zero, including
some normal FP32 inputs. Such planes now use the same CPU resize as the regular
CPU path. Bounded integer/float16 input skips the guard. The guard's reduction
scratch is included in workspace estimates, and its scalar download uses the
existing synchronization. Installed-package checks now exercise the new native
candidate and resize functions directly.

The same processing benchmark was rerun after these fixes. CPU one-core medians
were 144.4, 154.0, 198.3 and 331.5 ms; four-thread medians were 98.5, 96.8, 118.8
and 169.4 ms. GPU medians were 29.5, 28.8, 57.8 and 129.1 ms. All saved arrays
matched their pre-audit outputs bit for bit. The full suite passed 471 tests,
including tiny normal/subnormal and mixed-input GPU resize regressions. The
full-CLI table above is from before this follow-up; it was not rerun for the audit.

## End-to-end comparison against main

Measured on 2026-10-18 against main `1f9b0b2ffaa2d96bb8e1b9a23f495b8a7f01d831`,
verified as the remote main tip before testing. Main was extracted with
`git archive` and its own Rust extension built in release mode. The current
version was a separate snapshot of the `codex/cupy-speed` working tree,
including the uncommitted CPU port, CuPy optimizations and audit fixes.
Neither snapshot was edited during the runs.

Both used the same 120-candidate CSV and synthetic filterbank from the earlier
tests: 4096 uint8 channels, 1200–1500 MHz, 256 us sampling, DM 80, and widths
4/8/32/128 in a 40/40/20/20 split. All windows fit inside the observation.
The candidate identities were unique and matched against the saved HDF5
attributes after every run.

Hardware was the same V100-SXM2-16GB and six CPU cores, pinned to CPUs 0–5.
Both versions used the same Python 3.12.3 environment with NumPy 2.3.5,
SciPy 1.18.1 and h5py 3.16.0. Main used Numba 0.68.0 and numba-cuda 0.30.4;
the branch used CuPy 13.6.0. CUDA toolkit was 12.9, driver 580.178.04.
BLAS threads were limited to one before either CLI started.

Both CLIs used `-n 2`, `--time_size 256` and `--frequency_size 256`.
For CPU this gives two processes with three Rust threads each on both versions.
For GPU, main uses two processes sharing device 0; this branch uses its existing
single GPU process with two in-flight candidates. Main has 256 DM trials;
the branch was explicitly set to `--dmsteps 256 --gpu_inflight 2`.

Each sample times a fresh CLI subprocess from launch through exit, including
imports, worker/runtime setup, reads, transfers, computation, finishing and
gzip-9 HDF5 output. File/compiler disk caches were warm. There was no injected
profiling wrapper or implementation patch in main. Runs were serial, with
main/current order alternating across three repeats. Output validation happened
after the timer stopped.

The primary comparison sets `--output_dtype float32` on this branch to match
main's FP32 HDF5 products:

| Backend | Main wall samples | This branch wall samples | Main median | Branch median | Branch extra wall time |
| --- | --- | --- | ---: | ---: | ---: |
| CPU | 14.152, 14.212, 14.338 s | 14.750, 14.339, 14.413 s | 14.212 s | 14.413 s | +1.4% |
| GPU | 12.765, 12.929, 13.486 s | 15.039, 15.705, 15.402 s | 12.929 s | 15.402 s | +19.1% |

Throughput was 8.44/8.33 candidates per second for main/branch CPU and
9.28/7.79 for main/branch GPU. CPU is approximately tied. Main retains an
end-to-end GPU advantage on this workload.

Three additional runs per backend measured this branch's default FP16 output.
These were supplementary branch-only runs after the matched-FP32 comparison:

| Backend | Branch FP16 wall samples | Median | Main's default FP32 median |
| --- | --- | ---: | ---: |
| CPU | 13.948, 13.958, 14.250 s | 13.958 s | 14.212 s |
| GPU | 15.754, 15.297, 15.746 s | 15.746 s | 12.929 s |

Changing output precision did not close the GPU gap. The default branch GPU
run took 21.8% longer than main's default GPU run. On this fixture the branch's
CPU path also finished sooner than its GPU path, including all I/O and startup.

All 18 runs succeeded and each wrote 120 valid, finite products of the expected
shape/dtype: 2160 HDF5 files validated in total. Main logged **zero CPU fallbacks
across all 360 measured GPU candidates**, so its GPU result is genuine GPU work.
All versions used gzip level 9. Main/branch FP32 CPU files totaled 59.55/59.90 MB;
FP32 GPU files totaled 53.22/59.91 MB. Branch FP16 files totaled 31.22 MB.

This compares the complete implementations on identical inputs. Main retains
rotation-based dedispersion and its GPU frequency block averaging; the branch
keeps real right-hand samples, anti-aliased frequency resizing and FT detrending.
Output images are therefore not assumed identical. Code-level differences that
remain relevant to the GPU gap include main computing cropped time bins directly
and spreading host work across processes; the branch still builds native-time
GPU planes before binning. Their individual contributions were not isolated
by this wall-time benchmark.

Raw run commands, output counts, timings, fallback counts and byte totals are in
`/tmp/you-re-main-comparison/{cpu,gpu}-{float32,float16}/summary.json`.
`/tmp/you-re-main-comparison/results.json` records environment versions, dataset
hashes and source-manifest hashes; `run.py` is the direct-subprocess driver.
The original 380-candidate observation was not benchmarked.

## Single-GPU CLI scheduling follow-up

Profiling the 32 real Lovell candidates found 0.105 s of dedispersion inside a
9.50 s CLI run. The spawned GPU worker repeated about 1.4 s of scientific-stack
imports and exchanged file groups through process queues. Every singleton batch
also created and shut down a thread pool.

The single-GPU CLI now runs the existing worker in the CLI process. Singleton
batches call the existing GPU job runner directly. The library's default and
multi-GPU runs retain spawned workers; explicit in-process calls require one GPU
and no output callback. GPU kernels, real right-hand samples, resizing,
detrending and output precision defaults are unchanged.

Against main `1f9b0b2` and the original CuPy branch `6be5d48`, fresh measurements
on the same V100 and six CPU cores gave these end-to-end medians:

| Dataset | Main GPU FP32 | Original CuPy GPU FP32 | Fixed CuPy GPU FP32 | Fixed default FP16 |
| --- | ---: | ---: | ---: | ---: |
| 32 real candidates, 31 files | 7.188 s | 9.000 s | 5.880 s | 5.724 s |
| 120 synthetic candidates, one file | 13.212 s | 15.284 s | 13.474 s | 14.133 s |

Matched FP32 uses 34.7% less time than the original CuPy path on the real data
and 18.2% less than main. On the larger synthetic batch it uses 11.8% less time
than the original and 2.0% more than main. Default FP16 did not improve the
synthetic end-to-end time despite its smaller output files.

Each number is the median of three fresh CLI runs after an untimed full-data
warmup. Main/original/fixed order rotated across repeats, and supplementary FP16
runs followed. Settings remained `-n 2`, 256 by 256 output, 256 DM trials and two
in-flight CuPy candidates. Timers include launch through exit, imports, reads,
transfers, finishing and gzip-9 HDF5 output. Validation is outside the timer.
The real files have 800 or 672 uint8 channels and 256 us sampling; the synthetic
fixture is unchanged from the earlier comparison.

All 24 measured runs and six warmups passed: 1,824 measured and 2,280 total
outputs validated. Main logged no CPU fallbacks. All 456 original/fixed FP32
candidate pairs were bitwise identical across both datasets and three repeats.
The full Python suite passed 481 tests; scoped lint, formatting and independent
review passed. Physical multi-GPU execution was not benchmarked.

Raw commands, logs, products, fingerprints and timings are in
`/tmp/you-re-gpu-fixed-comparison`; diagnostics are in
`/tmp/you-re-gpu-investigation`. The real-data report and JSON are saved beside
the inputs as `/home/riftuser/data/R1_gpu_optimization.{md,json}`.

## Remaining opportunities

After arithmetic and launch improvements, measure CPU finishing and compressed
HDF5 output before adding scheduler processes. A writer process may overlap
compression better, but requires bounded queues, error handling and careful
ownership of candidate arrays. Multiple GPU-owning processes also duplicate
resident input and CUDA contexts. Neither change is justified just by counting
Python processes.

For wide pulses, a GPU prefix-sum path could reduce repeated native-time work,
but must account for its extra buffers in the memory budget and establish a
crossover including prefix construction. The common narrow-pulse workload
should not pay that setup cost without evidence.

## Reproducible processing benchmark

Build each checkout's Rust extension separately, then run the same harness and
Python environment against each checkout:

```bash
CUDA_PATH=/usr/local/cuda-12.9 python checks/benchmark_candidate_speed.py \
  --checkout /path/to/baseline --backend gpu --output /tmp/before-gpu.json
CUDA_PATH=/usr/local/cuda-12.9 python checks/benchmark_candidate_speed.py \
  --checkout /path/to/optimized --backend gpu --output /tmp/after-gpu.json
```

Use `--backend cpu` for CPU processing. Pin both CPU runs to the same core with
`taskset -c N` to distinguish algorithmic improvements from extra threads.
The default cases are deterministic uint8 input, 4096 channels, 1200–1500 MHz,
256 microsecond sampling, DM 80, widths 4/8/32/128 samples, and 256 by 256 FP32
products. There is no padding in these timing cases; regression tests cover
padding, masks and other input types.

The harness warms each case before three measured calls, records every wall
sample, and saves the final arrays alongside JSON. Its separate diagnostic
profile is not part of the wall samples. GPU input is resident before timing.
These numbers include product downloads and CPU finishing, but exclude file
reading, initial GPU upload, interpreter startup and HDF5 writes. They must not
be presented as full CLI throughput or as results on the original pulsar data.

For deployment decisions, also use a mixed-width candidate list with the full
CLI, alternate baseline/current order, and include startup, reads and writes.
Keep CUDA compilation warm for both checkouts and compare output files by
candidate metadata rather than directory order.
