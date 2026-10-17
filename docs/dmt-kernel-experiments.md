# DMT kernel experiments on V100

Measured on 2026-10-17 with a Tesla V100-SXM2-16GB, CuPy 13.6 and CUDA 12.9.
Production source SHA-256: `ef0c47434f622a613e7c281483d7ff65e248dddfa031558e8ffb31e6eb03cd49`.
These are benchmark-only experiments; production dispatch and kernels were unchanged.

## Measurement scope

Synthetic, resident input; 4096 channels, 400–800 MHz, 100 us sampling,
256 DM trials from zero through twice the candidate DM, 256 output time bins.
Candidate widths 2 and 256 produce time-bin factors 1 and 128. All timing
workloads have enough real samples and no edge padding. Smaller correctness
cases exercise padding, nonfinite floating input, partial DM/time/channel tiles,
and all seven supported source dtypes.

Wall times include `process_gpu`, product download and CPU product finishing.
They exclude file reading and the initial input upload. Final geometry timings
have no additional profiling wrappers; production event recording remains.
Standalone geometry finalists use five randomized paired rounds. Combined
loading/geometry comparisons use three randomized paired rounds. Tables show
medians, not isolated DMT kernel times. The runs use one GPU exclusively.

## Confirmed complete-processing times

| Input and workload | Baseline | Selected experiment | Time | Reduction |
| --- | ---: | --- | ---: | ---: |
| uint8, DM80, bin1 | 127.92 ms | T512 / DM8 / C256 | 72.62 ms | 43% |
| uint8, DM700, bin1 | 124.57 ms | T512 / DM4 / C256 | 74.14 ms | 40% |
| uint8, DM80, bin128 | 369.26 ms | Guarded endpoints + T512 / DM4 / EPT2 / C32 | 229.04 ms | 38% |
| uint8, DM700, bin128 | 1028.43 ms | Direct reads, production geometry | 345.59 ms | 66% |
| float16, DM80, bin128 | 727.92 ms | Production time/DM tile, C256 | 614.24 ms | 16% |
| float32, DM80, bin128 | 801.08 ms | Production time/DM tile, C256 | 642.70 ms | 20% |

T is the logical time tile, DM the trial tile, C the resident channel group,
and EPT the samples computed by each thread. EPT2 uses 256 threads for T512.
Each baseline above comes from the same paired experiment as its selected result.

## What helped and what did not

- Direct reads avoid shared staging over a wide sparse delay span. At DM700/bin128,
  adding the geometry winner to direct reads produced 344.51 ms versus 345.59 ms
  for direct reads alone. That roughly 1 ms difference does not justify selecting
  the more complex combination from this evidence.
- At DM80/bin128, endpoints alone took 270.18 ms and geometry alone 301.60 ms;
  their combination took 229.04 ms. This confirmation includes the host-side
  monotonicity guard. Unordered delay rows must retain the general bounds scan.
- Larger channel groups helped unbinned integer and wide floating workloads.
  This result applies to floating input too; it does not depend on uint8 packing.
- Two samples per thread helped wide-bin DMT, but was not a universal winner.
- Shared-loading caps of 1024/4096 floats improved the slow DM700 baseline,
  but lost to direct reads in those screens.
- T1024/DM4 and T1024/DM8 compiled but could not launch on this V100 because
  of resource limits, across all seven tested dtypes. They are excluded from rankings.

Floating-input median calculation took hundreds of milliseconds in diagnostic
passes, and FP32 scaling also had a measurable cost. These remain separate
optimization targets; improving DMT alone cannot remove them. CPU finishing
was about 50 ms in the staging profiles, including roughly 38 ms of resizing.
Profiling passes are separate observations and must not be summed into wall medians.

## Correctness and interpretation

The staging sweep recorded 238 zero-error comparisons. Final integer geometry
and combined products also matched the baseline exactly. Regrouping floating
reductions changed DMT by at most about 0.00065 absolute in the measured cases,
within the accepted `rtol=atol=0.003`; it does not satisfy the stricter 0.00002
check everywhere. FT products were unchanged. Relative error near zero is not
a useful standalone interpretation of these normalized products.

The initial staging screens/finals enabled built-in stage profiling. Their
historical endpoint timings excluded monotonicity-guard overhead. Those limits
are recorded in the staging JSON; the later combined comparison includes the
guard and uses unprofiled wall measurements. The reusable staging harness now
includes guard time and disables extra profiling for final wall rounds.

These results support a small workload-dependent configuration table, rather
than choosing one global kernel configuration. No runtime tuner, production
routing change, pre-decimation, or multi-GPU benchmark was introduced here.
Baseline and selected experimental sources compiled to PTX for compute_61 and
compute_120 with CUDA 12.9. This checks source compatibility only.
Synthetic V100 timings do not establish performance on Pascal or newer GPUs,
or on real candidate files.

## Artifacts

- `checks/benchmark_dmt_staging.py`: direct/shared loading and guarded bounds.
- `checks/benchmark_dmt_geometry.py`: bounded tile/EPT/channel-group sweep.
- `checks/benchmark_dmt_combined.py`: repeat the two wide-bin combination tests.
- `/tmp/benchmark_dmt_staging_results.json`: staging samples and correctness.
- `/tmp/benchmark_dmt_geometry_results.json`: geometry samples and correctness.
- `/tmp/benchmark_dmt_geometry_combined.json`: paired combined comparisons.

Run benchmarks with `CUDA_PATH=/usr/local/cuda-12.9` and the CuPy-enabled Python.
Use each script's `--help` for output paths and repeat counts. Reserve the GPU;
concurrent benchmarks invalidate the comparisons.
