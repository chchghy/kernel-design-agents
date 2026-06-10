# Vector Addition Benchmark Summary — H100 SXM5

**Date:** 2026-06-10
**GPU:** NVIDIA H100 80GB HBM3 (SM 9.0, 132 SMs)
**Theoretical Peak Bandwidth:** 3,352 GB/s (HBM3, 5120-bit @ 2.619 GHz DDR)

## Key Results at N = 2^28 (268M elements, 3.22 GB I/O)

| Kernel | Latency (us) | Bandwidth (GB/s) | % Peak | vs Baseline |
|--------|-------------|-------------------|--------|-------------|
| baseline | 1147.1 | 2808.3 | 83.8% | — |
| **float4** | **1040.6** | **3095.5** | **92.3%** | **+10.2%** |

## Multi-N Sweep (N=2^26, 64M elements)

| Kernel | Bandwidth (GB/s) | % Peak |
|--------|-------------------|--------|
| baseline (scalar, grid-stride) | 2771.0 | 82.7% |
| float2 (64-bit loads) | 2800.4 | 83.5% |
| **float4 (128-bit loads)** | **3042.1** | **90.7%** |
| ilp2x (2x float4, 256-bit/iter) | 3027.8 | 90.3% |
| cache_hint (L1::no_allocate PTX) | 3043.0 | 90.8% |
| float4_b128 | 3041.6 | 90.7% |
| float4_b256 | 3043.4 | 90.8% |
| float4_b512 | 3046.9 | 90.9% |
| float4_b1024 | 3051.1 | 91.0% |

## Architecture Evidence

- **SASS:** LDG.E.128.CONSTANT (128-bit vectorized loads), STG.E.128 (128-bit stores)
- **No register spills:** Zero STL/LDL instructions in SASS
- **PTX:** Available in profile/vector_add.ptx
- **Nsight Compute:** Blocked by GPU perf counter permissions (ERR_NVGPCTRPERM)

## Promoted Candidate

**float4_kernel** — 128-bit vectorized loads/stores, alignment-aware dispatch, scalar tail for N%4≠0.

- 3095.5 GB/s at N=2^28 (92.3% peak, exceeds 2500-2900 GB/s hard target)
- +10.2% over baseline (grid-stride + __restrict__ + size_t)
- Block size 256 threads (sweet spot; block=1024 is marginally better at +0.3% within noise)

## Correctness

All 9 kernel variants pass validation at all N sizes (0, 1, 127, 1023, 2^20-2^28).
Edge cases: all zeros, all ones, powers of two, near FLT_MAX/2, alternating signs — all PASS with max error = 0.
No NaN or inf values detected.
