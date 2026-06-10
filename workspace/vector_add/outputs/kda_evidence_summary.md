# KDA Evidence Summary — vector_add

## Task Contract Fulfillment

| Requirement | Status | Evidence |
|---|---|---|
| Correctness (1e-3 tolerance) | ✅ | All 9 kernels pass all N sizes (0-2^28), all edge patterns, 5-run stability PASS |
| H100 SM90 target | ✅ | NVIDIA H100 80GB HBM3, compute 9.0, 132 SMs |
| Pure CUDA C++ (no 3rd-party) | ✅ | Single .cu file, only stdlib + CUDA runtime |
| Improve over naive baseline | ✅ | +9.9% bandwidth improvement |

## Final Result

**Promoted Candidate:** `float4_kernel` — 128-bit vectorized loads/stores, alignment-aware dispatch, scalar tail

| Metric | Value |
|---|---|
| Bandwidth @ N=2^28 | **3,086 GB/s** |
| Baseline @ N=2^28 | 2,808 GB/s (grid-stride + __restrict__) |
| Improvement | **+9.9%** |
| % Peak HBM3 bandwidth | **92.1%** |
| Block size | 256 threads (1024 marginally better at +0.3%, within noise) |
| SASS verified | LDG.E.128 + STG.E.128, zero register spills |

## Candidate Ranking (N=2^26)

| Rank | Candidate | BW (GB/s) | vs Baseline |
|------|-----------|-----------|-------------|
| 1 | float4_b1024 | 3051 | +10.1% |
| 2 | float4_b512 | 3047 | +10.0% |
| 3 | float4_b256 | 3043 | +9.8% |
| 4 | cache_hint (CUDA fallback) | 3043 | +9.8% |
| 5 | float4_b128 | 3042 | +9.8% |
| 6 | float4 | 3042 | +9.8% |
| 7 | ilp2x | 3028 | +9.3% |
| 8 | float2 | 2800 | +1.1% |
| 9 | baseline | 2771 | — |

## Blocked Items

| Item | Reason |
|------|--------|
| Nsight Compute profiling | ERR_NVGPCTRPERM — GPU perf counters restricted to root |
| -maxrregcount sweep | 92.1% peak BW already achieved; memory-bound kernel not register-limited |
| cache_hint PTX path | Streaming kernel unlikely to benefit; CUDA fallback verified identical perf |

## Artifacts

| File | Content |
|------|---------|
| vector_add.cu | ~1050 lines, 9 kernels, full harness |
| benchmark.csv | Multi-N sweep results |
| candidates.jsonl | 63 evaluated candidates with promotion decisions |
| profile/float4_v1_promoted/ | ncu-report-skill structured run directory |
| runs/ | SASS disassembly + PTX |
| outputs/ | Summary reports + binary |
