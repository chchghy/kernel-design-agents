# Vector Add CUDA Kernel Optimization — Implementation Plan Draft

**Task:** High-performance element-wise vector addition CUDA kernel for NVIDIA H100 (SM90).
**Date:** 2026-06-10
**Status:** Draft

---

## 1. Baseline and Validation Path

### 1.1 Problem Definition

Element-wise vector addition: `C[i] = A[i] + B[i]` for all `i ∈ [0, N)`, where A, B, C are FP32 arrays of length N.

**Arithmetic intensity:** 1 FLOP / (3 × 4 bytes) = **0.083 FLOP/byte** — this kernel is purely **memory-bound** on H100.

The performance ceiling is dictated entirely by HBM3 bandwidth (~3.35 TB/s on H100 SXM5). The theoretical peak throughput for this operation (read A, read B, write C = 12 bytes per element) is:

```
Peak throughput = 3.35 TB/s / 12 bytes/elem ≈ 279 Gelem/s
Peak GB/s (effective) = 3.35 TB/s × (4 bytes × 3 / 12 bytes) ≈ 3.35 TB/s (total mem traffic)
```

### 1.2 H100 (SM90) Hardware Parameters

| Parameter | Value |
|-----------|-------|
| SMs (SXM5) | 132 |
| Max threads per SM | 2048 |
| Max threads per block | 1024 |
| Warp size | 32 |
| Max warps per SM | 64 |
| Max blocks per SM | 32 |
| Registers per SM | 65536 |
| L1 / Shared memory per SM | 256 KB (configurable) |
| L2 cache | 50 MB |
| HBM3 bandwidth (SXM5) | 3.35 TB/s |
| L1 cache line | 128 bytes |
| Memory transaction size | 32 / 64 / 128 bytes |

### 1.3 Baseline Implementation

A naive baseline kernel:
- Each thread processes **one element** (1 FP32 load from A, 1 from B, 1 store to C).
- Standard 1D grid with `(N + blockDim.x - 1) / blockDim.x` blocks, 256 threads/block.
- No vectorized access, no cache hints, no `__launch_bounds__`.
- Typical throughput: ~30–50% of peak HBM bandwidth due to uncoalesced 32-bit memory transactions and insufficient memory-level parallelism.

### 1.4 Validation Command

```bash
nvcc -O2 -std=c++17 -arch=sm_90 -lineinfo \
     vector_add.cu -o vector_add
./vector_add --validate
```

Validation checks:
1. `max|C_gpu[i] - C_cpu[i]| < 1e-3` for all `i`.
2. No NaN or inf values in GPU output.
3. All CUDA API calls return success.

### 1.5 Evaluation Command

```bash
# Warmup + benchmark (10 warmup iterations, 100 timed iterations)
./vector_add --benchmark --warmup 10 --iterations 100 --size $((1024*1024*256))  # 256M elements = 1 GiB per array
```

Metrics reported:
- Average kernel latency (μs)
- Effective throughput in GB/s (total bytes read+written / latency)
- Percentage of theoretical HBM bandwidth

---

## 2. Main Risks and Unknowns

| Risk | Severity | Mitigation |
|------|----------|------------|
| H100 not physically available in CI | High | The validation runtime environment is unknown. Kernel compiles for sm_90; if no H100 GPU is present, evaluation relies on theoretical analysis and Nsight Compute estimates. |
| Vectorized loads exceed alignment requirements | Medium | Use `__align__(16)` for float4; validate alignment at runtime. Misaligned pointers cause multiple narrower transactions. |
| Register spilling from aggressive unrolling | Medium | Profile with `-Xptxas -v` to check register usage; use `__launch_bounds__` and `-maxrregcount` as needed. |
| Tail elements (N not multiple of vector width) | Low | Handle with a clean-up loop for remaining elements after the vectorized main loop. |
| Implicit L2 caching hides L1 bypass benefits | Low | On streaming access patterns, L1::no_allocate reduces L1 pollution and frees cache for other warps' in-flight loads. The benefit is measurable on H100's 256 KB L1/SMEM. |
| Float4 store might be slower than float4 load | Low | Profile both. On H100, store throughput matches load throughput for aligned vectorized accesses. |

---

## 3. Candidate Implementation Directions

Ranked by expected value × confidence / risk:

### Candidate 1: Vectorized float4 Load/Store (High value, Low risk)

**Technique:** Replace per-element 32-bit loads with 128-bit float4 loads and stores. Each thread processes 4 elements per iteration instead of 1.

- `float4` loads: 16 bytes per instruction (4 float elements)
- Reduces instruction count by ~4×, saturates memory bus more efficiently
- Grid: `(N/4 + blockDim.x - 1) / blockDim.x` blocks

**Expected improvement:** 1.5–3× over baseline.
**Risk:** Minimal — alignment requirement is 16 bytes, easily satisfied by `cudaMalloc`.

### Candidate 2: Cache Policy Hints (Medium value, Low risk)

**Technique:** Apply `L1::no_allocate` streaming hint to all three arrays via PTX inline assembly. All three arrays are streaming (no data reuse), so L1 caching is wasteful.

```cuda
// PTX inline for vec4 load with L1 bypass
asm volatile(
    "ld.global.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w)
    : "l"(ptr_a)
);
```

**Expected improvement:** 5–15% over vectorized-only baseline, per KernelWiki data on memory-bound kernels.
**Risk:** Low. PTX inline assembly is well-documented for H100.

### Candidate 3: 256-bit (uint4/float4×2) Loads + ILP (Medium-High value, Medium risk)

**Technique:** Each thread processes 8 elements per iteration using two back-to-back float4 loads (instruction-level parallelism). This doubles the in-flight memory transactions per thread, hiding memory latency better.

```cuda
float4 a0, a1, b0, b1;
// ILP: interleave loads from A and B to maximize memory-level parallelism
a0 = *((float4*)a_ptr + idx);
b0 = *((float4*)b_ptr + idx);
a1 = *((float4*)a_ptr + idx + 1);
b1 = *((float4*)b_ptr + idx + 1);
```

**Expected improvement:** 10–30% over Candidate 1 on large vectors.
**Risk:** Increases register pressure (needs ~16 registers for 8 float values). Profile for spills.

### Candidate 4: Occupancy Tuning with __launch_bounds__ and Block Size (Medium value, Low risk)

**Technique:** Use `__launch_bounds__` to inform the compiler of expected thread count. Tune block size (128/256/512/1024) to maximize occupancy. On H100 with 2048 max threads/SM, 256-thread blocks allow 8 blocks/SM; 128-thread blocks allow 16 blocks/SM.

**Expected improvement:** 0–20% depending on baseline occupancy.
**Risk:** Low — `__launch_bounds__` is a compiler hint; worst case is no improvement.

### Candidate 5: Multi-Dimensional Grid for Very Large Vectors (Low value, Low risk)

**Technique:** Use 2D/3D grid to avoid integer overflow in 1D grid for N > 2^31. Not strictly an optimization but ensures correctness for large vectors.

**Expected improvement:** Correctness fix only.
**Risk:** Minimal.

### Candidate 6: Cooperative Groups / Async Copy (Exploratory, Low-Medium value, Medium risk)

**Technique:** Use `cuda::memcpy_async` or cooperative groups for potential overlap of computation and memory access. For a pure element-wise kernel with zero compute, this is unlikely to help.

**Expected improvement:** Minimal for pure memory-bound element-wise ops.
**Risk:** Added complexity; async copy pipeline requires shared memory staging, which reduces occupancy.

---

## 4. Recommended Implementation Sequence

1. **Create the full harness** (`vector_add.cu`) with CPU reference, validation, and benchmark infrastructure.
2. **Implement naive baseline** — 32-bit per-element kernel, 256 threads/block.
3. **Implement Candidate 1** — float4 vectorization (128-bit loads/stores).
4. **Implement Candidate 2** — Add L1 cache policy hints on top of Candidate 1.
5. **Implement Candidate 3** — 2× float4 ILP on top of Candidate 2.
6. **Implement Candidate 4** — Occupancy tuning (test block sizes 128/256/512).

Each candidate is validated for correctness before benchmarking.

---

## 5. First Concrete Implementation Steps

### Step 1: Write the harness and naive baseline

File: `vector_add.cu`

Structure:
```
vector_add.cu
├── #include <cuda_runtime.h>, <stdio.h>, <stdlib.h>, <math.h>, <chrono>
├── CUDA_CHECK macro
├── CPU reference: vector_add_cpu(float* C, const float* A, const float* B, size_t N)
├── Naive GPU kernel: vector_add_naive<<<...>>>
├── Validation: validate(C_gpu, C_cpu, N) → max absolute error
├── Benchmark: benchmark(kernel, args, warmup_iters, bench_iters) → avg_time_us, gb_s
├── main(): --validate | --benchmark | --size N
└── Makefile or compile notes
```

### Step 2: Compile and validate

```bash
nvcc -O2 -std=c++17 -arch=sm_90 -lineinfo vector_add.cu -o vector_add
./vector_add --validate
```

### Step 3: Benchmark baseline

```bash
./vector_add --benchmark --warmup 10 --iterations 100 --size 268435456
```

### Step 4: Iterate through candidates 1→4

After each candidate, re-run validation and benchmark. Record results.

---

## 6. Validation and Evaluation Commands

### Validation (correctness only)

```bash
nvcc -O2 -std=c++17 -arch=sm_90 -lineinfo vector_add.cu -o vector_add
./vector_add --validate --size 16777216    # 64 MiB per array
./vector_add --validate --size 268435456   # 1 GiB per array
./vector_add --validate --size 1048577     # Non-multiple of 4 (tests tail handling)
```

### Evaluation (performance)

```bash
# Large vector benchmark
./vector_add --benchmark --warmup 10 --iterations 100 --size 268435456

# Medium vector benchmark
./vector_add --benchmark --warmup 10 --iterations 100 --size 16777216

# Optional: Nsight Compute profiling (if H100 is available)
ncu --set full --kernel-name regex:vector_add ./vector_add --benchmark --size 16777216
```

---

## 7. Promotion / Revision / Rejection Criteria

### Promote (accept candidate)
- All validation checks pass (max error < 1e-3, no NaN/inf).
- Kernel latency is measurably lower than naive baseline (≥1.2× speedup).
- Effective bandwidth ≥ 60% of theoretical HBM peak.
- Candidate is stable across ≥3 sizes (medium: 16M, large: 256M, odd: 16M+1).

### Revise (iterate on candidate)
- Validation passes but performance is within 10% of baseline.
- Performance improves but correctness fails — debug the vectorization or tail logic.
- Effective bandwidth is below 40% of peak — check occupancy, alignment, or cache policy.

### Reject (discard direction)
- Candidate produces incorrect results that cannot be fixed without abandoning the technique.
- Performance is worse than baseline (e.g., due to register spilling or cache thrashing).
- Complexity is too high for marginal gain (e.g., < 5% improvement with significantly more code).

---

## 8. Expected Performance Progression

| Stage | Expected Bandwidth* | Expected Speedup |
|-------|-------------------|-----------------|
| Naive baseline | ~600–900 GB/s | 1.0× |
| + float4 vectorization | ~1.5–2.0 TB/s | 1.5–2.5× |
| + L1 cache policy | ~1.8–2.5 TB/s | 1.1–1.2× over prev |
| + ILP (2× float4) | ~2.2–2.8 TB/s | 1.1–1.2× over prev |
| + Occupancy tuning | ~2.3–3.0 TB/s | 1.0–1.1× over prev |

*Effective bandwidth of total memory traffic (read A + read B + write C). Theoretical peak for H100 SXM5 is 3.35 TB/s.

---

## References

- KernelWiki: `technique-vectorized-loads` — Wide Vectorized Loads and Cache Policies (NVFP4 GEMV optimization progression: 2000μs → 22.4μs, 89× improvement through vectorization + cache policy + register tuning)
- KernelWiki: `technique-cache-policy` — PTX Cache Policy Differentiation (L1::no_allocate, L1::evict_last) for streaming vs reused data
- NVIDIA H100 white paper / CUDA Programming Guide SM90
- CUDA C++ Best Practices Guide — Memory coalescing and vectorized access patterns
