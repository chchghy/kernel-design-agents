# Vector Addition CUDA Kernel — Implementation Plan Draft

## 1. Baseline

### 1.1 Current State

No existing CUDA source code in the workspace. We need to create:
- A naive baseline kernel (`vector_add_naive.cu`)
- A CPU reference implementation for correctness validation
- A benchmarking harness

### 1.2 Naive Baseline Design

```cuda
__global__ void vector_add_naive(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

Configured with 256 threads/block, grid sized to cover N elements. This represents the standard naive approach — one element per thread, standard 32-bit loads/stores.

### 1.3 Validation and Evaluation

**Validation command:**
```bash
nvcc -arch=sm_90 -O3 vector_add.cu -o vector_add && ./vector_add
```
The program outputs "PASS" or "FAIL" with max absolute error. All elements must match CPU reference within 1e-3 absolute tolerance.

**Evaluation command:**
```bash
nvcc -arch=sm_90 -O3 vector_add.cu -o vector_add && ./vector_add --benchmark
```
Runs 5 warm-up iterations + 20 timed iterations, reports average kernel latency (ms) and effective throughput (GB/s).

### 1.4 Expected Baseline Performance

For FP32 element-wise addition (N = 2^28 ≈ 268M elements, ~3.2 GB total), the theoretical lower bound is:
- Data transferred: 2 reads (A, B) + 1 write (C) = 12 bytes per element
- For N=268M elements: 12 × 268M = 3.22 GB
- At H100 peak bandwidth (3.35 TB/s): ~0.96 ms minimum
- Naive kernel with 32-bit loads: expect 60-70% of peak → ~1.4-1.6 ms → ~2000-2300 GB/s

## 2. Key Characteristics

### 2.1 Arithmetic Intensity

Vector addition is **deeply memory-bound**:
- Reads: 8 bytes/element (A[i] + B[i] in FP32)
- Writes: 4 bytes/element (C[i] in FP32)
- Compute: 1 FLOP/element (single addition)
- Arithmetic intensity: 1 FLOP / 12 bytes ≈ **0.083 FLOPs/byte**

The H100 roofline knee is at ~20 FLOPs/byte (67 TFLOPS / 3.35 TB/s). At 0.083 FLOPs/byte, the kernel operates deep in the memory-bound regime. **Compute optimizations will have zero benefit** — the entire optimization strategy must focus on memory bandwidth utilization.

### 2.2 H100 Hardware Reference

| Parameter | Value |
|---|---|
| Peak HBM3 bandwidth | 3.35 TB/s (SXM) |
| L2 cache | 50 MB |
| SMs | 132 |
| Max threads/SM | 2048 |
| Max blocks/SM | 32 |
| Max registers/SM | 65536 |
| Max shared memory/SM | 228 KB |
| Max threads per block | 1024 |
| Warp size | 32 |

### 2.3 Speed-of-Light Calculation

For N = 2^28 elements (1 GB per vector, 3 GB total I/O):
- **Theoretical minimum**: 3.0 GB / 3.35 TB/s ≈ **0.90 ms**
- Corresponding throughput: **3,352 GB/s**

No kernel can exceed this. A well-optimized kernel should achieve 80-90% of peak (2,680-3,017 GB/s).

## 3. Risks and Unknowns

| Risk | Severity | Mitigation |
|---|---|---|
| H100 memory controller saturation — 128-bit loads may not fully saturate due to L2 bandwidth limits | Medium | Try 128-bit and compare with 64-bit; measure with Nsight Compute |
| L1 cache line eviction — streaming access pattern may not benefit from L1 caching | Low | Use `L1::no_allocate` cache hint for all streams |
| Register spill from vectorized loads (float4) consumes more registers | Low | Use `-maxrregcount` to balance; 32-40 regs/thread is sufficient |
| Block size too small → insufficient occupancy to hide latency | Medium | Grid stride loop (multi-element per thread) to keep occupancy high with fewer blocks |
| Launch overhead for small N | Low | Not in scope; benchmark at large N (2^28) |
| Misaligned addresses cause split transactions | Low | Use `cudaMalloc` which returns 256-byte aligned memory |

## 4. Candidate Implementation Directions

Ranked by expected value (throughput improvement over naive) and risk:

### Candidate 1: Vectorized Memory Access (float4)

**Expected improvement:** High (~1.5-2x bandwidth utilization gain)
**Risk:** Low
**Description:** Replace scalar float loads/stores with `float4` (128-bit) vectorized loads/stores. Each thread processes 4 elements instead of 1, issuing 4x fewer memory transactions. This is the single most impactful change for a memory-bound kernel.

```cuda
// Key optimization: float4 loads/stores
float4 a_vec = reinterpret_cast<const float4*>(A)[idx];
float4 b_vec = reinterpret_cast<const float4*>(B)[idx];
float4 c_vec;
c_vec.x = a_vec.x + b_vec.x;
c_vec.y = a_vec.y + b_vec.y;
c_vec.z = a_vec.z + b_vec.z;
c_vec.w = a_vec.w + b_vec.w;
reinterpret_cast<float4*>(C)[idx] = c_vec;
```

**Validation:** Compare output with CPU reference. Vectorized kernel should produce identical results to naive (same FP32 operations, just batched).

### Candidate 2: Cache Policy + Multi-element Per Thread

**Expected improvement:** Medium (~1.1-1.3x over Candidate 1)
**Risk:** Low  
**Description:** Use `L1::no_allocate` cache hint via PTX inline assembly for all global memory loads. The vector addition kernel has no data reuse (each element read once), so L1 caching is counterproductive — it evicts potentially useful data and adds latency. Combine with processing multiple float4 chunks per thread (grid-stride loop) to amortize launch overhead and improve occupancy.

```cuda
// PTX with L1::no_allocate for streaming access
asm volatile(
    "ld.global.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
    : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w)
    : "l"(ptr_a));
```

### Candidate 3: Tuned Block Size + Register Budgeting

**Expected improvement:** Small-Medium (~1.05-1.15x over Candidate 2)
**Risk:** Low
**Description:** Sweep block sizes (128, 256, 512, 1024) and register limits (32, 40, 48, 56, 64) to find optimal occupancy configuration. For memory-bound kernels, higher occupancy helps hide memory latency. Use `__launch_bounds__` and `-maxrregcount`.

### Candidate 4: 256-bit Loads (v4.u64)

**Expected improvement:** Small (~1.0-1.05x over Candidate 2)
**Risk:** Medium — requires 32-byte alignment, may not improve over 128-bit if already saturating memory bus
**Description:** Use 256-bit loads (`ld.global.v4.u64`) to issue even wider memory transactions. This may help on H100's memory subsystem but benefits diminish beyond 128-bit for simple kernels.

## 5. First Implementation Steps

1. **Create the unified CUDA source file** (`vector_add.cu`) containing:
   - CPU reference function
   - Validation harness (check results, report max error)
   - Benchmark harness (warm-up iterations, timed iterations, throughput calculation)
   - Naive baseline kernel
   
2. **Compile and run baseline** to establish reference performance numbers.

3. **Implement Candidate 1 (float4 vectorization)** and validate correctness.

4. **Implement Candidate 2 (cache policy + grid-stride loop)** and benchmark.

5. **Implement Candidate 3 (tune block size + register count)** and benchmark.

6. **Optionally implement Candidate 4 (256-bit loads)** if bandwidth saturation is not achieved.

## 6. Exact Commands

### Validation
```bash
nvcc -arch=sm_90 -O3 -lineinfo vector_add.cu -o vector_add && ./vector_add
```

### Evaluation (Benchmark)
```bash
nvcc -arch=sm_90 -O3 -lineinfo vector_add.cu -o vector_add && ./vector_add --benchmark
```

### Performance Counters (Nsight Compute)
```bash
ncu --metrics gpu__time_duration.sum,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed ./vector_add
```

## 7. Promotion Criteria

### Evidence Required to Promote a Candidate

| Criterion | Threshold |
|---|---|
| Correctness | All elements match CPU reference within 1e-3 absolute tolerance. No NaN, no inf. |
| Stability | Passes validation 100% of the time across 5 repeated runs |
| Performance | Clear improvement in effective throughput (GB/s) vs. naive baseline |
| Resource utilization | Achieved occupancy ≥ 50% on H100; memory throughput ≥ 70% of peak |

### Evidence Required to Revise

- Performance regression or no improvement vs. best prior candidate
- Correctness failure
- Degraded occupancy without compensating throughput gain

### Evidence Required to Reject

- Candidate is fundamentally broken (e.g., uses shared memory for streaming data, tries tensor cores for element-wise ops)
- Candidate produces worse results than all prior candidates with no path to improvement
- Candidate relies on unavailable hardware features or breaks the task contract

## 8. Expected Final Performance

A fully optimized FP32 vector addition kernel on H100 should achieve:
- **Throughput:** 2,500-2,900 GB/s (75-87% of peak 3,352 GB/s)
- **Latency (N=2^28):** ~1.05-1.20 ms
- **Key optimization levers:** float4 vectorization, L1 cache bypass, multi-element per thread, optimal block size + register count

The gap to 100% of peak bandwidth (~0.90 ms) comes from: instruction dispatch overhead, L2 cache tag lookups, DRAM page activation latency, and the fact that vector addition has 3 separate memory streams (A read, B read, C write) which limits the achievable fraction of peak DRAM utilization.
