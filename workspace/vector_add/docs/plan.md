# Vector Add CUDA Kernel Optimization — Implementation Plan

## Goal Description

Implement a high-performance element-wise vector addition CUDA kernel (`C[i] = A[i] + B[i]` for FP32 arrays of length N) targeting NVIDIA H100 (SM90). The kernel is purely memory-bound (0.083 FLOP/byte arithmetic intensity). The implementation includes a CPU reference validator, a progressive series of optimized kernel candidates, and an isolated CUDA-event benchmark harness. The primary metric is logical memory bandwidth: `3 * N * sizeof(float) / kernel_time`. The objective is to reach ≥60% of theoretical HBM3 peak bandwidth on H100 SXM5 (≥2.01 TB/s logical) while maintaining correct results within `1e-3` absolute tolerance across all edge-case sizes.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Correctness — GPU output matches CPU reference.
  - Positive Tests (expected to PASS):
    - N = 0: kernel is a no-op; validation passes trivially.
    - N = 1, 2, 3, 4: small vectors; each element `|C_gpu[i] - C_cpu[i]| < 1e-3`.
    - N = 1023, 1024, 1025: non-multiple-of-block, exact-block-multiple, and block-multiple-plus-one.
    - N = 16,777,216 (64 MiB/array): medium size, full validation.
    - N = 268,435,456 (1 GiB/array): large size, sampled validation (every 1000th element).
    - N = 268,435,463 (1 GiB + 7): large non-multiple of 8, sampled validation.
  - Negative Tests (expected to FAIL):
    - Introduce a NaN in input A (e.g. `NAN` constant); output must contain no NaN that wasn't in inputs — but standard addition propagates NaN, so this test confirms expected IEEE behavior. The actual rejection is: if output contains NaN where neither input had NaN, fail.
    - Produce inf by using `FLT_MAX + FLT_MAX`; the test confirms inf output is correct (matches CPU), not rejected.
  - AC-1.1: No silent NaN or inf generation.
    - Positive: Verify `isfinite()` on every output element after addition of finite inputs.
    - Negative: Inject `0.0f/0.0f` at element 0; confirm validation detects and reports the NaN.
  - AC-1.2: All CUDA API calls succeed.
    - Positive: Every `cudaMalloc`, `cudaMemcpy`, `cudaEvent*`, kernel launch returns `cudaSuccess`.
    - Negative: Simulate an invalid launch config; confirm `cudaGetLastError()` catches it.

- AC-2: Baseline kernel serves as a fair scalar reference.
  - Positive Tests:
    - Baseline uses grid-stride loop, `__restrict__`, 64-bit (`size_t`) indexing, and processes ≥4 elements per thread per iteration.
    - Baseline compiles and validates correctly at all edge sizes.
    - Baseline runs at ≥40% of detected platform peak bandwidth on large N.
  - Negative Tests:
    - A strawman single-element-per-thread kernel without `__restrict__` would show artificially low bandwidth; such a kernel must not be the baseline.

- AC-3: Optimized kernel candidates show genuine hardware utilization improvement.
  - Positive Tests:
    - Candidate 1 (float4 vectorization) passes all correctness checks, including tail for non-multiple-of-4 N.
    - At least one candidate produces median logical bandwidth within 5% of the best-measured candidate.
    - The best candidate achieves ≥60% of theoretical HBM3 peak on detected H100 SXM5.
  - Negative Tests:
    - A candidate with register spills (detected via `-Xptxas -v` showing local memory usage) must be marked as degraded.
    - A candidate that produces `cudaErrorIllegalAddress` on edge sizes must be rejected.
  - AC-3.1: Cache-policy candidate is gated.
    - Positive: Compiles only under `__CUDA_ARCH__ >= 900`; produces `ld.global.L1::no_allocate` in SASS (verified via Nsight Compute or `cuobjdump`).
    - Negative: Fails to compile on sm_80 or lower hardware (expected and acceptable).

- AC-4: Benchmark harness provides reliable, isolated kernel timing.
  - Positive Tests:
    - Benchmark uses `cudaEvent_t` with `cudaEventCreate`/`cudaEventRecord`/`cudaEventSynchronize`/`cudaEventElapsedTime` for isolated kernel timing.
    - All allocations, host-device copies, and validation are outside the timed region.
    - Warmup iterations (≥10) are run before timed iterations (≥100).
    - Results include min, median, and standard deviation across timed iterations.
    - Logical bandwidth is computed as `3.0 * N * sizeof(float) / median_time_seconds / 1e9` (GB/s).
  - Negative Tests:
    - Timing that includes `cudaMemcpy` or `cudaDeviceSynchronize` inside the iteration loop would inflate reported time; harness must not do this.
    - Timing on N smaller than L2 cache size (50 MB) conflates cache bandwidth with DRAM bandwidth; harness must warn or reject.

- AC-5: Device-conditional reporting.
  - Positive Tests:
    - On detected H100 SXM5 (device name contains "H100", memory bus width ≥ 5120 bits, computed peak BW ≥ 3000 GB/s): target 2.01 TB/s logical bandwidth.
    - On non-H100 or H100 PCIe: report achieved bandwidth as fraction of detected platform peak. No absolute pass/fail.
  - Negative Tests:
    - Hard-coded "3.35 TB/s" without runtime verification would give misleading percentages on non-SXM hardware.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation includes a single `vector_add.cu` file (~400–600 lines) containing: a CPU reference implementation, a grid-stride scalar baseline kernel with `__restrict__` and 64-bit indexing, a float4-vectorized kernel, an experimental cache-policy kernel gated behind `__CUDA_ARCH__ >= 900`, an ILP-2x float4 kernel, a block-size tunable variant, a runtime device-properties query and bandwidth computation, CLI argument parsing for `--validate`/`--benchmark`/`--size`/`--kernel`, CUDA event-based isolated timing with min/median/stddev reporting, comprehensive edge-case tests including N=0, and templated kernel dispatch functions. All candidates that pass correctness are benchmarked and results recorded. No external dependencies beyond the CUDA runtime and standard C++ library.

### Lower Bound (Minimum Acceptable Scope)

The implementation includes a single `vector_add.cu` file with: a CPU reference, a grid-stride scalar baseline kernel, a float4-vectorized kernel, correctness validation against CPU reference at N=16M and N=256M, CUDA event-based isolated timing reporting average and median bandwidth, and a working compile/run cycle via `nvcc -O2 -std=c++17 -arch=sm_90 -lineinfo vector_add.cu -o vector_add && ./vector_add --benchmark --size 268435456`. At minimum, the float4 kernel must pass all correctness checks.

### Allowed Choices

- Can use: CUDA C++ with runtime API, PTX inline assembly for cache-policy hints (architecture-gated), `float4` vectorized loads/stores, grid-stride loop pattern, `__launch_bounds__`, `__restrict__`, CUDA events for timing, `size_t`/`uint64_t` for all indices, runtime device property queries, templated kernel dispatch with function pointers.
- Cannot use: Third-party libraries (cuBLAS, CUTLASS, Thrust), `cuda::memcpy_async`/`cp.async` (no shared-memory staging needed for this streaming pattern), cooperative groups, TMA (overkill for a 1D streaming op), dynamic parallelism, CUDA graphs.

> The draft specifies a deterministic design: single-file CUDA implementation with progressive optimization candidates evaluated via isolated benchmarks. The path boundaries reflect this specification — the implementation is self-contained within one `.cu` file with no external library dependencies.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The kernel is purely memory-bound. The optimization strategy follows a progression:

```
Baseline (grid-stride scalar + __restrict__)
  → float4 vectorization (128-bit loads/stores, 4 elements/thread/iter)
    → [optional] L1 cache bypass via PTX (#if __CUDA_ARCH__ >= 900)
      → [optional] 2× float4 ILP (8 elements/thread/iter)
        → block-size tuning
```

Each candidate is a separate templated kernel instantiation, selectable at runtime.

**Grid-stride loop pattern** (used by all candidates):

```cuda
template<int VectorWidth, bool UseCacheHints>
__global__ void vector_add_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    // Each thread processes VectorWidth elements per iteration
    size_t elem_idx = idx * VectorWidth;
    while (elem_idx + VectorWidth <= N) {
        // ... vectorized load, add, store ...
        elem_idx += stride * VectorWidth;
    }
    // Tail handling for remaining < VectorWidth elements
}
```

**float4 load/store:** Use `reinterpret_cast<const float4*>` for aligned access. `cudaMalloc` guarantees 256-byte alignment, satisfying float4's 16-byte requirement.

**Cache policy (experimental):** PTX inline asm with `ld.global.L1::no_allocate.v4.f32` replaces the float4 load. Only enabled when `__CUDA_ARCH__ >= 900`. This is a measured optimization — if Nsight Compute profiling shows no improvement or SASS regresses, it should be rejected.

**ILP (2× float4):** Load two float4 values from A, then two from B, before computing. This increases memory-level parallelism. Requires ~16 extra registers; must verify no spilling with `-Xptxas -v`.

**Block size tuning:** Test 128, 256, 512, 1024 threads/block. On H100: 256 or 512 typically works best for memory-bound kernels. Keep the simplest block size within 2% of the best-measured.

### Relevant References

- `/softhome/gehongyu/kernel-design-agents/skills/ncu-report-skill/helpers/harness_template.cu` — CUDA profiling harness template with CUDA_CHECK, alloc_device, CLI parsing, and synthetic data fill helpers.
- `/softhome/gehongyu/kernel-design-agents/skills/ncu-report-skill/helpers/safetensors_loader.h` — Header-only safetensors file reader (not needed for synthetic data benchmarks).
- KernelWiki `technique-vectorized-loads` — Wide vectorized loads (128-bit, 256-bit) and cache-policy differentiation for memory-bound kernels on Hopper/Blackwell.
- KernelWiki `technique-cache-policy` — PTX load cache qualifiers: `L1::no_allocate` for streaming data, `L1::evict_last` for reused data.

## Dependencies and Sequence

### Milestones

1. Milestone M1: Harness infrastructure and baseline kernel.
   - Phase A: Write `vector_add.cu` with CUDA_CHECK, CPU reference, CLI parsing (`--validate`, `--benchmark`, `--size`, `--kernel`).
   - Phase B: Implement the grid-stride scalar baseline kernel with `__restrict__`, `size_t` indexing, and `__launch_bounds__`.
   - Phase C: Implement validation logic (full and sampled comparison with CPU reference, NaN/inf detection).
   - Phase D: Implement CUDA event-based benchmark harness (warmup, timed iterations, min/median/stddev, logical bandwidth computation).
   - Phase E: Add runtime device info printing and peak bandwidth computation.

2. Milestone M2: float4 vectorized kernel candidate.
   - Step 1: Implement templated vector_add kernel with `VectorWidth=4`, scalar loads.
   - Step 2: Replace scalar loads with `reinterpret_cast<const float4*>` and float4 stores.
   - Step 3: Add tail handling: scalar prologue for start alignment, float4 main loop, scalar epilogue for remaining < 4 elements.
   - Step 4: Validate at all edge sizes (N=0..7, 1023..1025, 16M, 16M+1, 256M, 256M+7).

3. Milestone M3: Cache-policy candidate (experimental).
   - Step 1: Add PTX inline asm for `ld.global.L1::no_allocate.v4.f32` on A and B loads, gated behind `#if __CUDA_ARCH__ >= 900`.
   - Step 2: Provide scalar fallback path for non-H100 compilation.
   - Step 3: Benchmark; if < 3% improvement over float4 or SASS inspection shows regression, mark as rejected.

4. Milestone M4: ILP-2x and block-size tuning.
   - Step 1: Implement 2× float4 per iteration (VectorWidth=8).
   - Step 2: Check register usage with `-Xptxas -v`; if spills detected, note the degradation.
   - Step 3: Test block sizes 128, 256, 512, 1024 with the best-so-far kernel.
   - Step 4: Select simplest block within 2% of best; only add `__launch_bounds__` if reproducible > 2% win.

5. Milestone M5: Final benchmarking and result recording.
   - Step 1: Run full benchmark suite across all accepted candidates at N=256M.
   - Step 2: Record results with device info, kernel variant, block size, min/median/stddev time, logical GB/s, and percent of platform peak.
   - Step 3: Document which candidates were accepted, revised, or rejected, with rationale.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task-harness | Write vector_add.cu harness: CUDA_CHECK, CPU reference, CLI parsing, device info query, validation, and CUDA event benchmark infrastructure | AC-4, AC-5 | coding | - |
| task-baseline | Implement grid-stride scalar baseline kernel with __restrict__, size_t indexing, and 4-elem/thread unrolling | AC-1, AC-2 | coding | task-harness |
| task-baseline-validate | Validate baseline: compile, run all edge sizes, verify correctness, record baseline bandwidth | AC-1, AC-2 | coding | task-baseline |
| task-float4 | Implement float4-vectorized kernel with tail handling (prologue/main/epilogue) | AC-1, AC-3 | coding | task-baseline-validate |
| task-float4-validate | Validate and benchmark float4 kernel at all edge sizes; compare against baseline | AC-1, AC-3 | coding | task-float4 |
| task-cache-policy | Implement cache-policy kernel with PTX L1::no_allocate, gated behind #if __CUDA_ARCH__ >= 900 | AC-1, AC-3.1 | coding | task-float4-validate |
| task-cache-validate | Validate and benchmark cache-policy kernel; inspect SASS if H100 available; accept or reject | AC-1, AC-3.1 | coding | task-cache-policy |
| task-ilp2x | Implement ILP-2x float4 kernel with register-spill check | AC-1, AC-3 | coding | task-float4-validate |
| task-ilp2x-validate | Validate, benchmark, and check register usage of ILP-2x kernel | AC-1, AC-3 | coding | task-ilp2x |
| task-tune-blocks | Test block sizes 128/256/512/1024 on best kernel; select simplest within 2% of best | AC-3 | coding | task-float4-validate |
| task-final-benchmark | Run full benchmark suite on all accepted candidates at N=256M; record comprehensive results | AC-3, AC-4, AC-5 | coding | task-cache-validate, task-ilp2x-validate, task-tune-blocks |
| task-assess-plan | After implementation, assess whether plan acceptance criteria are satisfied and recommend promotion/next steps | AC-1 through AC-5 | analyze | task-final-benchmark |

## Claude-Codex Deliberation

### Agreements
- 64-bit (`size_t`/`uint64_t`) indexing is mandatory for correctness with large N.
- A grid-stride scalar baseline with `__restrict__` and unrolling is the correct fair reference, not a single-element-per-thread strawman.
- `cp.async`/cooperative-groups adds complexity with no benefit for a streaming global-to-global operation — rejected upfront.
- CUDA event timing with warmup, isolated kernel measurement, and min/median/stddev reporting is the correct benchmarking discipline.
- float4 alignment is satisfied by `cudaMalloc` (256-byte alignment), but tail handling for non-multiple-of-4 N is required.
- N=0 must be handled as a no-op before any kernel launch.
- Cache-policy PTX must be gated behind `__CUDA_ARCH__ >= 900` with a scalar fallback.
- Bandwidth metric is logical: `3 * N * sizeof(float) / kernel_time`. Actual DRAM bytes may differ slightly due to sector inefficiency.
- The 60%-of-peak bandwidth target is conditional on detected H100 SXM5 hardware.

### Resolved Disagreements
- **Promotion requiring improvement over baseline**: Claude originally proposed "at least one candidate must improve over baseline." Codex objected that a competent scalar baseline may already be near-optimal, making this an unreasonable gate. Resolution: Baseline is a reference; candidates are judged by absolute bandwidth. No candidate is required to exceed baseline. All results are recorded objectively.
- **Cache-policy acceptance metric**: Claude originally used "lower DRAM traffic" as the acceptance test for L1 cache policy. Codex correctly noted that `L1::no_allocate` does not reduce logical bytes read/written — it affects cache allocation behavior. Resolution: Cache-policy candidate requires SASS inspection confirming intended `ld.global.L1::no_allocate` instructions AND ≥3% median improvement over float4-only. If not met, reject.
- **H100 SXM5 detection rule**: Claude's initial "SMs >= 128 && peak BW >= 3000 GB/s" was too loose. Codex required device name matching. Resolution: Detection uses device name containing "H100" AND memory bus width ≥ 5120 bits AND computed peak BW ≥ 3000 GB/s. Non-H100 falls back to platform-relative reporting.
- **Peak bandwidth formula**: Codex noted the initial formula omitted kHz→Hz unit conversion. Resolution: `peak_GBps = 2.0 * memoryClockRate_kHz * 1000.0 * (memoryBusWidth_bits / 8.0) / 1e9`.
- **Compile target**: Codex noted `-arch=sm_90` conflicts with the portability fallback story. Resolution: The binary is explicitly H100/sm_90. The `#if __CUDA_ARCH__ >= 900` guard is compile-time safety, not a deployment target — the kernel is only validated on H100.

### Convergence Status
- Final Status: `converged`
- Rounds executed: 2
- All REQUIRED_CHANGES resolved across both rounds.
- No material disagreements remain.

## Pending User Decisions

- DEC-1: KDA benchmarking exercise vs. production-quality kernel?
  - Claude Position: Assume benchmarking exercise. Multiple candidates with experimental techniques (PTX cache hints, ILP variants) and runtime kernel selection are appropriate. The goal is to demonstrate optimization methodology and measure each technique's contribution.
  - Codex Position: N/A — agrees with Claude's assessment but asks for explicit confirmation.
  - Tradeoff Summary: A benchmarking exercise justifies keeping experimental candidates in the code even if some are rejected. A production kernel would strip all but the single best variant and remove PTX fragility.
  - Decision Status: `PENDING`

- DEC-2: Is H100 SXM5 guaranteed for benchmark evaluation?
  - Claude Position: Assume no — plan for device-conditional reporting. The code detects hardware at runtime and adjusts targets accordingly. If H100 is available, use the 2.01 TB/s target. If not, report platform-relative results.
  - Codex Position: N/A — agrees with the conditional approach but asks for explicit confirmation.
  - Tradeoff Summary: If H100 SXM5 is guaranteed, the 2.01 TB/s target can be treated as a hard promotion gate. If not, the plan's conditional approach is correct and no code changes are needed.
  - Decision Status: `PENDING`

- DEC-3: Runtime kernel selection vs. compile-time variants?
  - Claude Position: Use runtime selection via CLI `--kernel` flag and templated function pointer dispatch. This enables side-by-side benchmarking in a single binary.
  - Codex Position: Agrees runtime selection is fine for a benchmark exercise. Notes that compile-time `#define` would produce smaller binaries but is less convenient for comparison.
  - Tradeoff Summary: Runtime selection is slightly more complex but allows single-binary comparison. Compile-time variants are simpler but require multiple compilations for comparison. For this exercise, runtime is preferred.
  - Decision Status: `PENDING`

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- Use English for all code, comments, and commit messages per repository rules.
- Follow the naming conventions in the harness template: `CUDA_CHECK` macro, `alloc_device<T>` helper, `fill_f32_random` for synthetic data.
- Keep the implementation in a single file: `vector_add.cu`.

---

--- Original Design Draft Start ---

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

--- Original Design Draft End ---