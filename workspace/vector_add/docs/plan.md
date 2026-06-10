# H100 Vector Addition CUDA Kernel — Implementation Plan

## Goal Description

Design and implement a high-performance, fully optimized element-wise vector addition CUDA kernel (`C[i] = A[i] + B[i]`) in FP32 for NVIDIA H100 (SM90). The kernel must achieve correct results within the task contract tolerance, outperform a fair baseline, and maximize memory bandwidth utilization — this is a deeply memory-bound kernel (0.083 FLOPs/byte arithmetic intensity) so the entire optimization strategy focuses on memory throughput rather than compute.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Correct element-wise FP32 vector addition for all tested N sizes.
  - Positive Tests (expected to PASS):
    - N = 2^28 (268,435,456): all elements match CPU reference within 1e-3 absolute tolerance. No NaN or inf values in output.
    - N = 0: kernel returns immediately, no output written, no CUDA errors.
    - N = 1, 127, 1023 (non-multiples of 4): scalar tail correctly handles remaining elements.
    - Input data patterns: all zeros, all ones, alternating ±1, powers of two, uniform random [-1, 1], uniform random [-1000, 1000], values near FLT_MAX/2.
    - All power-of-two N from 2^20 to 2^28.
  - Negative Tests (expected to FAIL):
    - A kernel that produces NaN or inf output for finite inputs is rejected.
    - A kernel that passes N = 2^28 but fails N = 1023 (tail bug) is rejected.
    - A kernel with any element exceeding 1e-3 absolute error is rejected.
  - AC-1.1: Correctness stability across repeated runs.
    - Positive: 5 consecutive validation runs all return PASS with consistent max error.
    - Negative: Intermittent correctness failures (race conditions, out-of-bounds) cause rejection.

- AC-2: Fair baseline kernel that represents a competent starting point (not a strawman).
  - Positive Tests (expected to PASS):
    - Baseline uses: grid-stride loop, `const float* __restrict__` on inputs, `float* __restrict__` on output, `size_t` indexing, 256 or 512 threads/block.
    - Baseline compiles and passes all AC-1 tests.
    - Baseline benchmark produces stable, reproducible median latency at N = 2^28 (IQR < 10% of median).
  - Negative Tests (expected to FAIL):
    - A "naive" kernel without grid-stride or `__restrict__` is rejected as baseline — it misrepresents what a reasonable developer would write.
    - A baseline that uses `int` indexing and overflows at N = 2^28 is rejected.

- AC-3: At least one candidate kernel measurably outperforms the baseline.
  - Positive Tests (expected to PASS):
    - Candidate median bandwidth exceeds baseline median bandwidth, with the improvement direction confirmed across the power-of-two N sweep (2^20 through 2^28).
    - The improvement is visible at large N (>= 2^26) where kernel launch overhead is amortized.
  - Negative Tests (expected to FAIL):
    - A candidate that regresses at any tested N compared to baseline is investigated; persistent regressions at N >= 2^26 block promotion.
    - A candidate whose bandwidth improvement is smaller than the timing noise band (IQR overlap) is treated as "noise-level" and not promoted on performance grounds alone.

- AC-4: Reliable benchmark harness with proper GPU timing methodology.
  - Positive Tests (expected to PASS):
    - Uses CUDA events (`cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`).
    - Calls `cudaGetLastError()` and `cudaDeviceSynchronize()` after every kernel launch.
    - Runs 10 warm-up iterations + 50 timed iterations; reports trimmed mean (discard top/bottom 20%), median, and IQR.
    - Computes effective bandwidth as `3 * N * sizeof(float) / latency_median` bytes/second.
    - Prints H100 device identification via `cudaGetDeviceProperties` (device name, compute capability, theoretical memory bandwidth).
    - Appends structured results to `benchmark.csv`.
  - Negative Tests (expected to FAIL):
    - Harness without CUDA event synchronization is rejected (wall-clock timing includes launch overhead).
    - Harness without error checking after kernel launch is rejected.
    - Harness that does not report variance/IQR is rejected.

- AC-5: Architecture evidence for the final promoted candidate.
  - Positive Tests (expected to PASS):
    - SASS inspection (`cuobjdump -sass`) confirms vectorized load/store instructions (LDG.128 / STG.128 for float4 paths).
    - SASS confirms no local memory spills (no LDL/STL to stack) in the promoted candidate.
    - Nsight Compute metrics recorded: DRAM throughput, L2 throughput, achieved occupancy, register count, global load/store efficiency.
    - Candidate metadata recorded in `candidates.jsonl` with: name, parent, block_size, vector_width, latency_median_us, bandwidth_median_gbs, correctness, promotion_decision.
  - Negative Tests (expected to FAIL):
    - A candidate promoted without SASS verification is blocked — unexpected instruction lowering (e.g., scalarized float4) invalidates performance claims.
    - A candidate with register spills (STL/LDL detected in SASS) requires investigation before promotion.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

A single `vector_add.cu` file (under 600 lines) containing:
- CPU reference implementation
- CLI-driven validation and benchmark harness with CUDA event timing
- 6 candidate kernels as separate `__global__` functions:
  1. `baseline`: grid-stride scalar with `__restrict__` and `size_t`
  2. `float2`: 2 elements per thread via `float2` loads/stores
  3. `float4`: 4 elements per thread via `float4` loads/stores (primary candidate)
  4. `ilp2x`: 8 elements per thread via two `float4` operations per iteration
  5. `cache_hint`: float4 with L1::no_allocate PTX inline assembly (experimental, `#ifdef USE_PTX_CACHE_HINTS` gated, with pure CUDA fallback)
  6. `block_sweep`: float4 kernel templated on `blockDim` for block size sweep {128, 256, 512, 1024}
- Alignment-aware dispatch: check 16-byte alignment of all three pointers; fall back to scalar path if any are misaligned
- Scalar tail loop for `N % vector_width` remaining elements
- Multi-N benchmark sweep (2^20 through 2^28)
- SASS dump and Nsight Compute profiling for the final promoted candidate
- Artifact output to `benchmark.csv` and `candidates.jsonl`
- `-maxrregcount` experiment as a separate build variant (not compiled into the default binary)

### Lower Bound (Minimum Acceptable Scope)

A single `vector_add.cu` file containing:
- CPU reference implementation
- CLI harness with PASS/FAIL output and `--benchmark` mode
- `baseline` kernel (grid-stride, `__restrict__`, `size_t`)
- `float4` candidate kernel with alignment check and scalar tail
- Correctness validation at N = 2^28 (all power-of-two sizes from 2^20 to 2^28 strongly preferred)
- Benchmark with CUDA events, warm-up + timed iterations, median + IQR reporting
- `benchmark.csv` recording for baseline and float4 candidate

### Allowed Choices

- Can use: `float2`/`float4` vectorized types, `__restrict__` qualifiers, grid-stride loops, `constexpr` block sizes, `__launch_bounds__`, PTX inline assembly (experimental only, with CUDA C++ fallback), `-maxrregcount` (as build variant), `cuobjdump` for SASS inspection, Nsight Compute for profiling
- Cannot use: third-party libraries (cuBLAS, CUTLASS, Thrust), tensor cores (no `wgmma`/`tcgen05`), TMA (`cuda::memcpy_async`), cooperative groups, dynamic parallelism, CUDA graphs, `cudaMallocManaged`, unified memory

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

The implementation follows a single-file design with all kernels in one translation unit, dispatched from a CLI harness:

```
vector_add.cu
├── CPU reference:    void vector_add_cpu(const float*, const float*, float*, size_t)
├── Validation:       bool validate(const float*, const float*, size_t, float max_err)
├── Benchmark:        void benchmark(kernel_fn, config, benchmark_result&)
├── Baseline kernel:  __global__ void baseline(...)
├── float4 kernel:    __global__ void float4_add(...)
├── float2 kernel:    __global__ void float2_add(...)
├── ilp2x kernel:     __global__ void ilp2x_add(...)
├── cache_hint kernel: __global__ void cache_hint_add(...)
├── block_sweep:      template<int BLOCK> __global__ void float4_templated(...)
├── main():           parse --benchmark / --candidate / --validate flags, dispatch, report
└── device_info():    print cudaGetDeviceProperties for SKU identification
```

Key design patterns:
- All kernels use `const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, size_t N`.
- Grid calculation: `size_t total_threads = (N + elems_per_thread - 1) / elems_per_thread; size_t blocks = (total_threads + blockDim - 1) / blockDim;`
- Vectorized kernels compute `size_t idx = blockIdx.x * blockDim.x + threadIdx.x` (in float4 units), then `size_t base = idx * 4`. The main loop processes full float4 chunks; a tail handles `N % 4`.
- Alignment check: `uintptr_t ptr_a = reinterpret_cast<uintptr_t>(A); bool aligned = (ptr_a % 16 == 0) && (ptr_b % 16 == 0) && (ptr_c % 16 == 0);`
- Benchmark harness: `cudaEventRecord(start)`, kernel launch, `cudaEventRecord(stop)`, `cudaEventSynchronize(stop)`, `cudaEventElapsedTime(&ms, start, stop)`.
- H100 SKU detection: `cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);` — record `prop.name`, `prop.major.minor`, theoretical bandwidth from `prop.memoryClockRate * prop.memoryBusWidth / 8`.

### Relevant References

- `~/kernel-design-agents/skills/KernelWiki/wiki/techniques/vectorized-loads.md` — 128/256-bit load PTX, cache policy differentiation patterns
- `~/kernel-design-agents/skills/KernelWiki/wiki/patterns/memory-bound.md` — memory-bound kernel optimization checklist and caveats
- `~/kernel-design-agents/skills/ncu-report-skill/helpers/harness_template.cu` — CUDA profiling harness template with `CUDA_CHECK` macro and alloc patterns
- H100 whitepaper: 3,352 GB/s peak HBM3 bandwidth (SXM5), 132 SMs, 50 MB L2
- CUDA 13.1 Programming Guide: SM90 architecture, `__launch_bounds__`, PTX ISA reference for `ld.global.L1::no_allocate`

## Dependencies and Sequence

### Milestones

1. **M1: Unified Harness** — Create `vector_add.cu` with CPU reference, CLI (--validate / --benchmark / --candidate), CUDA event timing, error checking macros, device info printing.
   - Phase A: Implement CPU reference `vector_add_cpu()` with element-wise FP32 add.
   - Phase B: Implement CLI framework using `argc/argv` parsing.
   - Phase C: Implement `benchmark()` function with CUDA events, warm-up + timed iterations, trimmed mean statistics.
   - Phase D: Implement `validate()` function comparing GPU output against CPU reference with 1e-3 tolerance.
   - Phase E: Implement `device_info()` printing `cudaGetDeviceProperties` data.

2. **M2: Baseline Kernel + Validation** — Implement and validate the baseline kernel.
   - Step 1: Write `baseline` kernel: grid-stride loop, `__restrict__`, `size_t`.
   - Step 2: Run validation at all power-of-two N from 2^20 to 2^28.
   - Step 3: Run benchmark at N = 2^28, record baseline performance in `benchmark.csv`.
   - Depends on: M1.

3. **M3: float4 Candidate** — Primary vectorized candidate with alignment handling.
   - Step 1: Write `float4_add` kernel with `float4` loads/stores.
   - Step 2: Implement alignment check at dispatch: if any pointer is not 16-byte aligned, fall back to baseline.
   - Step 3: Implement scalar tail for `N % 4 != 0`.
   - Step 4: Validate correctness at all power-of-two N + irregular N (127, 1023).
   - Step 5: Benchmark at all power-of-two N, record in `benchmark.csv` and `candidates.jsonl`.
   - Depends on: M2.

4. **M4: Additional Candidates** — float2, ILP-2x, and comparative analysis.
   - Step 1: Write `float2_add` kernel.
   - Step 2: Write `ilp2x_add` kernel (two float4 per thread per iteration).
   - Step 3: Benchmark all candidates, identify top performer.
   - Depends on: M3.

5. **M5: Tuning Experiments** — Block size sweep, register budget, cache hints.
   - Step 1: Write `float4_templated<BLOCK>` and benchmark {128, 256, 512, 1024}.
   - Step 2: Build with `-maxrregcount={32,40,48,56,64}` and benchmark best block size.
   - Step 3: Write `cache_hint_add` kernel (PTX `L1::no_allocate`, `#ifdef USE_PTX_CACHE_HINTS` gated) and compare against float4.
   - Depends on: M4.

6. **M6: Final Candidate Promotion** — SASS/PTX inspection, Nsight profiling, promotion decision.
   - Step 1: Run `cuobjdump -sass vector_add` on the best candidate; verify LDG.128/STG.128, no STL/LDL spills.
   - Step 2: Run Nsight Compute with metrics: `dram__throughput`, `l1tex__throughput`, `sm__throughput`, `achieved_occupancy`, register count, global load/store efficiency.
   - Step 3: Select promoted candidate, update `candidates.jsonl` with promotion decision.
   - Step 4: Record final results in `benchmark.csv`.
   - Depends on: M5.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task-harness | Create unified `vector_add.cu` with CPU reference, CLI, CUDA event timing, error checking, device info, benchmark stats | AC-4 | coding | - |
| task-baseline | Implement `baseline` kernel (grid-stride, `__restrict__`, `size_t`) and validate at all power-of-two N | AC-1, AC-2 | coding | task-harness |
| task-float4 | Implement `float4_add` kernel with alignment check and scalar tail; validate and benchmark | AC-1, AC-3 | coding | task-baseline |
| task-float2 | Implement `float2_add` kernel and benchmark vs baseline | AC-3 | coding | task-float4 |
| task-ilp2x | Implement `ilp2x_add` kernel (2x float4 per thread) and benchmark | AC-3 | coding | task-float4 |
| task-block-sweep | Implement templated `float4_templated<BLOCK>` and benchmark {128,256,512,1024} | AC-3 | coding | task-float4 |
| task-reg-sweep | Build with `-maxrregcount={32,40,48,56,64}`, benchmark best block size variant | AC-3 | coding | task-block-sweep |
| task-cache-hint | Implement `cache_hint_add` kernel (PTX L1::no_allocate, `#ifdef` gated with CUDA fallback) | AC-3 | coding | task-float4 |
| task-benchmark-all | Run full multi-N sweep for all candidates, record in benchmark.csv and candidates.jsonl | AC-3, AC-4 | coding | task-float2, task-ilp2x, task-block-sweep, task-reg-sweep, task-cache-hint |
| task-sass-inspect | Run `cuobjdump -sass` on promoted candidate; verify LDG.128/STG.128 and no spills | AC-5 | coding | task-benchmark-all |
| task-ncu-profile | Run Nsight Compute on promoted candidate; record DRAM/L2/occupancy metrics | AC-5 | coding | task-benchmark-all |
| task-codex-review | Codex final review: assess candidate ranking, verify promotion rationale, check benchmark methodology | AC-3, AC-4, AC-5 | analyze | task-ncu-profile, task-sass-inspect |

## Claude-Codex Deliberation

### Agreements

- Baseline should be a fair grid-stride kernel with `__restrict__` and `size_t`, not a naive single-element strawman.
- Multi-N benchmark sweep (2^20 to 2^28) is essential for meaningful performance comparison; single N=2^28 can hide tail behavior and launch overhead.
- float4 vectorized loads/stores is the primary optimization lever; all other candidates are secondary refinements.
- Benchmark.csv and candidates.jsonl recording is required per the KDA workflow.
- Edge case handling (N=0, non-multiples of vector width, scalar tails) must be explicit and tested.
- Occupancy should not be a rigid promotion criterion on its own — it is an enabler, not a goal.
- SASS/PTX inspection for the final promoted candidate is necessary to verify emitted instruction quality.
- L1 cache policy hints via inline PTX are experimental and must be gated behind `#ifdef`, with a pure CUDA C++ fallback.
- H100 SXM vs PCIe distinction matters for performance targets; device SKU must be recorded at benchmark time.
- Store policy (streaming vs. write-back) is a valid optimization dimension; default write-back is correct for the host-readback pipeline.

### Resolved Disagreements

- **float4 impact magnitude**: Claude originally expected 1.5-2x gain; Codex noted scalar loads already coalesce at warp level. Resolved: float4 impact is measured, not assumed. The promotion criterion is measured improvement, not a preset multiplier.
- **Inline PTX scope**: Claude favored PTX as a standard candidate; Codex preferred pure CUDA C++ unless PTX proves necessary. Resolved: PTX is experimental only, `#ifdef` gated, with mandatory CUDA fallback and SASS verification.
- **ILP-2x benefit**: Claude thought it may help; Codex was skeptical. Resolved: ILP-2x is a separate candidate measured against float4; if no improvement, it is rejected.
- **Promotion threshold**: Claude initially said "clear improvement"; Codex requested a specific threshold. Resolved per user decision: >5% median bandwidth improvement is a guideline; candidates with smaller but consistent improvement can still be promoted with evidence.
- **Nsight Compute timing**: Claude suggested final candidate only; Codex preferred per-candidate profiling. Resolved: Nsight runs on the promoted candidate; per-candidate profiling is deferred to keep the candidate loop fast.
- **AC-1 tolerance standard**: Claude initially had dual "bit-exact" and "1e-3 tolerance" language. Resolved per user decision: 1e-3 absolute tolerance is the hard correctness gate; bit-exact match for normal finite values is a desirable property but not a separate gate.
- **Candidate count**: Claude listed 6 candidates but enumerated 7 (baseline + float2 + float4 + ilp2x + cache_hint + block_sweep + reg_sweep). Resolved: 7 distinct candidates, counted as 7.

### Convergence Status

- Final Status: `converged`
- Convergence rounds: 2 (Codex v1 analysis + 2 Codex review rounds)
- Second Codex review: CONVERGED, zero REQUIRED_CHANGES, zero UNRESOLVED

## Pending User Decisions

All user decisions have been resolved through the gen-plan process:

- DEC-1: Promotion performance threshold (>5% improvement)
  - Claude Position: >5% median bandwidth improvement at N >= 2^26 as a guideline
  - Codex Position: Specific numeric threshold to eliminate noise-level promotions
  - User Decision: Guideline/direction — candidates with smaller improvement can be promoted with evidence
  - Decision Status: RESOLVED

- DEC-2: Expected throughput target (2,500-2,900 GB/s)
  - Claude Position: Expected outcome range based on theoretical analysis
  - Codex Position: Must account for SXM vs PCIe distinction
  - User Decision: Hard requirement — for H100 SXM5; adjusted proportionally for other SKUs
  - Decision Status: RESOLVED

- DEC-3: Correctness tolerance (1e-3 absolute error)
  - Claude Position: 1e-3 is the task contract tolerance; bit-exact desirable for normal values
  - Codex Position: FP32 add should be bit-exact with same operation order
  - User Decision: 1e-3 is the hard correctness gate
  - Decision Status: RESOLVED

- DEC-4: Benchmark N size sweep
  - Claude Position: Multi-N sweep from 2^20 to 2^28
  - Codex Position: Include non-power-of-two sizes for tail coverage
  - User Decision: Power-of-two sweep (2^20 through 2^28)
  - Decision Status: RESOLVED

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- All code, comments, and documentation must be in English.
- Unified single-file implementation: `vector_add.cu`.

### Benchmark Artifact Schema

**benchmark.csv** columns:
```
candidate_name,N,block_size,vector_width,elems_per_thread,iter_warmup,iter_timed,latency_median_us,latency_iqr_us,bandwidth_median_gbs,device_name,compute_capability,theoretical_bw_gbs,compiler_flags,correctness_result,timestamp_utc
```

**candidates.jsonl** schema (one JSON object per line):
```json
{"name":"baseline","parent":null,"status":"baseline","block_size":256,"vector_width":1,"elems_per_thread":1,"latency_median_us":1234.5,"bandwidth_median_gbs":2100.0,"correctness":"PASS","promotion_decision":"baseline_reference","notes":"","timestamp_utc":"2026-06-10T00:00:00Z"}
```

## Output File Convention

The plan is written to `docs/plan.md`. No translated variant is generated (`alternative_plan_language` is disabled).

---
--- Original Design Draft Start ---

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

--- Original Design Draft End ---
