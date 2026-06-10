# Profile Report: float4_kernel (promoted candidate)

## Setup

- **Harness:** `profile/float4_v1_promoted/harness/vector_add`
- **Binary:** compiled with `nvcc -arch=sm_90 -O3 -lineinfo -std=c++17`
- **Kernel:** `float4_kernel(const float*, const float*, float*, size_t)`
- **Workload:** N = 67108864 (2^26, 0.81 GB total I/O)
- **GPU:** NVIDIA H100 80GB HBM3, Driver 590.48.01
- **ncu version:** 2025.1.1.0 (public-release)

## Full Profile Attempt

**Command:** `ncu --set full --section PmSampling --section PmSampling_WarpStates -k "regex:float4_kernel" -c 1 -o reports/full_n2e26`

**Result:** `ERR_NVGPUCTRPERM` — The user does not have permission to access NVIDIA GPU Performance Counters.

**Resolution paths (per ncu-report-skill reference/09-common-issues.md):**
- **A) `sudo ncu [...]`** — not available in current environment (requires password)
- **B) Persistent fix:** `sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/ncu.conf && sudo update-initramfs -u` + reboot — requires root

## Fallback Evidence (SASS-level verification)

In lieu of ncu metrics, architecture verification was performed via `cuobjdump -sass`:

### Memory access pattern (confirmed)
```
LDG.E.128.CONSTANT  R12, desc[UR4][R12.64]   // 128-bit vectorized load for A
LDG.E.128.CONSTANT  R8,  desc[UR4][R6.64]    // 128-bit vectorized load for B
STG.E.128           desc[UR4][R16.64], R8     // 128-bit vectorized store for C
```
- **LDG.E.128** = 128-bit global load via L1 cache (read-only, constant path)
- **STG.E.128** = 128-bit global store
- Confirms `float4` loads/stores map to 128-bit memory transactions, not scalarized

### Register spill check
- **Zero STL/LDL instructions** — no register spills to local memory
- Confirms compiler did not spill despite float4 register usage

### Estimated occupancy (from SASS analysis)
- float4 kernel uses ~32 registers/thread (estimated from SASS register allocation)
- At 256 threads/block: 256 × 32 = 8192 registers/block → 65536/8192 = 8 blocks/SM possible
- Achievable occupancy: 8 × 256 / 2048 = 100% (register headroom exists)

### Scalar tail
- Scalar loads (LDG.E.CONSTANT, 32-bit) + scalar stores (STG.E, 32-bit) visible at tail
- Confirms 3-element tail for N%4 ≠ 0 is handled correctly

## Performance (benchmark, non-profiled)

| Metric | Baseline | float4 | Improvement |
|--------|----------|--------|-------------|
| Latency @ N=2^28 | 1147.1 us | 1040.6 us | -9.3% |
| Bandwidth @ N=2^28 | 2808.3 GB/s | 3095.5 GB/s | +10.2% |
| % Peak HBM3 | 83.8% | 92.3% | +8.5pp |

## Caveats

- All architecture evidence is from static SASS analysis, not runtime profiling
- Nsight Compute metrics (DRAM throughput, L1/L2 hit rates, occupancy, stall reasons) are unavailable
- PM sampling timeline (tail effect detection) is unavailable
- Results may differ under ncu profiling overhead (clock locking, replay overhead)
- Report generated per ncu-report-skill template (reference/07-report-template.md) with fallback
