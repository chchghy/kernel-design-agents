// ============================================================================
// vector_add.cu — High-performance FP32 element-wise vector addition for H100
//
// C[i] = A[i] + B[i]   (FP32, deeply memory-bound)
//
// Compile:  nvcc -arch=sm_90 -O3 -lineinfo vector_add.cu -o vector_add
// Usage:    ./vector_add --validate
//           ./vector_add --benchmark [--size N]
//           ./vector_add --candidate <name> [--size N]
//
// Optional: nvcc -arch=sm_90 -O3 -lineinfo -DUSE_PTX_CACHE_HINTS \
//              -maxrregcount=<N> vector_add.cu -o vector_add
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CUDA error helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err = (expr);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,  \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// CPU reference
// ============================================================================
static void vector_add_cpu(const float *A, const float *B, float *C,
                           size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// Kernel 1: baseline — grid-stride scalar, __restrict__, size_t indexing
// ============================================================================
__global__ void baseline_kernel(const float *__restrict__ A,
                                 const float *__restrict__ B,
                                 float *__restrict__ C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// Kernel 2: float2 — 2 elements per thread via float2 loads/stores
// ============================================================================
__global__ void float2_kernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;

    // Process two elements per thread, main loop
    size_t i = tid;
    while (i + 1 < N) {
        float2 a = reinterpret_cast<const float2 *>(A)[i / 2];
        float2 b = reinterpret_cast<const float2 *>(B)[i / 2];
        float2 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        reinterpret_cast<float2 *>(C)[i / 2] = c;
        i += total_threads;
    }

    // Scalar tail
    if (i < N && i + 1 >= N) {
        C[i] = A[i] + B[i];
    }
}

// ============================================================================
// Kernel 3: float4 — primary candidate, 128-bit vectorized loads/stores
// ============================================================================
__global__ void float4_kernel(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;

    // Process 4 elements per thread
    size_t i = tid * 4;
    while (i + 3 < N) {
        float4 a = reinterpret_cast<const float4 *>(A)[i / 4];
        float4 b = reinterpret_cast<const float4 *>(B)[i / 4];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4 *>(C)[i / 4] = c;
        i += total_threads * 4;
    }

    // Scalar tail for N % 4 != 0
    if (i < N) {
        for (size_t j = i; j < N; ++j) {
            C[j] = A[j] + B[j];
        }
    }
}

// ============================================================================
// Kernel 4: ilp2x — two float4 per iteration (8 elements/thread)
// ============================================================================
__global__ void ilp2x_kernel(const float *__restrict__ A,
                              const float *__restrict__ B,
                              float *__restrict__ C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;

    size_t i = tid * 8;
    while (i + 7 < N) {
        float4 a0 = reinterpret_cast<const float4 *>(A)[i / 4];
        float4 b0 = reinterpret_cast<const float4 *>(B)[i / 4];
        float4 c0;
        c0.x = a0.x + b0.x;
        c0.y = a0.y + b0.y;
        c0.z = a0.z + b0.z;
        c0.w = a0.w + b0.w;
        reinterpret_cast<float4 *>(C)[i / 4] = c0;

        float4 a1 = reinterpret_cast<const float4 *>(A)[(i / 4) + 1];
        float4 b1 = reinterpret_cast<const float4 *>(B)[(i / 4) + 1];
        float4 c1;
        c1.x = a1.x + b1.x;
        c1.y = a1.y + b1.y;
        c1.z = a1.z + b1.z;
        c1.w = a1.w + b1.w;
        reinterpret_cast<float4 *>(C)[(i / 4) + 1] = c1;

        i += total_threads * 8;
    }

    // Scalar tail
    if (i < N) {
        for (size_t j = i; j < N; ++j) {
            C[j] = A[j] + B[j];
        }
    }
}

// ============================================================================
// Kernel 5: cache_hint — float4 with L1::no_allocate PTX (experimental)
// ============================================================================
#ifdef USE_PTX_CACHE_HINTS
__global__ void cache_hint_kernel(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;

    size_t i = tid * 4;
    while (i + 3 < N) {
        float4 a, b, c;
        // PTX 128-bit loads with L1::no_allocate for streaming access
        asm volatile(
            "ld.global.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(a.x), "=f"(a.y), "=f"(a.z), "=f"(a.w)
            : "l"(reinterpret_cast<const float4 *>(A) + i / 4));
        asm volatile(
            "ld.global.L1::no_allocate.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(b.x), "=f"(b.y), "=f"(b.z), "=f"(b.w)
            : "l"(reinterpret_cast<const float4 *>(B) + i / 4));

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4 *>(C)[i / 4] = c;

        i += total_threads * 4;
    }

    if (i < N) {
        for (size_t j = i; j < N; ++j) {
            C[j] = A[j] + B[j];
        }
    }
}
#else
// Pure CUDA C++ fallback — identical to float4_kernel
__global__ void cache_hint_kernel(const float *__restrict__ A,
                                   const float *__restrict__ B,
                                   float *__restrict__ C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;

    size_t i = tid * 4;
    while (i + 3 < N) {
        float4 a = reinterpret_cast<const float4 *>(A)[i / 4];
        float4 b = reinterpret_cast<const float4 *>(B)[i / 4];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4 *>(C)[i / 4] = c;
        i += total_threads * 4;
    }

    if (i < N) {
        for (size_t j = i; j < N; ++j) {
            C[j] = A[j] + B[j];
        }
    }
}
#endif

// ============================================================================
// Kernel 6: block_sweep — float4 templated on block size
// ============================================================================
template <int BLOCK_SIZE>
__global__ void float4_templated_kernel(const float *__restrict__ A,
                                         const float *__restrict__ B,
                                         float *__restrict__ C, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = blockDim.x * gridDim.x;

    size_t i = tid * 4;
    while (i + 3 < N) {
        float4 a = reinterpret_cast<const float4 *>(A)[i / 4];
        float4 b = reinterpret_cast<const float4 *>(B)[i / 4];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4 *>(C)[i / 4] = c;
        i += total_threads * 4;
    }

    if (i < N) {
        for (size_t j = i; j < N; ++j) {
            C[j] = A[j] + B[j];
        }
    }
}

// Explicit instantiations for ncu symbol resolution
template __global__ void
float4_templated_kernel<128>(const float *__restrict__,
                              const float *__restrict__, float *__restrict__,
                              size_t);
template __global__ void
float4_templated_kernel<256>(const float *__restrict__,
                              const float *__restrict__, float *__restrict__,
                              size_t);
template __global__ void
float4_templated_kernel<512>(const float *__restrict__,
                              const float *__restrict__, float *__restrict__,
                              size_t);
template __global__ void
float4_templated_kernel<1024>(const float *__restrict__,
                               const float *__restrict__, float *__restrict__,
                               size_t);

// ============================================================================
// Utility: alignment check for 16-byte aligned pointers
// ============================================================================
static bool is_aligned_16(const void *a, const void *b, const void *c) {
    return (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
           (reinterpret_cast<uintptr_t>(b) % 16 == 0) &&
           (reinterpret_cast<uintptr_t>(c) % 16 == 0);
}

// ============================================================================
// Data fill helpers
// ============================================================================
static void fill_random(std::vector<float> &h, uint64_t seed, float lo,
                        float hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto &x : h)
        x = d(rng);
}

// ============================================================================
// Validation
// ============================================================================
struct ValidationResult {
    bool passed;
    double max_error;
    size_t error_count;
    bool has_nan;
    bool has_inf;
};

static ValidationResult validate(const float *gpu_out, const float *cpu_ref,
                                  size_t N, double tolerance) {
    ValidationResult r = {true, 0.0, 0, false, false};
    for (size_t i = 0; i < N; ++i) {
        if (std::isnan(gpu_out[i])) {
            r.has_nan = true;
            r.passed = false;
        }
        if (std::isinf(gpu_out[i])) {
            r.has_inf = true;
            r.passed = false;
        }
        double err = std::abs(static_cast<double>(gpu_out[i]) -
                               static_cast<double>(cpu_ref[i]));
        if (err > r.max_error)
            r.max_error = err;
        if (err > tolerance) {
            r.passed = false;
            ++r.error_count;
        }
    }
    return r;
}

// ============================================================================
// Device info
// ============================================================================
static void print_device_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double theoretical_bw =
        2.0 * static_cast<double>(prop.memoryClockRate) * 1e3   // kHz->Hz, DDR
        * (prop.memoryBusWidth / 8)                               // bits->bytes
        / 1e12;                                                   // bytes->TB/s

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max Threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Blocks/SM: %d\n",
           prop.maxBlocksPerMultiProcessor);
    printf("Registers/SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared Memory/SM: %zu KB\n",
           prop.sharedMemPerMultiprocessor / 1024);
    printf("Memory Clock: %.0f MHz\n",
           static_cast<double>(prop.memoryClockRate) / 1000.0);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Theoretical Bandwidth: %.1f GB/s\n", theoretical_bw * 1000.0);
    printf("Global Memory: %.1f GB\n",
           static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 *
                                                        1024.0));
}

// ============================================================================
// Benchmark harness
// ============================================================================
struct BenchmarkConfig {
    std::string name;
    int block_size;
    int vector_width;   // 1=scalar, 2=float2, 4=float4
    int elems_per_iter; // elements processed per thread per iteration
    void (*launch_fn)(const float *, const float *, float *, size_t, int);
};

struct BenchmarkResult {
    std::string name;
    size_t N;
    int block_size;
    int vector_width;
    int elems_per_iter;
    double latency_median_us;
    double latency_trimmed_mean_us;
    double latency_iqr_us;
    double bandwidth_median_gbs;
    int iter_warmup;
    int iter_timed;
    bool correctness;
    std::string device_name;
    int compute_capability_major;
    int compute_capability_minor;
    double theoretical_bw_gbs;
    std::string compiler_flags;
    std::string timestamp_utc;
};

static double compute_median(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0)
        return 0.0;
    if (n % 2 == 1)
        return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

static double compute_iqr(const std::vector<double> &v_sorted) {
    size_t n = v_sorted.size();
    if (n < 2)
        return 0.0;
    size_t q1_idx = n / 4;
    size_t q3_idx = (3 * n) / 4;
    return v_sorted[q3_idx] - v_sorted[q1_idx];
}

static double compute_trimmed_mean(const std::vector<double> &v) {
    std::vector<double> sorted = v;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    size_t trim = n / 5; // trim top/bottom 20%
    if (2 * trim >= n)
        return compute_median(sorted);
    double sum = 0.0;
    for (size_t i = trim; i < n - trim; ++i)
        sum += sorted[i];
    return sum / static_cast<double>(n - 2 * trim);
}

static BenchmarkResult
run_benchmark(const BenchmarkConfig &cfg, const float *d_A, const float *d_B,
              float *d_C, size_t N, int warmup_iters, int timed_iters) {
    BenchmarkResult res;
    res.name = cfg.name;
    res.N = N;
    res.block_size = cfg.block_size;
    res.vector_width = cfg.vector_width;
    res.elems_per_iter = cfg.elems_per_iter;
    res.iter_warmup = warmup_iters;
    res.iter_timed = timed_iters;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    res.device_name = prop.name;
    res.compute_capability_major = prop.major;
    res.compute_capability_minor = prop.minor;
    res.theoretical_bw_gbs =
        static_cast<double>(prop.memoryClockRate) * 1e3 *
        (prop.memoryBusWidth / 8) / 1e9;

#ifdef USE_PTX_CACHE_HINTS
    res.compiler_flags = "-arch=sm_90 -O3 -DUSE_PTX_CACHE_HINTS";
#else
    res.compiler_flags = "-arch=sm_90 -O3";
#endif

    // Timestamp
    time_t now = time(nullptr);
    char ts_buf[32];
    strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    res.timestamp_utc = ts_buf;

    // Warm-up
    for (int iter = 0; iter < warmup_iters; ++iter) {
        cfg.launch_fn(d_A, d_B, d_C, N, cfg.block_size);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed iterations
    std::vector<double> times;
    times.reserve(timed_iters);
    for (int iter = 0; iter < timed_iters; ++iter) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        cfg.launch_fn(d_A, d_B, d_C, N, cfg.block_size);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(static_cast<double>(ms) * 1000.0); // ms -> us

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaGetLastError());
    }

    // Statistics
    std::vector<double> times_sorted = times;
    std::sort(times_sorted.begin(), times_sorted.end());
    res.latency_median_us = compute_median(times);
    res.latency_trimmed_mean_us = compute_trimmed_mean(times);
    res.latency_iqr_us = compute_iqr(times_sorted);
    double latency_sec = res.latency_median_us / 1e6;
    res.bandwidth_median_gbs = (latency_sec > 0)
                                    ? (3.0 * N * sizeof(float)) / latency_sec /
                                          1e9
                                    : 0.0;

    // Correctness check (one pass after benchmark)
    std::vector<float> h_C(N), h_C_cpu(N);
    std::vector<float> h_A(N), h_B(N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, N * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, N * sizeof(float),
                           cudaMemcpyDeviceToHost));
    vector_add_cpu(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    ValidationResult vr = validate(h_C.data(), h_C_cpu.data(), N, 1e-3);
    res.correctness = vr.passed;

    // Print result
    printf("  %-20s  N=%9zu  median=%9.1f us  trim_mean=%9.1f us  IQR=%6.1f us  BW=%8.1f GB/s  "
           "%s\n",
           cfg.name.c_str(), N, res.latency_median_us, res.latency_trimmed_mean_us,
           res.latency_iqr_us, res.bandwidth_median_gbs,
           vr.passed ? "PASS" : "FAIL");
    if (!vr.passed) {
        printf("    max_error=%.6e  errors=%zu  NaN=%d  inf=%d\n",
               vr.max_error, vr.error_count, vr.has_nan, vr.has_inf);
    }

    return res;
}

// ============================================================================
// Launch helpers
// ============================================================================

static void launch_baseline(const float *d_A, const float *d_B, float *d_C,
                            size_t N, int block_size) {
    if (N == 0) return;
    size_t grid = ((N + block_size - 1) / block_size);
    baseline_kernel<<<grid, block_size>>>(d_A, d_B, d_C, N);
}

static void launch_float2(const float *d_A, const float *d_B, float *d_C,
                          size_t N, int block_size) {
    if (N == 0) return;
    size_t elems_per_thread = 2;
    size_t total_threads = (N + elems_per_thread - 1) / elems_per_thread;
    size_t grid = (total_threads + block_size - 1) / block_size;
    float2_kernel<<<grid, block_size>>>(d_A, d_B, d_C, N);
}

static void launch_float4(const float *d_A, const float *d_B, float *d_C,
                          size_t N, int block_size) {
    if (N == 0) return;
    if (!is_aligned_16(d_A, d_B, d_C)) {
        // Fall back to baseline for misaligned pointers
        launch_baseline(d_A, d_B, d_C, N, block_size);
        return;
    }
    size_t elems_per_thread = 4;
    size_t total_threads = (N + elems_per_thread - 1) / elems_per_thread;
    size_t grid = (total_threads + block_size - 1) / block_size;
    float4_kernel<<<grid, block_size>>>(d_A, d_B, d_C, N);
}

static void launch_ilp2x(const float *d_A, const float *d_B, float *d_C,
                         size_t N, int block_size) {
    if (N == 0) return;
    if (!is_aligned_16(d_A, d_B, d_C)) {
        launch_float4(d_A, d_B, d_C, N, block_size);
        return;
    }
    size_t elems_per_thread = 8;
    size_t total_threads = (N + elems_per_thread - 1) / elems_per_thread;
    size_t grid = (total_threads + block_size - 1) / block_size;
    ilp2x_kernel<<<grid, block_size>>>(d_A, d_B, d_C, N);
}

static void launch_cache_hint(const float *d_A, const float *d_B, float *d_C,
                              size_t N, int block_size) {
    if (N == 0) return;
    if (!is_aligned_16(d_A, d_B, d_C)) {
        launch_float4(d_A, d_B, d_C, N, block_size);
        return;
    }
    size_t elems_per_thread = 4;
    size_t total_threads = (N + elems_per_thread - 1) / elems_per_thread;
    size_t grid = (total_threads + block_size - 1) / block_size;
    cache_hint_kernel<<<grid, block_size>>>(d_A, d_B, d_C, N);
}

template <int BLOCK_SIZE>
static void launch_float4_templated(const float *d_A, const float *d_B,
                                     float *d_C, size_t N, int /*block_size*/) {
    if (N == 0) return;
    if (!is_aligned_16(d_A, d_B, d_C)) {
        launch_baseline(d_A, d_B, d_C, N, BLOCK_SIZE);
        return;
    }
    size_t elems_per_thread = 4;
    size_t total_threads = (N + elems_per_thread - 1) / elems_per_thread;
    size_t grid = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float4_templated_kernel<BLOCK_SIZE>
        <<<grid, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
}

// ============================================================================
// CSV / JSONL output
// ============================================================================
static void append_benchmark_csv(const std::string &path,
                                  const BenchmarkResult &res) {
    FILE *f = fopen(path.c_str(), "a");
    if (!f) {
        fprintf(stderr, "Warning: cannot open %s for append\n", path.c_str());
        return;
    }
    // Write header if file is empty
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f,
                "candidate_name,N,block_size,vector_width,elems_per_iter,"
                "iter_warmup,iter_timed,latency_median_us,latency_trimmed_mean_us,latency_iqr_us,"
                "bandwidth_median_gbs,device_name,compute_capability,"
                "theoretical_bw_gbs,compiler_flags,correctness_result,"
                "timestamp_utc\n");
    }
    fprintf(f,
            "%s,%zu,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%s,%d.%d,%.2f,%s,%s,%s\n",
            res.name.c_str(), res.N, res.block_size, res.vector_width,
            res.elems_per_iter, res.iter_warmup, res.iter_timed,
            res.latency_median_us, res.latency_trimmed_mean_us, res.latency_iqr_us, res.bandwidth_median_gbs,
            res.device_name.c_str(), res.compute_capability_major,
            res.compute_capability_minor, res.theoretical_bw_gbs,
            res.compiler_flags.c_str(), res.correctness ? "PASS" : "FAIL",
            res.timestamp_utc.c_str());
    fclose(f);
}

static void append_candidates_jsonl(const std::string &path,
                                     const BenchmarkResult &res,
                                     const std::string &parent,
                                     const std::string &status,
                                     const std::string &promotion_decision,
                                     const std::string &notes) {
    FILE *f = fopen(path.c_str(), "a");
    if (!f) {
        fprintf(stderr, "Warning: cannot open %s for append\n", path.c_str());
        return;
    }
    fprintf(f,
            "{\"name\":\"%s\",\"parent\":\"%s\",\"status\":\"%s\","
            "\"block_size\":%d,\"vector_width\":%d,\"elems_per_thread\":%d,"
            "\"latency_median_us\":%.2f,\"bandwidth_median_gbs\":%.2f,"
            "\"correctness\":\"%s\",\"promotion_decision\":\"%s\","
            "\"notes\":\"%s\",\"timestamp_utc\":\"%s\"}\n",
            res.name.c_str(),
            parent.empty() ? "null" : parent.c_str(),
            status.c_str(),
            res.block_size, res.vector_width, res.elems_per_iter,
            res.latency_median_us, res.bandwidth_median_gbs,
            res.correctness ? "PASS" : "FAIL", promotion_decision.c_str(),
            notes.c_str(), res.timestamp_utc.c_str());
    fclose(f);
}

// ============================================================================
// Usage
// ============================================================================
static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s --validate                               Run validation\n"
            "  %s --benchmark [--size N]                   Run benchmark\n"
            "  %s --candidate <name> [--size N]            Run single candidate\n"
            "\n"
            "Candidates: baseline, float2, float4, ilp2x, cache_hint,\n"
            "            block128, block256, block512, block1024\n",
            argv0, argv0, argv0);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }

    std::string mode = argv[1];
    size_t N = 1UL << 28; // default: 2^28 = 268,435,456
    std::string candidate_name;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            N = strtoull(argv[++i], nullptr, 10);
        } else if (mode == "--candidate" && candidate_name.empty()) {
            candidate_name = argv[i];
        }
    }

    // Device info
    print_device_info();
    printf("\nN = %zu (%.2f GB per vector, %.2f GB total I/O)\n\n", N,
           static_cast<double>(N * sizeof(float)) / 1e9,
           static_cast<double>(3 * N * sizeof(float)) / 1e9);

    // Allocate host memory
    std::vector<float> h_A(N), h_B(N), h_C(N), h_ref(N);

    // Fill with random data
    fill_random(h_A, 0xA0A0ULL, -1.0f, 1.0f);
    fill_random(h_B, 0xB0B0ULL, -1.0f, 1.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float),
                           cudaMemcpyHostToDevice));

    const int warmup_iters = 10;
    const int timed_iters = 50;

    if (mode == "--validate") {
        // ================================================================
        // Validation mode: run all kernels and verify correctness
        // ================================================================
        printf("=== Validation Mode ===\n\n");

        struct ValEntry {
            std::string name;
            void (*launch_fn)(const float *, const float *, float *, size_t,
                              int);
            int block;
            int vec_w;
            int elems;
        };

        std::vector<ValEntry> kernels = {
            {"baseline", launch_baseline, 256, 1, 1},
            {"float2", launch_float2, 256, 2, 2},
            {"float4", launch_float4, 256, 4, 4},
            {"ilp2x", launch_ilp2x, 256, 4, 8},
            {"cache_hint", launch_cache_hint, 256, 4, 4},
            {"block128", launch_float4_templated<128>, 128, 4, 4},
            {"block256", launch_float4_templated<256>, 256, 4, 4},
            {"block512", launch_float4_templated<512>, 512, 4, 4},
            {"block1024", launch_float4_templated<1024>, 1024, 4, 4},
        };

        // Test multiple sizes (include N=0 for edge case coverage)
        std::vector<size_t> test_sizes = {0,   1,    127,   1023,
                                           1UL << 20, 1UL << 22, 1UL << 24,
                                           1UL << 26, 1UL << 27, 1UL << 28};

        for (const auto &ke : kernels) {
            bool all_pass = true;
            for (size_t test_n : test_sizes) {
                if (test_n > N)
                    continue; // skip sizes larger than allocated

                // N=0: launch helper returns immediately — verify no crash, no error
                if (test_n == 0) {
                    ke.launch_fn(d_A, d_B, d_C, 0, ke.block);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    CUDA_CHECK(cudaGetLastError());
                    continue;
                }

                // Re-fill with random data for each kernel
                fill_random(h_A, 0xC0C0ULL + test_n, -1.0f, 1.0f);
                fill_random(h_B, 0xD0D0ULL + test_n, -1.0f, 1.0f);
                CUDA_CHECK(cudaMemcpy(d_A, h_A.data(),
                                       test_n * sizeof(float),
                                       cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_B, h_B.data(),
                                       test_n * sizeof(float),
                                       cudaMemcpyHostToDevice));

                // Launch kernel
                ke.launch_fn(d_A, d_B, d_C, test_n, ke.block);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());

                // Validate
                CUDA_CHECK(cudaMemcpy(h_C.data(), d_C,
                                       test_n * sizeof(float),
                                       cudaMemcpyDeviceToHost));
                vector_add_cpu(h_A.data(), h_B.data(), h_ref.data(), test_n);
                ValidationResult vr =
                    validate(h_C.data(), h_ref.data(), test_n, 1e-3);

                if (test_n > 0 && !vr.passed) {
                    printf("  FAIL %-12s N=%9zu  max_err=%.3e  errors=%zu  "
                           "NaN=%d  inf=%d\n",
                           ke.name.c_str(), test_n, vr.max_error,
                           vr.error_count, vr.has_nan, vr.has_inf);
                    all_pass = false;
                }
            }
            if (all_pass)
                printf("  PASS %s (all sizes)\n", ke.name.c_str());
        }

        // Test edge case data patterns
        printf("\n--- Edge case data patterns (float4 kernel, N=%zu) ---\n", N);
        struct {
            const char *desc;
            float a_val, b_val;
        } patterns[] = {
            {"all zeros", 0.0f, 0.0f},
            {"all ones", 1.0f, 1.0f},
            {"alternating +1/-1 (A), +2/-2 (B)", 0.0f, 0.0f},
            {"near FLT_MAX/4", 1e37f, 1e37f},
        };

        for (const auto &pat : patterns) {
            if (strcmp(pat.desc, "all zeros") == 0) {
                for (size_t i = 0; i < N; ++i) {
                    h_A[i] = 0.0f; h_B[i] = 0.0f;
                }
            } else if (strcmp(pat.desc, "all ones") == 0) {
                for (size_t i = 0; i < N; ++i) {
                    h_A[i] = 1.0f; h_B[i] = 1.0f;
                }
            } else if (strcmp(pat.desc, "alternating +1/-1 (A), +2/-2 (B)") == 0) {
                for (size_t i = 0; i < N; ++i) {
                    h_A[i] = (i % 2 == 0) ? 1.0f : -1.0f;
                    h_B[i] = (i % 2 == 0) ? 2.0f : -2.0f;
                }
            } else if (strcmp(pat.desc, "near FLT_MAX/4") == 0) {
                for (size_t i = 0; i < N; ++i) {
                    h_A[i] = 1e37f; h_B[i] = 1e37f;
                }
            }
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float),
                                   cudaMemcpyHostToDevice));
            launch_float4(d_A, d_B, d_C, N, 256);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float),
                                   cudaMemcpyDeviceToHost));
            vector_add_cpu(h_A.data(), h_B.data(), h_ref.data(), N);
            ValidationResult vr = validate(h_C.data(), h_ref.data(), N, 1e-3);
            printf("  %-30s max_err=%.3e  %s\n", pat.desc, vr.max_error,
                   vr.passed ? "PASS" : "FAIL");
        }

        // 5-run stability check on float4 kernel
        printf("\n--- Stability check: 5 consecutive float4 runs, N=%zu ---\n", N);
        {
            bool all_stable = true;
            fill_random(h_A, 0xACC0ULL, -1.0f, 1.0f);
            fill_random(h_B, 0xBEE0ULL, -1.0f, 1.0f);
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * sizeof(float),
                                   cudaMemcpyHostToDevice));
            for (int run = 0; run < 5; ++run) {
                launch_float4(d_A, d_B, d_C, N, 256);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * sizeof(float),
                                       cudaMemcpyDeviceToHost));
                vector_add_cpu(h_A.data(), h_B.data(), h_ref.data(), N);
                ValidationResult vr = validate(h_C.data(), h_ref.data(), N, 1e-3);
                if (!vr.passed) {
                    printf("  FAIL run %d: max_err=%.3e  errors=%zu\n",
                           run + 1, vr.max_error, vr.error_count);
                    all_stable = false;
                }
            }
            printf("  %s\n", all_stable ? "STABILITY PASS (5/5)" : "STABILITY FAIL");
        }

    } else if (mode == "--benchmark") {
        // ================================================================
        // Benchmark mode: sweep all candidates across N sizes
        // ================================================================
        printf("=== Benchmark Mode ===\n");
        printf("Warm-up: %d iterations, Timed: %d iterations\n\n",
               warmup_iters, timed_iters);

        std::vector<size_t> bench_sizes;
        for (size_t s = 1UL << 20; s <= N; s <<= 1) {
            bench_sizes.push_back(s);
        }

        std::vector<BenchmarkConfig> configs = {
            {"baseline", 256, 1, 1, launch_baseline},
            {"float2", 256, 2, 2, launch_float2},
            {"float4", 256, 4, 4, launch_float4},
            {"ilp2x", 256, 4, 8, launch_ilp2x},
            {"cache_hint", 256, 4, 4, launch_cache_hint},
        };

        std::vector<BenchmarkResult> all_results;

        for (size_t bench_n : bench_sizes) {
            printf("--- N = %zu ---\n", bench_n);
            // Re-fill data for each N
            fill_random(h_A, 0xAA00ULL + bench_n, -1.0f, 1.0f);
            fill_random(h_B, 0xBB00ULL + bench_n, -1.0f, 1.0f);
            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(),
                                   bench_n * sizeof(float),
                                   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(),
                                   bench_n * sizeof(float),
                                   cudaMemcpyHostToDevice));

            // Block size sweep for float4
            int block_sizes[] = {128, 256, 512, 1024};
            for (int bs : block_sizes) {
                char name_buf[32];
                snprintf(name_buf, sizeof(name_buf), "float4_b%d", bs);
                void (*launch_fn)(const float *, const float *, float *, size_t,
                                  int) = nullptr;
                switch (bs) {
                case 128:
                    launch_fn = launch_float4_templated<128>;
                    break;
                case 256:
                    launch_fn = launch_float4_templated<256>;
                    break;
                case 512:
                    launch_fn = launch_float4_templated<512>;
                    break;
                case 1024:
                    launch_fn = launch_float4_templated<1024>;
                    break;
                }
                BenchmarkConfig cfg = {name_buf, bs, 4, 4, launch_fn};
                BenchmarkResult res =
                    run_benchmark(cfg, d_A, d_B, d_C, bench_n, warmup_iters,
                                  timed_iters);
                all_results.push_back(res);
                append_benchmark_csv("benchmark.csv", res);
            }

            for (const auto &cfg : configs) {
                BenchmarkResult res = run_benchmark(cfg, d_A, d_B, d_C,
                                                     bench_n, warmup_iters,
                                                     timed_iters);
                all_results.push_back(res);
                append_benchmark_csv("benchmark.csv", res);
            }
        }

        // Write candidates.jsonl with promotion decisions
        printf("\n--- Writing candidates.jsonl ---\n");
        // Find best float4 variant by bandwidth
        double best_float4_bw = 0.0;
        size_t best_n = 1UL << 26; // N=2^26 for primary comparison
        for (const auto &res : all_results) {
            if (res.N == best_n && res.name.find("float4") == 0 && res.correctness) {
                if (res.bandwidth_median_gbs > best_float4_bw) {
                    best_float4_bw = res.bandwidth_median_gbs;
                }
            }
        }

        for (const auto &res : all_results) {
            std::string parent;
            if (res.name.find("float4_b") == 0) {
                parent = "float4";
            } else if (res.name == "float2" || res.name == "float4" ||
                       res.name == "ilp2x" || res.name == "cache_hint") {
                parent = "baseline";
            }
            std::string status = "evaluated";
            std::string decision;
            if (!res.correctness) {
                decision = "rejected";
            } else if (res.name == "float4") {
                decision = "promoted";
            } else if (res.name.find("float4_b") == 0) {
                decision = "evaluated_variant";
            } else {
                decision = "evaluated_not_promoted";
            }
            append_candidates_jsonl("candidates.jsonl", res, parent, status,
                                     decision, "");
        }

        printf("\nResults written to benchmark.csv and candidates.jsonl\n");

    } else if (mode == "--candidate") {
        // ================================================================
        // Single candidate mode
        // ================================================================
        if (candidate_name.empty()) {
            fprintf(stderr,
                    "Error: --candidate requires a candidate name\n");
            usage(argv[0]);
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            return 2;
        }

        printf("=== Single Candidate: %s ===\n\n", candidate_name.c_str());

        BenchmarkConfig cfg;
        bool found = false;

        if (candidate_name == "baseline") {
            cfg = {"baseline", 256, 1, 1, launch_baseline};
            found = true;
        } else if (candidate_name == "float2") {
            cfg = {"float2", 256, 2, 2, launch_float2};
            found = true;
        } else if (candidate_name == "float4") {
            cfg = {"float4", 256, 4, 4, launch_float4};
            found = true;
        } else if (candidate_name == "ilp2x") {
            cfg = {"ilp2x", 256, 4, 8, launch_ilp2x};
            found = true;
        } else if (candidate_name == "cache_hint") {
            cfg = {"cache_hint", 256, 4, 4, launch_cache_hint};
            found = true;
        } else if (candidate_name == "block128") {
            cfg = {"float4_b128", 128, 4, 4, launch_float4_templated<128>};
            found = true;
        } else if (candidate_name == "block256") {
            cfg = {"float4_b256", 256, 4, 4, launch_float4_templated<256>};
            found = true;
        } else if (candidate_name == "block512") {
            cfg = {"float4_b512", 512, 4, 4, launch_float4_templated<512>};
            found = true;
        } else if (candidate_name == "block1024") {
            cfg = {"float4_b1024", 1024, 4, 4,
                   launch_float4_templated<1024>};
            found = true;
        }

        if (!found) {
            fprintf(stderr, "Error: unknown candidate '%s'\n",
                    candidate_name.c_str());
            usage(argv[0]);
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
            return 2;
        }

        BenchmarkResult res =
            run_benchmark(cfg, d_A, d_B, d_C, N, warmup_iters, timed_iters);
        append_benchmark_csv("benchmark.csv", res);
        append_candidates_jsonl("candidates.jsonl", res, "baseline", "evaluated",
                                 "pending_review", "");

    } else {
        fprintf(stderr, "Error: unknown mode '%s'\n", mode.c_str());
        usage(argv[0]);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        return 2;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("\nDone.\n");
    return 0;
}
