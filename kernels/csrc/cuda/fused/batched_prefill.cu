// Batched prompt prefill — weight-amortized tile GEMMs for Qwen3.5 dense-hybrid.
// Skeleton: API + launch stub. Implementation fills KV via WMMA/cp.async GEMM
// tiles (Q/K/V/O, SwiGLU FFN), GDN scan, and paged int8 causal attention.

#include "sparkinfer/kernels/prefill.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace sparkinfer {
namespace kernels {

namespace {

__global__ void batched_prefill_ready_stub(int n_tokens) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && n_tokens > 0) {}
}

static bool prefill_debug() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED_DEBUG");
        on = (e && e[0] == '1') ? 1 : 0;
    }
    return on != 0;
}

} // namespace

bool launch_batched_prefill(
    const BatchedPrefillConfig& cfg,
    const int* /*tokens*/,
    int n_tokens,
    void* /*kv_mgr*/,
    uint64_t /*seq_id*/,
    cudaStream_t stream)
{
    if (n_tokens <= 0 || cfg.hidden <= 0 || cfg.n_layers <= 0) return false;

    // Placeholder launch keeps the link path buildable; real work lands in follow-up patches.
    batched_prefill_ready_stub<<<1, 1, 0, stream>>>(n_tokens);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[batched_prefill] stub launch failed: %s\n", cudaGetErrorString(e));
        return false;
    }

    if (prefill_debug())
        fprintf(stderr, "[batched_prefill] stub n_tokens=%d (not implemented — use token loop)\n",
                n_tokens);
    return false;
}

} // namespace kernels
} // namespace sparkinfer
