// Batched prompt prefill — weight-amortized tile GEMMs for Qwen3.5 dense-hybrid.
// Tile Q4_K kernels live in gemv.cu (launch_mmvq_q4k_tile, launch_attn_qkv_mmvq_q4k_tile).
// Runtime orchestration: runtime/src/models/qwen35_batched_prefill.inl

#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/gemm.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace sparkinfer {
namespace kernels {

namespace {

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
    cudaStream_t /*stream*/)
{
    if (n_tokens <= 0 || cfg.hidden <= 0 || cfg.n_layers <= 0) return false;
    if (prefill_debug())
        fprintf(stderr, "[batched_prefill] launch_batched_prefill is a stub; "
                "runtime uses qwen35_batched_prefill.inl (n_tokens=%d)\n", n_tokens);
    return false;
}

} // namespace kernels
} // namespace sparkinfer
