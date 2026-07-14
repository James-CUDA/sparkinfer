#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Batched prefill kernel surface (Qwythos / Qwen3.5 dense-hybrid).
// Tile GEMMs amortize Q4_K weight reads over ROWS token rows per launch.
// GDN + full-attention layers fill KV / recurrent state for decode reuse.

struct BatchedPrefillConfig {
    int hidden = 0;
    int n_layers = 0;
    int n_q_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int vocab = 0;
    int full_attn_interval = 0;  // hybrid: every Nth layer is full attention
    bool hybrid = false;
    float rms_eps = 1e-6f;
    float rope_theta = 1e6f;
};

// Returns true when the batched path ingested all prompt tokens. False = caller
// should fall back to the per-token forward_token loop.
bool launch_batched_prefill(
    const BatchedPrefillConfig& cfg,
    const int* __restrict__ tokens,   // host [n_tokens]
    int n_tokens,
    void* kv_mgr,                     // KVCacheManager* (opaque to kernels lib)
    uint64_t seq_id,
    cudaStream_t stream);

}} // namespace sparkinfer::kernels
