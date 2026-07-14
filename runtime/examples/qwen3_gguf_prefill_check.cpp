// A/B correctness harness: batched prefill vs forward_token loop.
// Usage: qwen3_gguf_prefill_check <model.gguf> <n_tokens>
//
// With SPARKINFER_PREFILL_BATCHED=1, compares greedy argmax chain after batched
// ingest vs the token loop. Exits 0 on match, 1 on divergence (once implemented).

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model.gguf> <n_tokens>\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n");
        return 0;
    }

    const std::string path = argv[1];
    const int n_tokens = atoi(argv[2]);
    if (n_tokens <= 0) {
        printf("[FAIL] n_tokens must be > 0\n");
        return 1;
    }

    sparkinfer::GGUF g;
    if (!g.open(path)) {
        printf("[FAIL] open gguf\n");
        return 1;
    }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq = std::max(cfg.max_seq, n_tokens + 16);

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = n_tokens >= 4096;
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_gguf(path)) {
        printf("[FAIL] load\n");
        return 1;
    }
    if (!kv.allocate(0, cfg.max_seq)) {
        printf("[FAIL] KV allocate (max_seq=%d)\n", cfg.max_seq);
        return 1;
    }

    std::vector<int> prompt((size_t)n_tokens, 100);

    // Reference: token loop
    std::vector<int> loop_next;
    loop_next.reserve((size_t)n_tokens);
    int tok = 100;
    for (int i = 0; i < n_tokens; i++) {
        tok = model.forward_token(tok, i);
        loop_next.push_back(tok);
    }

    if (!getenv("SPARKINFER_PREFILL_BATCHED") || getenv("SPARKINFER_PREFILL_BATCHED")[0] != '1') {
        printf("SKIP batched path (set SPARKINFER_PREFILL_BATCHED=1)\n");
        printf("loop reference: first=%d last=%d\n", loop_next.front(), loop_next.back());
        return 0;
    }

    if (!model.prefill_batched(prompt)) {
        printf("SKIP batched prefill unavailable (guards failed or unsupported config)\n");
        printf("loop reference: first=%d last=%d\n", loop_next.front(), loop_next.back());
        return 0;
    }

    printf("[FAIL] batched path ran but post-check not implemented\n");
    return 1;
}
