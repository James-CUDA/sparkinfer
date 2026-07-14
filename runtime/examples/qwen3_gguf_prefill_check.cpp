// A/B correctness harness: batched prefill vs teacher-forced forward_token loop.
// Usage: qwen3_gguf_prefill_check <model.gguf> <n_tokens>
//
// With SPARKINFER_PREFILL_BATCHED=1, compares first decode step + logits after
// batched ingest vs the token loop. Exits 0 on match, 1 on divergence.

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int argmax(const float* logits, int vocab) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < vocab; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

static float kl_div(const float* p, const float* q, int n) {
    double kl = 0.0;
    for (int i = 0; i < n; i++) {
        const double pi = std::exp((double)p[i]);
        const double qi = std::exp((double)q[i]);
        if (pi > 1e-30) kl += pi * (std::log(pi) - std::log(std::max(qi, 1e-30)));
    }
    return (float)kl;
}

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

    std::vector<int> prompt((size_t)n_tokens);
    for (int i = 0; i < n_tokens; i++) prompt[(size_t)i] = 100 + (i % 17);

    const int decode_probe = 100;
    std::vector<float> logits_ref((size_t)cfg.vocab), logits_bat((size_t)cfg.vocab);

    // Reference: teacher-forced token loop through the prompt.
    for (int i = 0; i < n_tokens; i++)
        (void)model.forward_token(prompt[(size_t)i], i);
    const int dec_ref = model.forward_token(decode_probe, n_tokens);
    model.copy_logits(logits_ref.data());

    if (!getenv("SPARKINFER_PREFILL_BATCHED") || getenv("SPARKINFER_PREFILL_BATCHED")[0] != '1') {
        printf("SKIP batched path (set SPARKINFER_PREFILL_BATCHED=1)\n");
        printf("loop decode@%d=%d\n", n_tokens, dec_ref);
        return 0;
    }

    kv.free(0);
    if (!kv.allocate(0, cfg.max_seq)) {
        printf("[FAIL] KV re-allocate\n");
        return 1;
    }

    if (!model.prefill_batched(prompt)) {
        printf("SKIP batched prefill unavailable (guards failed or unsupported config)\n");
        return 0;
    }

    const int dec_bat = model.forward_token(decode_probe, n_tokens);
    model.copy_logits(logits_bat.data());

    const int am_ref = argmax(logits_ref.data(), cfg.vocab);
    const int am_bat = argmax(logits_bat.data(), cfg.vocab);
    const float kl = kl_div(logits_ref.data(), logits_bat.data(), cfg.vocab);

    printf("decode@%d ref=%d bat=%d %s\n", n_tokens, dec_ref, dec_bat,
           dec_ref == dec_bat ? "OK" : "MISMATCH");
    printf("argmax ref=%d bat=%d %s\n", am_ref, am_bat, am_ref == am_bat ? "OK" : "MISMATCH");
    printf("KL %.6f\n", kl);

    const bool ok = (dec_ref == dec_bat) && (am_ref == am_bat) && (kl < 1e-3f);
    printf("%s batched prefill parity (n=%d)\n", ok ? "PASS" : "FAIL", n_tokens);
    return ok ? 0 : 1;
}
