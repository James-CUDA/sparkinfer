// Chunked batched-prefill orchestration (included from qwen35.cpp).

namespace {

static int prefill_tile_rows() {
    static int rows = -1;
    if (rows < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_TILE_ROWS");
        rows = e ? atoi(e) : 8;
        if (rows < 1) rows = 1;
        if (rows > 16) rows = 16;
    }
    return rows;
}

static bool prefill_debug() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED_DEBUG");
        on = (e && e[0] == '1') ? 1 : 0;
    }
    return on != 0;
}

static bool pf_q4k_type(int t) { return t == 12; }

static void pf_fail(const char* why) {
    if (prefill_debug()) fprintf(stderr, "[batched_prefill] fallback: %s\n", why);
}

} // namespace

bool Qwen35Model::prefill_batched_impl(const std::vector<int>& tokens) {
    Impl& s = *p_;
    if (tokens.empty()) { pf_fail("empty"); return false; }
    // Batched ingest fills KV outside the decode CUDA graph; drop any capture so the
    // next forward_token() re-captures from the batched KV state.
    if (s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec);
        cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
    }
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    if (!s.gguf || !s.use_llama || !s.use_pq || !c.dense_ffn || c.top_k != 1) {
        pf_fail("guards"); return false;
    }
    if (H != 2048 && H != 4096) { pf_fail("hidden"); return false; }
    if (!s.kv->block_table(s.seq_id)) {
        const int need_seq = c.max_seq > (int)tokens.size() ? c.max_seq : (int)tokens.size();
        if (!s.kv->allocate(s.seq_id, need_seq)) {
            pf_fail("kv"); return false;
        }
    }
    if (c.n_shared > 0) { pf_fail("shared"); return false; }

    const int TILE = prefill_tile_rows();
    if (TILE <= 1) { pf_fail("tile_rows"); return false; }

    auto ensure_bufs = [&](int M) {
        if (M <= s.pf_tile_cap) return;
        auto freep = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        freep(s.pf_toks); freep(s.pf_pos);
        freep(s.pf_x); freep(s.pf_xn); freep(s.pf_ao); freep(s.pf_h); freep(s.pf_hn);
        freep(s.pf_routed); freep(s.pf_q); freep(s.pf_k); freep(s.pf_v); freep(s.pf_attn);
        freep(s.pf_qraw); freep(s.pf_qgate);
        freep(s.pf_lin_qkv); freep(s.pf_lin_z); freep(s.pf_lin_alpha); freep(s.pf_lin_beta);
        freep(s.pf_lin_q); freep(s.pf_lin_k); freep(s.pf_lin_v); freep(s.pf_lin_gdn); freep(s.pf_lin_norm);
        freep(s.pf_aq81); freep(s.pf_aq81_q); freep(s.pf_ffn_scratch); freep(s.pf_mf_h);
        s.pf_tile_cap = M;
        s.pf_toks = s.alloc<int>(M);
        s.pf_pos = s.alloc<int>(M);
        s.pf_x = s.alloc<bf16>((size_t)M * H);
        s.pf_xn = s.alloc<bf16>((size_t)M * H);
        s.pf_ao = s.alloc<bf16>((size_t)M * H);
        s.pf_h = s.alloc<bf16>((size_t)M * H);
        s.pf_hn = s.alloc<bf16>((size_t)M * H);
        s.pf_routed = s.alloc<bf16>((size_t)M * H);
        s.pf_q = s.alloc<bf16>((size_t)M * s.qdim);
        s.pf_k = s.alloc<bf16>((size_t)M * s.kvdim);
        s.pf_v = s.alloc<bf16>((size_t)M * s.kvdim);
        s.pf_attn = s.alloc<bf16>((size_t)M * s.qdim);
        if (c.hybrid) {
            s.pf_qraw = s.alloc<bf16>((size_t)M * s.qdim * 2);
            s.pf_qgate = s.alloc<bf16>((size_t)M * s.qdim);
        }
        s.pf_aq81 = s.alloc<char>((size_t)M * kernels::llama_q8_1_bytes(H));
        s.pf_aq81_q = s.alloc<char>((size_t)M * kernels::llama_q8_1_bytes(s.qdim));
        s.pf_ffn_scratch = s.alloc<char>((size_t)M * c.top_k * kernels::llama_q8_1_bytes(c.moe_ffn));
        s.pf_mf_h = s.alloc<float>((size_t)M * c.moe_ffn);
        if (c.hybrid) {
            s.pf_lin_qkv = s.alloc<bf16>((size_t)M * s.linear_qkvdim);
            s.pf_lin_z = s.alloc<bf16>((size_t)M * s.linear_vdim);
            s.pf_lin_alpha = s.alloc<bf16>((size_t)M * c.linear_v_heads);
            s.pf_lin_beta = s.alloc<bf16>((size_t)M * c.linear_v_heads);
            s.pf_lin_q = s.alloc<bf16>((size_t)M * s.linear_qdim);
            s.pf_lin_k = s.alloc<bf16>((size_t)M * s.linear_qdim);
            s.pf_lin_v = s.alloc<bf16>((size_t)M * s.linear_vdim);
            s.pf_lin_gdn = s.alloc<bf16>((size_t)M * s.linear_vdim);
            s.pf_lin_norm = s.alloc<bf16>((size_t)M * s.linear_vdim);
        }
    };
    ensure_bufs(TILE);
    cudaStream_t st = s.stream;
    int* btable = s.kv->block_table(s.seq_id);
    const int n = (int)tokens.size();

    if (c.hybrid) {
        cu(cudaMemsetAsync(s.lin_state, 0,
            (size_t)c.n_layers * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim * sizeof(float), st),
           "batched linear state reset");
        cu(cudaMemsetAsync(s.lin_conv_state, 0,
            (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim * sizeof(bf16), st),
           "batched linear conv reset");
    }

    const bool kv8 = s.kv->int8_kv();
    const int kv_elem = kv8 ? 1 : 2;
    const bool partial_rope = (c.rope_dim > 0 && c.rope_dim < c.head_dim);
    const bool sparse_avail = s.sparse_budget > 0 && kv8 &&
                              c.head_dim == 256 && c.n_q_heads == c.n_kv_heads * 4;

    for (int base = 0; base < n; base += TILE) {
        const int M = (n - base < TILE) ? (n - base) : TILE;
        std::vector<int> hpos((size_t)M);
        for (int t = 0; t < M; ++t) hpos[(size_t)t] = base + t;
        cu(cudaMemcpyAsync(s.pf_toks, tokens.data() + base, (size_t)M * sizeof(int),
                           cudaMemcpyHostToDevice, st), "pf toks");
        cu(cudaMemcpyAsync(s.pf_pos, hpos.data(), (size_t)M * sizeof(int),
                           cudaMemcpyHostToDevice, st), "pf pos");

        kernels::launch_embedding(s.pf_toks, s.w.embed_tokens, s.pf_x, M, H, st);
        kernels::launch_rmsnorm(s.pf_x, s.w.layers[0].input_norm, s.pf_xn, M, H, c.rms_eps, st);

        for (int L = 0; L < c.n_layers; ++L) {
            const Qwen35LayerWeights& w = s.w.layers[L];

            if (w.linear_attn) {
                kernels::launch_quantize_q8_1_blocks_tile(s.pf_xn, s.pf_aq81, M, H, st);
                if (pf_q4k_type(w.wqkv_type) && pf_q4k_type(w.wqkv_gate_type) &&
                    pf_q4k_type(w.ssm_alpha_type) && pf_q4k_type(w.ssm_beta_type)) {
                    kernels::launch_mmvq_q4k_tile(s.pf_aq81, w.wqkv, s.pf_lin_qkv, M, s.linear_qkvdim, H, st);
                    kernels::launch_mmvq_q4k_tile(s.pf_aq81, w.wqkv_gate, s.pf_lin_z, M, s.linear_vdim, H, st);
                    kernels::launch_mmvq_q4k_tile(s.pf_aq81, w.ssm_alpha, s.pf_lin_alpha, M, c.linear_v_heads, H, st);
                    kernels::launch_mmvq_q4k_tile(s.pf_aq81, w.ssm_beta, s.pf_lin_beta, M, c.linear_v_heads, H, st);
                } else {
                    pf_fail("gdn qtypes"); return false;
                }

                for (int t = 0; t < M; ++t) {
                    const int pos = base + t;
                    const int seqlen = pos + 1;
                    bf16* lin_qkv = s.pf_lin_qkv + (size_t)t * s.linear_qkvdim;
                    bf16* lin_z = s.pf_lin_z + (size_t)t * s.linear_vdim;
                    bf16* lin_alpha = s.pf_lin_alpha + (size_t)t * c.linear_v_heads;
                    bf16* lin_beta = s.pf_lin_beta + (size_t)t * c.linear_v_heads;
                    bf16* lin_q = s.pf_lin_q + (size_t)t * s.linear_qdim;
                    bf16* lin_k = s.pf_lin_k + (size_t)t * s.linear_qdim;
                    bf16* lin_v = s.pf_lin_v + (size_t)t * s.linear_vdim;
                    bf16* lin_gdn = s.pf_lin_gdn + (size_t)t * s.linear_vdim;
                    bf16* lin_norm = s.pf_lin_norm + (size_t)t * s.linear_vdim;
                    bf16* ao = s.pf_ao + (size_t)t * H;

                    bf16* conv_state = s.lin_conv_state +
                        (size_t)L * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
                    if (c.linear_head_dim == 128 && c.linear_q_heads == 16 && c.linear_v_heads == 32) {
                        kernels::launch_qwen36_conv_split_l2norm_fused(lin_qkv, w.ssm_conv, conv_state,
                            lin_q, lin_k, lin_v, c.linear_q_heads, c.linear_v_heads,
                            c.linear_head_dim, c.linear_conv_kernel, c.rms_eps, st);
                    } else {
                        kernels::launch_qwen36_conv_split_l2(lin_qkv, w.ssm_conv, conv_state,
                            lin_q, lin_k, lin_v, c.linear_q_heads, c.linear_v_heads,
                            c.linear_head_dim, c.linear_conv_kernel, c.rms_eps, st);
                    }
                    float* layer_state = s.lin_state +
                        (size_t)L * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim;
                    kernels::launch_qwen36_gdn_ar(lin_q, lin_k, lin_v, lin_alpha, lin_beta,
                        w.ssm_dt, w.ssm_a, layer_state, lin_gdn,
                        c.linear_q_heads, c.linear_v_heads, c.linear_head_dim, st);
                    kernels::launch_qwen36_gated_norm(lin_gdn, lin_z, w.ssm_norm, lin_norm,
                        c.linear_v_heads, c.linear_head_dim, c.rms_eps, st);
                    if (!pf_q4k_type(w.ssm_out_type)) return false;
                    kernels::launch_quantize_q8_1_blocks(lin_norm, s.aq81, s.linear_vdim, st);
                    kernels::launch_mmvq_q4k(s.aq81, w.ssm_out, ao, H, s.linear_vdim, st);
                }
            } else {
                if (!(pf_q4k_type(w.wq_type) && pf_q4k_type(w.wk_type) && pf_q4k_type(w.wv_type) &&
                      pf_q4k_type(w.wo_type)))
                    return false;
                const int nq = w.q_has_gate ? s.qdim * 2 : s.qdim;
                bf16* yq = w.q_has_gate ? s.pf_qraw : s.pf_q;
                kernels::launch_quantize_q8_1_blocks_tile(s.pf_xn, s.pf_aq81, M, H, st);
                kernels::launch_attn_qkv_mmvq_q4k_tile(s.pf_aq81, w.wq, w.wk, w.wv,
                    yq, s.pf_k, s.pf_v, M, nq, s.kvdim, s.kvdim, H, st);

                void* kpool = (char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
                void* vpool = (char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
                void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
                void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;

                for (int t = 0; t < M; ++t) {
                    const int pos = base + t;
                    const int seqlen = pos + 1;
                    s.h_scalars[1] = pos;
                    s.h_scalars[2] = pos;
                    s.h_scalars[3] = seqlen;
                    cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, 4 * sizeof(int),
                                       cudaMemcpyHostToDevice, st), "pf scalars");

                    bf16* q = s.pf_q + (size_t)t * s.qdim;
                    bf16* qraw = s.pf_qraw + (size_t)t * s.qdim * 2;
                    bf16* qgate = s.pf_qgate + (size_t)t * s.qdim;
                    bf16* k = s.pf_k + (size_t)t * s.kvdim;
                    bf16* v = s.pf_v + (size_t)t * s.kvdim;
                    bf16* attn = s.pf_attn + (size_t)t * s.qdim;

                    if (w.q_has_gate)
                        kernels::launch_qwen36_split_q_gate(qraw, q, qgate, c.n_q_heads, c.head_dim, st);

                    if (partial_rope && s.use_qkfuse) {
                        if (kv8) {
                            if (w.q_has_gate) {
                                kernels::launch_qknorm_rope_kv_partial_int8_gated(qraw, q, qgate, k, v,
                                    w.q_norm, w.k_norm, kpool, vpool, kscale, vscale, btable, s.d_pos, 1,
                                    c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                                    s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                            } else {
                                kernels::launch_qknorm_rope_kv_partial_int8(q, k, v, w.q_norm, w.k_norm,
                                    kpool, vpool, kscale, vscale, btable, s.d_pos, 1,
                                    c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                                    s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                            }
                        } else {
                            kernels::launch_qknorm_rope_kv_partial(q, k, v, w.q_norm, w.k_norm,
                                (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                                c.rope_theta, c.rms_eps, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                        }
                    } else {
                        return false;
                    }

                    const bool sparse_on = sparse_avail && seqlen >= s.sparse_min_ctx;
                    if (sparse_on && kv8) {
                        kernels::launch_fa_kv_window_select(s.d_seqlen, s.sparse_sel, c.n_kv_heads,
                            s.kv->block_size(), s.sparse_budget, s.sparse_window, st);
                        kernels::launch_flash_decode_split_sparse(q, kpool, vpool, btable, s.d_seqlen,
                            s.sparse_sel, s.fa_m, s.fa_l, s.fa_acc, c.n_q_heads, c.n_kv_heads, c.head_dim,
                            s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits, s.sparse_budget,
                            1.f / sqrtf((float)c.head_dim), kscale, vscale, st);
                        kernels::launch_fa_combine_hd256(s.fa_m, s.fa_l, s.fa_acc, attn, c.n_q_heads,
                            s.n_splits, nullptr, st);
                    } else {
                        kernels::launch_flash_decode_split(q, kpool, vpool, btable, s.d_seqlen, attn,
                            s.fa_m, s.fa_l, s.fa_acc, 1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                            s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                            1.f / sqrtf((float)c.head_dim), st, nullptr, seqlen,
                            kscale, vscale, kv8 ? 1 : 0, w.q_has_gate ? qgate : nullptr);
                    }
                    if (w.q_has_gate)
                        kernels::launch_qwen36_mul_sigmoid(attn, qgate, s.qdim, st);
                }
                kernels::launch_quantize_q8_1_blocks_tile(s.pf_attn, s.pf_aq81_q, M, s.qdim, st);
                kernels::launch_mmvq_q4k_tile(s.pf_aq81_q, w.wo, s.pf_ao, M, H, s.qdim, st);
            }

            kernels::launch_add_rmsnorm2(s.pf_x, s.pf_ao, w.post_attn_norm, s.pf_h, s.pf_hn, M, H, c.rms_eps, st);
            kernels::launch_quantize_q8_1_blocks_tile(s.pf_hn, s.pf_aq81, M, H, st);
            kernels::launch_moe_expert_ffn_q4k(s.pf_hn, w.gate_q, w.up_q, w.down_q,
                w.gate_qtype, w.up_qtype, w.down_qtype,
                s.mf_ids, s.mf_weights, s.pf_routed, s.pf_mf_h, s.pf_ffn_scratch,
                M, c.top_k, H, c.moe_ffn, s.pf_aq81, st);

            const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
            kernels::launch_add_rmsnorm2(s.pf_h, s.pf_routed, nextnorm, s.pf_x, s.pf_xn, M, H, c.rms_eps, st);
        }
    }

    cu(cudaStreamSynchronize(st), "batched prefill sync");
    if (prefill_debug())
        fprintf(stderr, "[batched_prefill] ingested n=%d tile_rows=%d\n", n, TILE);
    return true;
}
