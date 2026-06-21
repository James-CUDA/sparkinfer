# blackwell-infer / runtime

Edge AI inference runtime for **NVIDIA RTX Spark** and RTX 5090-class GPUs.

Part of [gittensor-ai-lab](https://github.com/gittensor-ai-lab) — SN74 decentralized kernel and runtime optimization network.

---

## Primary Target: NVIDIA RTX Spark

RTX Spark is NVIDIA's Blackwell-based AI superchip: 20-core ARM CPU + Blackwell GPU + **128 GB unified LPDDR5X memory** in a single package. It runs 120B-parameter models locally and delivers ~1 PFLOP of AI compute — making it the first consumer-grade platform where datacenter-scale MoE inference is native.

- [NVIDIA RTX Spark × MediaTek — Architecture Overview](https://www.mediatek.com/products/personal-computing/nvidia-rtx-spark)
- [RTX Spark laptops launching Fall 2026 — Tom's Guide](https://www.tomsguide.com/computing/gaming-laptops/all-8-laptops-launching-with-nvidia-rtx-spark-this-fall-and-what-they-can-do)
- [NVIDIA RTX AI Garage: Local Agents on RTX Spark](https://blogs.nvidia.com/blog/rtx-ai-garage-computex-spark-local-agents/)

**Why RTX Spark changes the runtime problem:**

| Before (cloud/datacenter era) | RTX Spark era |
|---|---|
| Optimize for TP=8, batch=1024 | Optimize for single-node, batch=1–64 |
| Memory bandwidth = HBM (3+ TB/s) | Memory bandwidth = LPDDR5X (~273 GB/s) |
| Expert eviction = rare | Expert scheduling = critical |
| KV cache = secondary concern | KV cache = primary memory pressure |
| vLLM / SGLang sufficient | New runtime layer required |

---

## Hardware Targets

| Device | Memory | Bandwidth | Arch | Status |
|---|---|---|---|---|
| **RTX Spark** | 128 GB unified | ~273 GB/s | sm_100 | **Primary** |
| RTX 5090 | 32 GB GDDR7 | 1.79 TB/s | sm_100 | Secondary |
| Jetson Thor | TBD | TBD | sm_100 | Planned |

---

## What This Runtime Does

Standard serving stacks (vLLM, SGLang) are designed for distributed datacenter inference — tensor parallelism across multiple GPUs, large batch throughput, HBM-class bandwidth. On RTX Spark, those assumptions break:

- No TP — everything runs on one chip
- Unified memory means CPU and GPU share the same pool — expert weights don't need PCIe transfer
- LPDDR5X bandwidth (~273 GB/s) is 6–7× lower than GDDR7 — memory layout and scheduling matter more than raw kernel FLOPS
- Interactive latency targets (< 50 ms TTFT) require a latency-first scheduler, not a throughput-first one

This runtime is built specifically for that environment.

---

## Components

```
include/blackwell/
├── runtime.h       — Runtime lifecycle, device query
├── scheduler.h     — Continuous batching, chunked prefill, priority preemption
└── kv_cache.h      — Paged KV block allocator (FP8 compression option)

src/
├── scheduler/      — Latency-first scheduler implementation
├── memory/         — Unified memory-aware allocator
└── cuda_graph/     — CUDA graph capture and replay engine
```

---

## Target Models

| Model | Total / Active | Q4_K_M Size | Notes |
|---|---|---|---|
| Qwen3.5-35B-A3B | 35B / 3B | ~20 GB | 256 experts, 2 KV heads |
| Gemma 4 26B-A4B | 26B / 4B | ~14.6 GB | Interleaved local/global attn, head_dim=512 |

Both fit in RTX Spark's 128 GB unified memory with significant headroom for KV cache.

---

## Related Repos

| Repo | Purpose |
|---|---|
| [kernels](https://github.com/gittensor-ai-lab/kernels) | CUDA/Triton kernels: flash decode, MoE grouped GEMM, fused ops |
| [moe](https://github.com/gittensor-ai-lab/moe) | MoE engine: router, expert allocator, grouped GEMM dispatcher |
| [bench](https://github.com/gittensor-ai-lab/bench) | Reproducible benchmarks for RTX Spark, RTX 5090, Jetson Thor |
| [agent](https://github.com/gittensor-ai-lab/agent) | Kernel design agents: NCU report parsing, auto-tuning loops |

---

## Build

```bash
cmake -B build \
  -DCMAKE_CUDA_ARCHITECTURES="100" \   # sm_100 = Blackwell (RTX Spark / RTX 5090)
  -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

---

## SN74 / Gittensor

This repository is part of **SN74** on [Gittensor](https://github.com/gittensor-ai-lab) — a decentralized network where human engineers and AI agents co-design and continuously improve kernels, memory systems, and MoE routing for edge AI hardware. All optimizations are reproducible: source-required, benchmark-verified, no hidden training or closed binaries.
