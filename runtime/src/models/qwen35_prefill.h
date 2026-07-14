#pragma once
#include <cstdint>
#include <vector>

namespace sparkinfer {

class Qwen35Model;

// Runtime orchestration for batched prompt prefill (see kernels/prefill.h).
// Kept in a separate translation unit so Qwen35Model::Impl stays private.
namespace qwen35_prefill {

bool enabled();          // SPARKINFER_PREFILL_BATCHED=1
int max_context();       // SPARKINFER_PREFILL_BATCHED_MAXCTX (default 65536)

// Ingest prompt tokens into KV + hybrid state without LM head on interior steps.
// Returns false → caller must use forward_token loop.
bool run(Qwen35Model& model, const std::vector<int>& tokens);

} // namespace qwen35_prefill
} // namespace sparkinfer
