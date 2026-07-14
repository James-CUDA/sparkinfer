#include "qwen35_prefill.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/kernels/prefill.h"

#include <cstdlib>

namespace sparkinfer {
namespace qwen35_prefill {

bool enabled() {
    static int on = -1;
    if (on < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED");
        on = (e && e[0] == '1') ? 1 : 0;
    }
    return on != 0;
}

int max_context() {
    static int lim = -1;
    if (lim < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED_MAXCTX");
        lim = e ? atoi(e) : 65536;
        if (lim <= 0) lim = 65536;
    }
    return lim;
}

bool run(Qwen35Model& model, const std::vector<int>& tokens) {
    if (!enabled()) return false;
    if (tokens.empty()) return false;
    if ((int)tokens.size() > max_context()) return false;
    return model.prefill_batched(tokens);
}

} // namespace qwen35_prefill
} // namespace sparkinfer
