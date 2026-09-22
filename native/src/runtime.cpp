// runtime.cpp — implements the public C ABI declared in
// native/include/tinymind.h.
//
// STATUS: real as of this phase. tm_load_model, tm_generate, and tm_embed
// all run a genuine forward pass against a genuine loaded .tm model —
// see model.cpp/ops.h and docs/architecture/native-model-contract.md for
// what they compute, and STATUS.md for how this was verified (logits
// compared directly against the real Python model on identical weights
// and input, to within float32 precision).
//
// tm_generate's prefill is O(T) sequential forward_one_token_cached calls
// rather than one batched prefill call — a real simplification, not a
// stub: it reuses the already-equivalence-tested decode path directly
// (see native/tests/test_model_equivalence.cpp) instead of a second
// prefill-specific code path, at the cost of prefill being O(T) forward
// calls instead of O(1) — correctness over premature optimization, per
// the brief's own section 6.
#include "../include/tinymind.h"

#include <cstring>
#include <random>
#include <sstream>
#include <string>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

// The real definition of the opaque handle declared in tinymind.h. Kept
// out of the public header per that header's own design note ("do not
// expose implementation internals through the public ABI").
struct tm_context {
    bool has_model = false;
    tinymind::Model model;
};

namespace {

// Minimal JSON string escaping for embedding generated text into the
// tm_generate response envelope — handles exactly the characters JSON
// requires escaping (RFC 8259 section 7), nothing more.
std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

int write_response(const std::string& json, char* out_buffer, size_t out_buffer_size) {
    if (json.size() + 1 > out_buffer_size) return TM_ERROR_BUFFER_TOO_SMALL;
    std::memcpy(out_buffer, json.c_str(), json.size() + 1);
    return TM_OK;
}

}  // namespace

tm_context_t* tm_create(void) {
    try {
        return new tm_context();
    } catch (const std::bad_alloc&) {
        return nullptr;
    }
}

int tm_load_model(tm_context_t* ctx, const char* path) {
    if (ctx == nullptr || path == nullptr) return TM_ERROR_INVALID_ARGUMENT;
    try {
        ctx->model = tinymind::Model::load(path);
        ctx->has_model = true;
        return TM_OK;
    } catch (const tinymind::TmFormatError&) {
        return TM_ERROR_MODEL_CORRUPT;
    } catch (const std::exception&) {
        return TM_ERROR_INTERNAL;
    }
}

int tm_generate(tm_context_t* ctx, const char* prompt,
                const tm_generation_config_t* config,
                char* out_buffer, size_t out_buffer_size) {
    if (ctx == nullptr || prompt == nullptr || config == nullptr || out_buffer == nullptr) {
        return TM_ERROR_INVALID_ARGUMENT;
    }
    if (!ctx->has_model) return TM_ERROR_NOT_LOADED;

    try {
        tinymind::ByteTokenizer tokenizer;
        std::vector<int32_t> prompt_ids = tokenizer.encode(std::string(prompt), /*add_bos=*/true);
        const auto& geo = ctx->model.geometry();

        if (static_cast<int>(prompt_ids.size()) > geo.max_seq_len) {
            return write_response(
                "{\"error\": \"prompt is longer than this model's max_seq_len\"}", out_buffer, out_buffer_size)
                       == TM_OK ? TM_ERROR_GENERATION_FAILED : TM_ERROR_BUFFER_TOO_SMALL;
        }

        tinymind::KVCache cache(geo.num_layers, geo.max_seq_len, geo.num_kv_heads, geo.head_dim());
        std::vector<float> logits;
        for (size_t t = 0; t < prompt_ids.size(); ++t) {
            logits = ctx->model.forward_one_token_cached(prompt_ids[t], cache, static_cast<int>(t));
        }

        int max_new_tokens = config->max_new_tokens;
        int available = geo.max_seq_len - static_cast<int>(prompt_ids.size());
        if (max_new_tokens > available) max_new_tokens = available;  // see module note: an ordinary
                                                                      // stopping condition, not an error —
                                                                      // matches generation.py's own clamp

        std::mt19937 rng = tinymind::make_rng(config->seed);
        tinymind::SelectTokenOptions options;
        options.temperature = config->temperature;
        options.top_k = config->top_k > 0 ? config->top_k : -1;
        options.top_p = config->top_p > 0.0f ? config->top_p : -1.0f;

        std::vector<int32_t> generated_ids;
        std::string finish_reason = "length";
        for (int i = 0; i < max_new_tokens; ++i) {
            size_t next = tinymind::select_token(logits, options, rng);
            generated_ids.push_back(static_cast<int32_t>(next));
            if (static_cast<int32_t>(next) == tinymind::ByteTokenizer::kEos) {
                finish_reason = "stop";
                break;
            }
            int position = static_cast<int>(prompt_ids.size()) + static_cast<int>(generated_ids.size()) - 1;
            logits = ctx->model.forward_one_token_cached(static_cast<int32_t>(next), cache, position);
        }

        std::string text = tokenizer.decode(generated_ids);
        std::ostringstream json;
        json << "{\"text\": \"" << json_escape(text) << "\", "
            << "\"tokens_generated\": " << generated_ids.size() << ", "
            << "\"finish_reason\": \"" << finish_reason << "\"}";

        return write_response(json.str(), out_buffer, out_buffer_size);
    } catch (const std::exception&) {
        return TM_ERROR_GENERATION_FAILED;
    }
}

int tm_reset(tm_context_t* ctx) {
    if (ctx == nullptr) return TM_ERROR_INVALID_ARGUMENT;
    // Resetting clears conversation/KV state, not loaded weights (per
    // tinymind.model.backend.ModelBackend.reset()'s contract on the Python
    // side) — and every tm_generate/tm_embed call above builds its own
    // fresh KVCache internally rather than holding one across calls, the
    // same reasoning EchoBackend and TransformerBackend's own reset()
    // document for themselves on the Python side. Nothing to clear here.
    return TM_OK;
}

int tm_embed(tm_context_t* ctx, const char* text,
            float* out_embedding, size_t out_embedding_capacity, size_t* out_dim) {
    if (ctx == nullptr || text == nullptr || out_dim == nullptr) return TM_ERROR_INVALID_ARGUMENT;
    if (!ctx->has_model) return TM_ERROR_NOT_LOADED;

    try {
        tinymind::ByteTokenizer tokenizer;
        std::vector<int32_t> ids = tokenizer.encode(std::string(text), /*add_bos=*/true, /*add_eos=*/true);
        const auto& geo = ctx->model.geometry();
        if (static_cast<int>(ids.size()) > geo.max_seq_len) {
            ids.resize(static_cast<size_t>(geo.max_seq_len));  // truncate rather than fail — see
                                                                // this function's design note below
        }

        std::vector<float> hidden_states;
        ctx->model.forward_with_hidden_states(ids, hidden_states);

        // Mean-pool over sequence positions, then L2-normalize — the same
        // approach TransformerBackend.embed() uses on the Python side
        // (tinymind/model/backends/transformer.py), for the same reason:
        // no calibrated sentence-embedding head is trained in this
        // delivery, so a simple, honest pooling of real hidden states is
        // what's actually implementable — see that file's docstring.
        *out_dim = static_cast<size_t>(geo.hidden_size);
        if (out_embedding == nullptr || out_embedding_capacity < static_cast<size_t>(geo.hidden_size)) {
            return TM_ERROR_BUFFER_TOO_SMALL;
        }

        int T = static_cast<int>(ids.size());
        for (int d = 0; d < geo.hidden_size; ++d) {
            double sum = 0.0;
            for (int t = 0; t < T; ++t) sum += hidden_states[static_cast<size_t>(t) * geo.hidden_size + d];
            out_embedding[d] = static_cast<float>(sum / T);
        }
        double norm = 0.0;
        for (int d = 0; d < geo.hidden_size; ++d) norm += static_cast<double>(out_embedding[d]) * out_embedding[d];
        norm = std::sqrt(norm);
        if (norm > 0.0) {
            for (int d = 0; d < geo.hidden_size; ++d) out_embedding[d] = static_cast<float>(out_embedding[d] / norm);
        }
        return TM_OK;
    } catch (const std::exception&) {
        return TM_ERROR_GENERATION_FAILED;
    }
}

void tm_free(tm_context_t* ctx) {
    delete ctx;  // delete on nullptr is well-defined (a no-op), matching this header's documented contract
}
