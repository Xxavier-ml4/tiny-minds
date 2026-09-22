// model.h — the native model: architecture forward pass + weight loading.
//
// STATUS: real as of this phase. Loads a real `.tm` file (tm_reader.h)
// and runs a real forward pass (ops.h) matching
// docs/architecture/native-model-contract.md. Single-sequence (no batch
// dimension) — the mobile/on-device inference case this project targets
// runs one conversation at a time per native context (see
// native/include/tinymind.h's per-context design note), not batched
// training-style throughput.
#ifndef TINYMIND_MODEL_H
#define TINYMIND_MODEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache.h"
#include "ops.h"
#include "tensor.h"
#include "tm_reader.h"

namespace tinymind {

// Mirrors tinymind.model.config.ModelConfig field-for-field (see that
// Python module's docstring) — kept in sync deliberately so a `.tm` file's
// metadata block (tinymind.runtime.format) decodes into the same shape on
// both sides of the native boundary.
struct ModelGeometry {
    int32_t hidden_size = 0;
    int32_t num_layers = 0;
    int32_t num_heads = 0;
    int32_t num_kv_heads = 0;
    int32_t intermediate_size = 0;
    int32_t max_seq_len = 0;
    int32_t vocab_size = 0;
    double rope_theta = 10000.0;
    double norm_epsilon = 1e-6;
    bool tie_embeddings = true;

    int32_t head_dim() const { return hidden_size / num_heads; }
};

struct LayerWeights {
    std::vector<float> attn_norm_weight;
    std::vector<float> q_proj_weight;
    std::vector<float> k_proj_weight;
    std::vector<float> v_proj_weight;
    std::vector<float> o_proj_weight;
    std::vector<float> mlp_norm_weight;
    std::vector<float> gate_proj_weight;
    std::vector<float> up_proj_weight;
    std::vector<float> down_proj_weight;
};

class Model {
public:
    // Loads and validates a .tm file (bounds/overlap/header-region checks,
    // checksum-verified tensor reads — see tm_reader.h), reconstructing
    // the exact architecture described in the file's metadata. Throws
    // TmFormatError on any malformed/corrupt/incompatible file — a model
    // file is untrusted input (brief section 28) on this side of the
    // boundary exactly as much as on the Python side.
    static Model load(const std::string& path);

    // Runs one forward pass for `input_ids` (T token ids), returning
    // logits over the vocabulary for every position: [T, vocab_size],
    // row-major, matching tinymind.model.model.TinyMindTransformer.forward
    // with use_cache=false. No softmax applied (native-model-contract.md
    // section 9: softmax only ever happens at the sampling layer).
    std::vector<float> forward(const std::vector<int32_t>& input_ids);

    // Same as forward(), but also writes the pre-LM-head hidden states
    // ([T, hidden_size]) into `out_hidden_states` — the native counterpart
    // to tinymind.model.model.TinyMindTransformer.forward's
    // output_hidden_states=True, used by tm_embed (see runtime.cpp) the
    // same way TransformerBackend.embed() uses it on the Python side.
    std::vector<float> forward_with_hidden_states(const std::vector<int32_t>& input_ids,
                                                   std::vector<float>& out_hidden_states);

    // Cached, single-new-token decode step — mirrors
    // tinymind.model.generation.generate_with_cache_ids's decode loop.
    // `cache` must already hold every prior position's K/V; this call
    // both reads and (via KVCache::update, invoked internally per layer)
    // extends it by exactly one position. Returns logits for just the
    // new token: [vocab_size].
    std::vector<float> forward_one_token_cached(int32_t token_id, KVCache& cache, int position);

    const ModelGeometry& geometry() const { return geometry_; }
    size_t parameter_count() const { return parameter_count_; }

private:
    ModelGeometry geometry_;
    std::vector<float> embed_tokens_;  // [vocab_size, hidden_size]
    std::vector<LayerWeights> layers_;
    std::vector<float> final_norm_weight_;
    std::vector<float> lm_head_weight_;  // only populated if !tie_embeddings
    RopeCache rope_cache_;
    size_t parameter_count_ = 0;

    std::vector<float> read_float_tensor(const TmFile& tm, const std::string& name, size_t expected_count);
};

}  // namespace tinymind

#endif  // TINYMIND_MODEL_H
