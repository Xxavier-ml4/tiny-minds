// model.cpp — real weight loading and forward pass, matching
// docs/architecture/native-model-contract.md. See model.h for the class
// shape and STATUS.md for how this was verified (comparison against the
// real Python model's logits on identical weights/input, not just unit
// tests of each op in isolation).
#include "model.h"

#include <stdexcept>

namespace tinymind {

namespace {

constexpr const char* kArchitectureTag = "tinymind-transformer-v1";

int require_int(const JsonValue& config, const char* key) {
    return config.get(key).as_int();
}

}  // namespace

std::vector<float> Model::read_float_tensor(const TmFile& tm, const std::string& name,
                                            size_t expected_count) {
    auto raw = tm.read_tensor(name);
    if (raw.size() != expected_count * sizeof(float)) {
        throw TmFormatError("tensor '" + name + "' has " + std::to_string(raw.size()) +
                            " bytes, expected " + std::to_string(expected_count * sizeof(float)) +
                            " (" + std::to_string(expected_count) + " float32 values)");
    }
    std::vector<float> values(expected_count);
    std::memcpy(values.data(), raw.data(), raw.size());
    return values;
}

Model Model::load(const std::string& path) {
    TmFile tm = TmFile::load(path);

    const std::string architecture = tm.metadata().get("architecture").as_string();
    if (architecture != kArchitectureTag) {
        throw TmFormatError("this .tm file declares architecture '" + architecture +
                            "', expected '" + std::string(kArchitectureTag) + "' — it was not "
                            "exported by tinymind.model.tm_export.export_to_tm, or is a "
                            "different format version");
    }

    const JsonValue& config = tm.metadata().get("model_config");

    Model model;
    ModelGeometry& geo = model.geometry_;
    geo.hidden_size = require_int(config, "hidden_size");
    geo.num_layers = require_int(config, "num_layers");
    geo.num_heads = require_int(config, "num_heads");
    geo.num_kv_heads = require_int(config, "num_kv_heads");
    geo.intermediate_size = require_int(config, "intermediate_size");
    geo.max_seq_len = require_int(config, "max_seq_len");
    geo.vocab_size = require_int(config, "vocab_size");
    geo.rope_theta = config.get("rope_theta").as_number();
    geo.norm_epsilon = config.get("norm_epsilon").as_number();
    geo.tie_embeddings = config.get("tie_embeddings").as_bool();

    if (geo.hidden_size <= 0 || geo.num_layers <= 0 || geo.num_heads <= 0 || geo.num_kv_heads <= 0 ||
        geo.intermediate_size <= 0 || geo.max_seq_len <= 0 || geo.vocab_size <= 0) {
        throw TmFormatError("model_config declares a non-positive dimension — refusing to load");
    }
    if (geo.hidden_size % geo.num_heads != 0) {
        throw TmFormatError("hidden_size is not divisible by num_heads in model_config");
    }
    if (geo.num_heads % geo.num_kv_heads != 0) {
        throw TmFormatError("num_heads is not a multiple of num_kv_heads in model_config");
    }

    const int head_dim = geo.head_dim();
    if (head_dim % 2 != 0) {
        throw TmFormatError("head_dim (hidden_size / num_heads) must be even for RoPE");
    }

    model.embed_tokens_ = model.read_float_tensor(
        tm, "embed_tokens", static_cast<size_t>(geo.vocab_size) * geo.hidden_size);

    model.layers_.resize(geo.num_layers);
    for (int layer_idx = 0; layer_idx < geo.num_layers; ++layer_idx) {
        std::string prefix = "block_" + std::to_string(layer_idx) + ".";
        LayerWeights& layer = model.layers_[layer_idx];
        layer.attn_norm_weight = model.read_float_tensor(tm, prefix + "attn_norm.weight", geo.hidden_size);
        layer.q_proj_weight = model.read_float_tensor(
            tm, prefix + "attention.q_proj.weight",
            static_cast<size_t>(geo.num_heads) * head_dim * geo.hidden_size);
        layer.k_proj_weight = model.read_float_tensor(
            tm, prefix + "attention.k_proj.weight",
            static_cast<size_t>(geo.num_kv_heads) * head_dim * geo.hidden_size);
        layer.v_proj_weight = model.read_float_tensor(
            tm, prefix + "attention.v_proj.weight",
            static_cast<size_t>(geo.num_kv_heads) * head_dim * geo.hidden_size);
        layer.o_proj_weight = model.read_float_tensor(
            tm, prefix + "attention.o_proj.weight",
            static_cast<size_t>(geo.hidden_size) * geo.num_heads * head_dim);
        layer.mlp_norm_weight = model.read_float_tensor(tm, prefix + "mlp_norm.weight", geo.hidden_size);
        layer.gate_proj_weight = model.read_float_tensor(
            tm, prefix + "mlp.gate_proj.weight",
            static_cast<size_t>(geo.intermediate_size) * geo.hidden_size);
        layer.up_proj_weight = model.read_float_tensor(
            tm, prefix + "mlp.up_proj.weight",
            static_cast<size_t>(geo.intermediate_size) * geo.hidden_size);
        layer.down_proj_weight = model.read_float_tensor(
            tm, prefix + "mlp.down_proj.weight",
            static_cast<size_t>(geo.hidden_size) * geo.intermediate_size);
    }

    model.final_norm_weight_ = model.read_float_tensor(tm, "final_norm.weight", geo.hidden_size);

    if (!geo.tie_embeddings) {
        model.lm_head_weight_ = model.read_float_tensor(
            tm, "lm_head.weight", static_cast<size_t>(geo.vocab_size) * geo.hidden_size);
    }

    model.rope_cache_ = RopeCache::build(head_dim, geo.max_seq_len, geo.rope_theta);

    size_t count = model.embed_tokens_.size();
    for (const auto& layer : model.layers_) {
        count += layer.attn_norm_weight.size() + layer.q_proj_weight.size() + layer.k_proj_weight.size() +
                layer.v_proj_weight.size() + layer.o_proj_weight.size() + layer.mlp_norm_weight.size() +
                layer.gate_proj_weight.size() + layer.up_proj_weight.size() + layer.down_proj_weight.size();
    }
    count += model.final_norm_weight_.size() + model.lm_head_weight_.size();
    model.parameter_count_ = count;

    return model;
}

std::vector<float> Model::forward(const std::vector<int32_t>& input_ids) {
    std::vector<float> discard_hidden_states;
    return forward_with_hidden_states(input_ids, discard_hidden_states);
}

std::vector<float> Model::forward_with_hidden_states(const std::vector<int32_t>& input_ids,
                                                      std::vector<float>& out_hidden_states) {
    const ModelGeometry& geo = geometry_;
    const int T = static_cast<int>(input_ids.size());
    if (T == 0) throw std::invalid_argument("Model::forward() called with an empty input_ids");
    if (T > geo.max_seq_len) {
        throw std::invalid_argument("input length " + std::to_string(T) +
                                    " exceeds max_seq_len " + std::to_string(geo.max_seq_len));
    }
    for (int32_t id : input_ids) {
        if (id < 0 || id >= geo.vocab_size) {
            throw std::invalid_argument("token id " + std::to_string(id) + " is outside [0, " +
                                        std::to_string(geo.vocab_size) + ")");
        }
    }

    const int head_dim = geo.head_dim();
    const int q_width = geo.num_heads * head_dim;
    const int kv_width = geo.num_kv_heads * head_dim;

    // Embedding lookup: hidden[t, :] = embed_tokens[input_ids[t], :]
    std::vector<float> hidden(static_cast<size_t>(T) * geo.hidden_size);
    for (int t = 0; t < T; ++t) {
        const float* row = embed_tokens_.data() + static_cast<size_t>(input_ids[t]) * geo.hidden_size;
        std::memcpy(hidden.data() + static_cast<size_t>(t) * geo.hidden_size, row,
                   geo.hidden_size * sizeof(float));
    }

    std::vector<float> normed(static_cast<size_t>(T) * geo.hidden_size);
    std::vector<float> q(static_cast<size_t>(T) * q_width);
    std::vector<float> k(static_cast<size_t>(T) * kv_width);
    std::vector<float> v(static_cast<size_t>(T) * kv_width);
    std::vector<float> attn_raw(static_cast<size_t>(T) * q_width);
    std::vector<float> attn_out(static_cast<size_t>(T) * geo.hidden_size);
    std::vector<float> mlp_out(static_cast<size_t>(T) * geo.hidden_size);
    std::vector<float> scratch_gate, scratch_up;

    for (int layer_idx = 0; layer_idx < geo.num_layers; ++layer_idx) {
        const LayerWeights& layer = layers_[layer_idx];

        rmsnorm(hidden.data(), T, geo.hidden_size, layer.attn_norm_weight.data(),
               static_cast<float>(geo.norm_epsilon), normed.data());

        linear(normed.data(), T, geo.hidden_size, layer.q_proj_weight.data(), q_width, q.data());
        linear(normed.data(), T, geo.hidden_size, layer.k_proj_weight.data(), kv_width, k.data());
        linear(normed.data(), T, geo.hidden_size, layer.v_proj_weight.data(), kv_width, v.data());

        apply_rope(q.data(), T, geo.num_heads, head_dim, rope_cache_, /*position_offset=*/0);
        apply_rope(k.data(), T, geo.num_kv_heads, head_dim, rope_cache_, /*position_offset=*/0);

        causal_attention(q.data(), k.data(), v.data(), T, geo.num_heads, geo.num_kv_heads, head_dim,
                         attn_raw.data());
        linear(attn_raw.data(), T, q_width, layer.o_proj_weight.data(), geo.hidden_size, attn_out.data());

        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += attn_out[i];

        rmsnorm(hidden.data(), T, geo.hidden_size, layer.mlp_norm_weight.data(),
               static_cast<float>(geo.norm_epsilon), normed.data());
        swiglu_mlp(normed.data(), T, geo.hidden_size, geo.intermediate_size,
                  layer.gate_proj_weight.data(), layer.up_proj_weight.data(), layer.down_proj_weight.data(),
                  mlp_out.data(), scratch_gate, scratch_up);

        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] += mlp_out[i];
    }

    rmsnorm(hidden.data(), T, geo.hidden_size, final_norm_weight_.data(),
           static_cast<float>(geo.norm_epsilon), normed.data());

    out_hidden_states = normed;  // post-final-norm hidden states, same point Python's output_hidden_states captures

    std::vector<float> logits(static_cast<size_t>(T) * geo.vocab_size);
    const float* lm_head = geo.tie_embeddings ? embed_tokens_.data() : lm_head_weight_.data();
    linear(normed.data(), T, geo.hidden_size, lm_head, geo.vocab_size, logits.data());

    return logits;
}

std::vector<float> Model::forward_one_token_cached(int32_t token_id, KVCache& cache, int position) {
    const ModelGeometry& geo = geometry_;
    if (token_id < 0 || token_id >= geo.vocab_size) {
        throw std::invalid_argument("token id out of range");
    }
    if (position >= geo.max_seq_len) {
        throw std::invalid_argument("position exceeds max_seq_len");
    }
    const int head_dim = geo.head_dim();
    const int q_width = geo.num_heads * head_dim;
    const int kv_width = geo.num_kv_heads * head_dim;

    std::vector<float> hidden(geo.hidden_size);
    std::memcpy(hidden.data(), embed_tokens_.data() + static_cast<size_t>(token_id) * geo.hidden_size,
               geo.hidden_size * sizeof(float));

    std::vector<float> normed(geo.hidden_size);
    std::vector<float> q(q_width), k_new(kv_width), v_new(kv_width);
    std::vector<float> attn_raw(q_width), attn_out(geo.hidden_size), mlp_out(geo.hidden_size);
    std::vector<float> scratch_gate, scratch_up;
    std::vector<float> scores(position + 1);

    for (int layer_idx = 0; layer_idx < geo.num_layers; ++layer_idx) {
        const LayerWeights& layer = layers_[layer_idx];

        rmsnorm(hidden.data(), 1, geo.hidden_size, layer.attn_norm_weight.data(),
               static_cast<float>(geo.norm_epsilon), normed.data());

        linear(normed.data(), 1, geo.hidden_size, layer.q_proj_weight.data(), q_width, q.data());
        linear(normed.data(), 1, geo.hidden_size, layer.k_proj_weight.data(), kv_width, k_new.data());
        linear(normed.data(), 1, geo.hidden_size, layer.v_proj_weight.data(), kv_width, v_new.data());

        apply_rope(q.data(), 1, geo.num_heads, head_dim, rope_cache_, position);
        apply_rope(k_new.data(), 1, geo.num_kv_heads, head_dim, rope_cache_, position);

        cache.append(layer_idx, k_new.data(), v_new.data());

        // Attend the single new query position against every cached
        // key/value up to and including this position (GQA-grouped, same
        // convention as ops.h::causal_attention).
        int n_rep = geo.num_heads / geo.num_kv_heads;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (int h = 0; h < geo.num_heads; ++h) {
            int kv_h = h / n_rep;
            const float* q_vec = q.data() + static_cast<size_t>(h) * head_dim;
            for (int s = 0; s <= position; ++s) {
                const float* k_vec = cache.key_at(layer_idx, s) + static_cast<size_t>(kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
                scores[s] = dot * scale;
            }
            softmax_inplace(scores.data(), position + 1);
            float* out_vec = attn_raw.data() + static_cast<size_t>(h) * head_dim;
            for (int d = 0; d < head_dim; ++d) out_vec[d] = 0.0f;
            for (int s = 0; s <= position; ++s) {
                const float* v_vec = cache.value_at(layer_idx, s) + static_cast<size_t>(kv_h) * head_dim;
                float weight = scores[s];
                for (int d = 0; d < head_dim; ++d) out_vec[d] += weight * v_vec[d];
            }
        }

        linear(attn_raw.data(), 1, q_width, layer.o_proj_weight.data(), geo.hidden_size, attn_out.data());
        for (int i = 0; i < geo.hidden_size; ++i) hidden[i] += attn_out[i];

        rmsnorm(hidden.data(), 1, geo.hidden_size, layer.mlp_norm_weight.data(),
               static_cast<float>(geo.norm_epsilon), normed.data());
        swiglu_mlp(normed.data(), 1, geo.hidden_size, geo.intermediate_size,
                  layer.gate_proj_weight.data(), layer.up_proj_weight.data(), layer.down_proj_weight.data(),
                  mlp_out.data(), scratch_gate, scratch_up);
        for (int i = 0; i < geo.hidden_size; ++i) hidden[i] += mlp_out[i];
    }

    rmsnorm(hidden.data(), 1, geo.hidden_size, final_norm_weight_.data(),
           static_cast<float>(geo.norm_epsilon), normed.data());

    std::vector<float> logits(geo.vocab_size);
    const float* lm_head = geo.tie_embeddings ? embed_tokens_.data() : lm_head_weight_.data();
    linear(normed.data(), 1, geo.hidden_size, lm_head, geo.vocab_size, logits.data());
    return logits;
}

}  // namespace tinymind
