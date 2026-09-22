// kv_cache.h — fixed-capacity key/value cache storage.
//
// STATUS: real for what it actually is — a bounds-checked, pre-allocated
// buffer for per-layer, per-head key/value vectors, indexed by position.
// It does not perform attention math (that reads from this buffer; it
// belongs in model.cpp, which is a stub — see that file) and does not know
// about any specific model's dimensions beyond what it's constructed with,
// so like tensor.h and sampler.h it doesn't need a trained model to be
// correct, and unlike model.cpp it isn't one.
#ifndef TINYMIND_KV_CACHE_H
#define TINYMIND_KV_CACHE_H

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tinymind {

class KVCache {
public:
    // num_layers x max_seq_len x num_kv_heads x head_dim, for keys and
    // values separately — the shape every GQA/MQA/MHA attention variant in
    // tinymind.model.config.ModelConfig needs (num_kv_heads == num_heads
    // degenerates to plain MHA, which this buffer handles identically; see
    // that Python module's docstring on the attention_type field).
    KVCache(int num_layers, int max_seq_len, int num_kv_heads, int head_dim)
        : num_layers_(num_layers), max_seq_len_(max_seq_len),
          num_kv_heads_(num_kv_heads), head_dim_(head_dim) {
        if (num_layers <= 0 || max_seq_len <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
            throw std::invalid_argument("KVCache: all dimensions must be positive");
        }
        size_t per_layer = static_cast<size_t>(max_seq_len) * num_kv_heads * head_dim;
        keys_.assign(static_cast<size_t>(num_layers) * per_layer, 0.0f);
        values_.assign(static_cast<size_t>(num_layers) * per_layer, 0.0f);
        current_length_ = 0;
    }

    int current_length() const { return current_length_; }
    int max_seq_len() const { return max_seq_len_; }

    // Append one position's key/value vectors (already interleaved across
    // heads: num_kv_heads * head_dim floats each) for `layer`. Returns
    // false (does not write, does not throw) if the cache is full — the
    // caller (model.cpp, once implemented) decides whether that's an error
    // or a signal to evict/slide the window (brief section 13's
    // `sliding_window` config field).
    bool append(int layer, const float* key_vec, const float* value_vec) {
        check_layer(layer);
        if (current_length_ >= max_seq_len_) return false;
        size_t offset = slot_offset(layer, current_length_);
        size_t width = static_cast<size_t>(num_kv_heads_) * head_dim_;
        std::copy(key_vec, key_vec + width, keys_.begin() + offset);
        std::copy(value_vec, value_vec + width, values_.begin() + offset);
        if (layer == num_layers_ - 1) {
            // Only advance the shared position counter once every layer for
            // this position has been written, i.e. after the last layer —
            // callers must append layers 0..num_layers-1 in order for one
            // position before appending the next position.
            current_length_ += 1;
        }
        return true;
    }

    const float* key_at(int layer, int position) const {
        check_layer(layer);
        check_position(position);
        return keys_.data() + slot_offset(layer, position);
    }

    const float* value_at(int layer, int position) const {
        check_layer(layer);
        check_position(position);
        return values_.data() + slot_offset(layer, position);
    }

    void reset() { current_length_ = 0; }  // keep the allocation, drop the content

private:
    void check_layer(int layer) const {
        if (layer < 0 || layer >= num_layers_) throw std::out_of_range("KVCache: layer out of range");
    }
    void check_position(int position) const {
        if (position < 0 || position >= max_seq_len_) {
            throw std::out_of_range("KVCache: position out of range");
        }
    }
    size_t slot_offset(int layer, int position) const {
        size_t per_layer = static_cast<size_t>(max_seq_len_) * num_kv_heads_ * head_dim_;
        size_t per_position = static_cast<size_t>(num_kv_heads_) * head_dim_;
        return static_cast<size_t>(layer) * per_layer + static_cast<size_t>(position) * per_position;
    }

    int num_layers_;
    int max_seq_len_;
    int num_kv_heads_;
    int head_dim_;
    int current_length_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

// Defined in kv_cache.cpp — memory footprint in bytes for a given
// configuration, without needing an instance (for RAM accounting per the
// engineering brief sections 26-27 and 50).
size_t kv_cache_footprint_bytes(int num_layers, int max_seq_len, int num_kv_heads, int head_dim);

}  // namespace tinymind

#endif  // TINYMIND_KV_CACHE_H
