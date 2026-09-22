// ops.h — the actual transformer math: linear projection, RMSNorm, RoPE,
// causal self-attention, SwiGLU. Forward-pass only (inference has no
// backward pass to compute) — this is why native/src needs no autodiff
// engine the way tinymind/model/tensor.py does on the Python side; see
// that file's docstring for why the Python side needs one at all.
//
// STATUS: real. Every formula here is written directly against
// docs/architecture/native-model-contract.md, which is itself extracted
// from the actual Python implementation (tinymind/model/*.py) — not
// independently derived. Correctness is verified end-to-end by comparing
// this code's output logits against the real Python model's logits on
// identical weights and input (native/tests/test_model_equivalence.cpp),
// not just by the formulas looking right in isolation.
//
// Simple, unoptimized loops throughout (no SIMD, no blocking, no
// threading) — brief section 6: "prioritize correctness over premature
// optimization." A row-major [T, features] layout is used for activations
// throughout, matching NumPy's default (row-major) layout on the Python
// side.
#ifndef TINYMIND_OPS_H
#define TINYMIND_OPS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace tinymind {

// y[t, o] = sum_i x[t, i] * W[o, i]  — W is [out_features, in_features],
// the same orientation as tinymind.model.linear.Linear.weight (see
// native-model-contract.md section 2: "y = x @ W^T", W stored [out, in]).
inline void linear(const float* x, int T, int in_features,
                   const float* W, int out_features, float* y) {
    for (int t = 0; t < T; ++t) {
        const float* x_row = x + static_cast<size_t>(t) * in_features;
        float* y_row = y + static_cast<size_t>(t) * out_features;
        for (int o = 0; o < out_features; ++o) {
            const float* w_row = W + static_cast<size_t>(o) * in_features;
            float sum = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
}

// RMSNorm: x * rsqrt(mean(x^2, axis=-1) + eps) * weight — see
// native-model-contract.md section 6. eps inside the sqrt, one weight per
// feature, no mean-subtraction.
inline void rmsnorm(const float* x, int T, int hidden_size, const float* weight,
                    float eps, float* out) {
    for (int t = 0; t < T; ++t) {
        const float* row = x + static_cast<size_t>(t) * hidden_size;
        float* out_row = out + static_cast<size_t>(t) * hidden_size;
        double sum_sq = 0.0;  // accumulate in double: hidden_size can be large enough that
                              // float accumulation would visibly drift from NumPy's own
                              // (also effectively higher-precision, pairwise-summed) mean().
        for (int i = 0; i < hidden_size; ++i) {
            sum_sq += static_cast<double>(row[i]) * static_cast<double>(row[i]);
        }
        float variance = static_cast<float>(sum_sq / hidden_size);
        float inv_rms = 1.0f / std::sqrt(variance + eps);
        for (int i = 0; i < hidden_size; ++i) {
            out_row[i] = row[i] * inv_rms * weight[i];
        }
    }
}

// RoPE frequency cache: cos/sin, each [max_seq_len, head_dim], the second
// half of the last axis a copy of the first half — see
// native-model-contract.md section 4 and tinymind/model/positional.py.
struct RopeCache {
    std::vector<float> cos;
    std::vector<float> sin;
    int head_dim;
    int max_seq_len;

    static RopeCache build(int head_dim, int max_seq_len, double theta) {
        RopeCache cache;
        cache.head_dim = head_dim;
        cache.max_seq_len = max_seq_len;
        cache.cos.resize(static_cast<size_t>(max_seq_len) * head_dim);
        cache.sin.resize(static_cast<size_t>(max_seq_len) * head_dim);
        int half = head_dim / 2;
        std::vector<double> inv_freq(half);
        for (int i = 0; i < half; ++i) {
            inv_freq[i] = 1.0 / std::pow(theta, static_cast<double>(2 * i) / head_dim);
        }
        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < half; ++i) {
                double angle = static_cast<double>(pos) * inv_freq[i];
                float c = static_cast<float>(std::cos(angle));
                float s = static_cast<float>(std::sin(angle));
                cache.cos[static_cast<size_t>(pos) * head_dim + i] = c;
                cache.cos[static_cast<size_t>(pos) * head_dim + half + i] = c;  // duplicated second half
                cache.sin[static_cast<size_t>(pos) * head_dim + i] = s;
                cache.sin[static_cast<size_t>(pos) * head_dim + half + i] = s;
            }
        }
        return cache;
    }
};

// Applies RoPE in place to one [T, num_heads, head_dim] tensor (Q or K),
// using absolute position `position_offset + t` for row t — see
// native-model-contract.md section 4's rotate-half formula.
inline void apply_rope(float* x, int T, int num_heads, int head_dim,
                       const RopeCache& cache, int position_offset) {
    int half = head_dim / 2;
    std::vector<float> rotated(head_dim);
    for (int t = 0; t < T; ++t) {
        int pos = position_offset + t;
        const float* cos_row = cache.cos.data() + static_cast<size_t>(pos) * head_dim;
        const float* sin_row = cache.sin.data() + static_cast<size_t>(pos) * head_dim;
        for (int h = 0; h < num_heads; ++h) {
            float* vec = x + (static_cast<size_t>(t) * num_heads + h) * head_dim;
            // rotate_half(x) = concat(-x[half:], x[:half])
            for (int i = 0; i < half; ++i) {
                rotated[i] = -vec[half + i];
                rotated[half + i] = vec[i];
            }
            for (int i = 0; i < head_dim; ++i) {
                vec[i] = vec[i] * cos_row[i] + rotated[i] * sin_row[i];
            }
        }
    }
}

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// down(silu(gate(x)) * up(x)) — native-model-contract.md section 7.
inline void swiglu_mlp(const float* x, int T, int hidden_size, int intermediate_size,
                       const float* gate_w, const float* up_w, const float* down_w,
                       float* out, std::vector<float>& scratch_gate, std::vector<float>& scratch_up) {
    scratch_gate.resize(static_cast<size_t>(T) * intermediate_size);
    scratch_up.resize(static_cast<size_t>(T) * intermediate_size);
    linear(x, T, hidden_size, gate_w, intermediate_size, scratch_gate.data());
    linear(x, T, hidden_size, up_w, intermediate_size, scratch_up.data());
    for (size_t i = 0; i < scratch_gate.size(); ++i) {
        scratch_gate[i] = silu(scratch_gate[i]) * scratch_up[i];
    }
    linear(scratch_gate.data(), T, intermediate_size, down_w, hidden_size, out);
}

// Numerically-stable softmax over the first `n` elements of `x`, in place —
// native-model-contract.md section 9 (max-subtraction before exp).
inline void softmax_inplace(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; ++i) max_val = std::max(max_val, x[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = static_cast<float>(1.0 / sum);
    for (int i = 0; i < n; ++i) x[i] *= inv_sum;
}

// Causal self-attention for one layer, non-cached (full sequence at once).
// Q: [T, num_heads, head_dim], K/V: [T, num_kv_heads, head_dim] (already
// RoPE'd for Q/K by the caller). GQA repetition (native-model-contract.md
// section 3: kv head i//n_rep, contiguous blocks, not interleaved) is
// applied inline rather than materializing a repeated K/V copy.
inline void causal_attention(const float* q, const float* k, const float* v,
                             int T, int num_heads, int num_kv_heads, int head_dim,
                             float* out) {
    int n_rep = num_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(T);

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / n_rep;
        for (int t = 0; t < T; ++t) {
            const float* q_vec = q + (static_cast<size_t>(t) * num_heads + h) * head_dim;
            // Causal: query position t attends to key positions 0..t only.
            for (int s = 0; s <= t; ++s) {
                const float* k_vec = k + (static_cast<size_t>(s) * num_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) dot += q_vec[d] * k_vec[d];
                scores[s] = dot * scale;
            }
            softmax_inplace(scores.data(), t + 1);

            float* out_vec = out + (static_cast<size_t>(t) * num_heads + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) out_vec[d] = 0.0f;
            for (int s = 0; s <= t; ++s) {
                const float* v_vec = v + (static_cast<size_t>(s) * num_kv_heads + kv_h) * head_dim;
                float weight = scores[s];
                for (int d = 0; d < head_dim; ++d) out_vec[d] += weight * v_vec[d];
            }
        }
    }
}

}  // namespace tinymind

#endif  // TINYMIND_OPS_H
