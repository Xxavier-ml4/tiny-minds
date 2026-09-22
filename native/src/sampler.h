// sampler.h — token selection over a logits vector.
//
// STATUS: real, and deliberately a straight port of the tested Python
// implementation in tinymind/runtime/sampling.py (softmax, temperature,
// top-k, top-p, argmax, weighted sampling) — pure math over a
// caller-supplied logits vector, so like tensor.h it needs no trained
// model to be correct. Keeping the two implementations logically identical
// (same functions, same order of operations in select_token) means the
// Python reference test suite (tests/test_sampling.py) is effectively also
// a specification for this file, even though nothing here executes it
// directly.
#ifndef TINYMIND_SAMPLER_H
#define TINYMIND_SAMPLER_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace tinymind {

inline std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) return {};
    float peak = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_vals(logits.size());
    float total = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_vals[i] = std::exp(logits[i] - peak);
        total += exp_vals[i];
    }
    if (total <= 0.0f) {
        return std::vector<float>(logits.size(), 1.0f / logits.size());
    }
    for (float& v : exp_vals) v /= total;
    return exp_vals;
}

inline std::vector<float> apply_temperature(const std::vector<float>& logits, float temperature) {
    if (temperature < 0.0f) throw std::invalid_argument("temperature must be >= 0");
    if (temperature == 0.0f) return logits;  // caller should use argmax at temperature 0
    std::vector<float> out(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) out[i] = logits[i] / temperature;
    return out;
}

inline std::vector<float> top_k_filter(const std::vector<float>& probs, int k) {
    if (k <= 0) throw std::invalid_argument("k must be positive");
    if (static_cast<size_t>(k) >= probs.size()) return probs;
    std::vector<float> sorted_desc = probs;
    std::sort(sorted_desc.begin(), sorted_desc.end(), std::greater<float>());
    float threshold = sorted_desc[k - 1];
    std::vector<float> filtered(probs.size());
    float total = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        filtered[i] = (probs[i] >= threshold) ? probs[i] : 0.0f;
        total += filtered[i];
    }
    if (total > 0.0f) {
        for (float& v : filtered) v /= total;
    }
    return filtered;
}

inline std::vector<float> top_p_filter(const std::vector<float>& probs, float p) {
    if (!(p > 0.0f && p <= 1.0f)) throw std::invalid_argument("p must be in (0, 1]");
    std::vector<size_t> order(probs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return probs[a] > probs[b]; });

    std::vector<bool> keep(probs.size(), false);
    float cumulative = 0.0f;
    bool any_kept = false;
    for (size_t idx : order) {
        if (cumulative >= p && any_kept) break;
        keep[idx] = true;
        any_kept = true;
        cumulative += probs[idx];
    }
    std::vector<float> filtered(probs.size());
    float total = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        filtered[i] = keep[i] ? probs[i] : 0.0f;
        total += filtered[i];
    }
    if (total > 0.0f) {
        for (float& v : filtered) v /= total;
    }
    return filtered;
}

inline size_t argmax(const std::vector<float>& values) {
    if (values.empty()) throw std::invalid_argument("argmax of an empty sequence");
    return static_cast<size_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

inline size_t weighted_sample(const std::vector<float>& probs, std::mt19937& rng) {
    if (probs.empty()) throw std::invalid_argument("sample from an empty distribution");
    float total = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (total <= 0.0f) throw std::invalid_argument("sample from a distribution that sums to <= 0");
    std::uniform_real_distribution<float> dist(0.0f, total);
    float target = dist(rng);
    float cumulative = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (cumulative >= target) return i;
    }
    return probs.size() - 1;  // floating-point fallback, mirrors sampling.py
}

struct SelectTokenOptions {
    float temperature = 0.0f;
    int top_k = -1;   // -1 = unset, mirrors Python's Optional[int] = None
    float top_p = -1.0f;  // -1 = unset
};

inline size_t select_token(const std::vector<float>& logits, const SelectTokenOptions& options,
                           std::mt19937& rng) {
    if (options.temperature == 0.0f) return argmax(logits);
    std::vector<float> probs = softmax(apply_temperature(logits, options.temperature));
    if (options.top_k > 0) probs = top_k_filter(probs, options.top_k);
    if (options.top_p > 0.0f) probs = top_p_filter(probs, options.top_p);
    return weighted_sample(probs, rng);
}

// Defined in sampler.cpp — a seeded RNG factory, for reproducible sampling
// given tinymind.runtime.generation.GenerationConfig.seed on the Python side.
std::mt19937 make_rng(uint64_t seed);

}  // namespace tinymind

#endif  // TINYMIND_SAMPLER_H
