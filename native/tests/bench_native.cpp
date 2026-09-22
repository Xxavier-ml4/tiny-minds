// bench_native.cpp — measures the native runtime on a real `.tm` model: load time, full-sequence prefill latency,
// single-token cached decode latency, and peak resident memory. Prints one JSON object on stdout.
//
// This is a *measurement* program: it has no notion of correct output (test_model_equivalence.cpp + the Python
// equivalence tests own that). Prompt tokens are a fixed deterministic pattern, decode is greedy, so a run is
// reproducible for a given model. Built with plain g++ -O2 (no -march=native): the numbers are for this
// machine's CPU, NOT a phone; they show the native kernels' cost, and a device must be measured separately.
//
// Usage: bench_native <model.tm> <prompt_len> <decode_tokens> [repeats=5]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../src/kv_cache.h"
#include "../src/model.h"

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static int argmax(const std::vector<float>& v) {
    return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Resident-set figures from /proc/self/status. getrusage's ru_maxrss is NOT used: on Linux a child process inherits
// its parent's high-water mark across fork+exec, so a benchmark launched from a big Python process would report the
// parent's memory (the first version of this benchmark did exactly that: 316 MB for a 5 MB model). VmHWM/VmRSS are
// read from the fresh address space and are this process's own.
static long status_kb(const char* key) {
    FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long value = -1;
    const size_t n = std::strlen(key);
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, key, n) == 0) { value = std::atol(line + n); break; }
    }
    std::fclose(f);
    return value;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.tm> <prompt_len> <decode_tokens> [repeats]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int prompt_len = std::atoi(argv[2]);
    const int decode_tokens = std::atoi(argv[3]);
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 5;
    const long baseline_rss_kb = status_kb("VmRSS:");  // the bare process, before the model is loaded

    auto t0 = Clock::now();
    tinymind::Model model;
    try {
        model = tinymind::Model::load(path);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "load failed: %s\n", exc.what());
        return 1;
    }
    const double load_ms = ms_since(t0);
    const auto& g = model.geometry();
    if (prompt_len + decode_tokens > g.max_seq_len) {
        std::fprintf(stderr, "prompt_len + decode_tokens exceeds max_seq_len %d\n", g.max_seq_len);
        return 2;
    }

    std::vector<int32_t> prompt(prompt_len);
    for (int i = 0; i < prompt_len; ++i) prompt[i] = 4 + (i * 7 + 3) % 200;  // byte-token range of the byte tokenizer

    std::vector<double> prefill_ms, decode_ms_per_token;
    long checksum = 0;
    for (int r = 0; r < repeats; ++r) {
        // full-sequence forward over the prompt (what a host does to ingest a prompt in one call)
        auto a = Clock::now();
        std::vector<float> logits = model.forward(prompt);
        prefill_ms.push_back(ms_since(a));
        checksum += argmax(std::vector<float>(logits.end() - g.vocab_size, logits.end()));

        // cached decoding: fill the cache token by token, then generate greedily
        tinymind::KVCache cache(g.num_layers, g.max_seq_len, g.num_kv_heads, g.head_dim());
        std::vector<float> step;
        for (int i = 0; i < prompt_len; ++i) step = model.forward_one_token_cached(prompt[i], cache, i);
        int token = argmax(step);
        auto b = Clock::now();
        for (int i = 0; i < decode_tokens; ++i) {
            step = model.forward_one_token_cached(token, cache, prompt_len + i);
            token = argmax(step);
            checksum += token;
        }
        decode_ms_per_token.push_back(ms_since(b) / decode_tokens);
    }
    const long peak_rss_kb = status_kb("VmHWM:");
    const double decode_med = median(decode_ms_per_token);
    std::printf("{\"model\": \"%s\", \"parameters\": %zu, \"hidden\": %d, \"layers\": %d, \"heads\": %d, \"kv_heads\": %d, "
                "\"prompt_len\": %d, \"decode_tokens\": %d, \"repeats\": %d, \"load_ms\": %.2f, "
                "\"prefill_full_forward_ms_median\": %.2f, \"prefill_tokens_per_sec\": %.1f, "
                "\"decode_ms_per_token_median\": %.3f, \"decode_tokens_per_sec\": %.1f, \"baseline_rss_kb\": %ld, \"peak_rss_kb\": %ld, \"checksum\": %ld}\n",
                path.c_str(), model.parameter_count(), g.hidden_size, g.num_layers, g.num_heads, g.num_kv_heads, prompt_len,
                decode_tokens, repeats, load_ms, median(prefill_ms), prompt_len / (median(prefill_ms) / 1000.0), decode_med,
                1000.0 / decode_med, baseline_rss_kb, peak_rss_kb, checksum);
    return 0;
}
