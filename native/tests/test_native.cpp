// test_native.cpp — plain assert()-based tests for the parts of native/src
// that are real (tensor.h, sampler.h, kv_cache.h, tokenizer.h) plus the
// documented-honest-failure behavior of the parts that are stubs
// (tm_load_model, tm_generate, tm_embed via the public C ABI).
//
// Uses assert() rather than a test framework (GoogleTest, Catch2) on
// purpose: this delivery's sandbox has no network access to fetch one, and
// the brief's own section 52 lists "unit tests" as a requirement without
// mandating a specific framework — assert() is real, dependency-free, and
// exercised below by actually compiling and running this file (see
// STATUS.md for the exact command used, since cmake itself isn't available
// in this sandbox either — this file's CMakeLists.txt is written for a
// normal environment that does have it, same caveat as
// native/CMakeLists.txt).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

#include "../include/tinymind.h"
#include "../src/kv_cache.h"
#include "../src/sampler.h"
#include "../src/tensor.h"
#include "../src/tokenizer.h"

static void test_tensor() {
    tinymind::Tensor t({2, 3}, tinymind::DType::kFloat32);
    assert(t.num_elements() == 6);
    assert(t.byte_size() == 24);
    t.at_f32(0) = 1.5f;
    t.at_f32(5) = 2.5f;
    assert(t.at_f32(0) == 1.5f);
    assert(t.at_f32(5) == 2.5f);

    bool threw = false;
    try {
        t.at_f32(6);  // out of range
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
    printf("test_tensor: OK\n");
}

static void test_sampler() {
    std::vector<float> logits = {1.0f, 5.0f, 1.0f};

    auto probs = tinymind::softmax(logits);
    float total = 0.0f;
    for (float p : probs) total += p;
    assert(std::fabs(total - 1.0f) < 1e-5f);

    assert(tinymind::argmax(logits) == 1);

    tinymind::SelectTokenOptions greedy_opts;
    greedy_opts.temperature = 0.0f;
    std::mt19937 rng = tinymind::make_rng(42);
    assert(tinymind::select_token(logits, greedy_opts, rng) == 1);

    // Stochastic: index 1 should win the large majority of draws.
    tinymind::SelectTokenOptions stochastic_opts;
    stochastic_opts.temperature = 1.0f;
    int counts[3] = {0, 0, 0};
    for (int i = 0; i < 2000; ++i) {
        counts[tinymind::select_token(logits, stochastic_opts, rng)]++;
    }
    assert(counts[1] > counts[0] + counts[2]);
    printf("test_sampler: OK\n");
}

static void test_kv_cache() {
    tinymind::KVCache cache(/*num_layers=*/2, /*max_seq_len=*/4, /*num_kv_heads=*/2, /*head_dim=*/3);
    assert(cache.current_length() == 0);

    std::vector<float> key(6, 1.0f), value(6, 2.0f);
    assert(cache.append(0, key.data(), value.data()));
    assert(cache.current_length() == 0);  // only advances after the LAST layer for a position
    assert(cache.append(1, key.data(), value.data()));
    assert(cache.current_length() == 1);

    const float* stored_key = cache.key_at(0, 0);
    assert(stored_key[0] == 1.0f);

    cache.reset();
    assert(cache.current_length() == 0);
    printf("test_kv_cache: OK\n");
}

static void test_tokenizer() {
    tinymind::ByteTokenizer tok;
    auto ids = tok.encode("hi", /*add_bos=*/true, /*add_eos=*/true);
    assert(ids.size() == 4);  // BOS + 'h' + 'i' + EOS
    assert(ids.front() == tinymind::ByteTokenizer::kBos);
    assert(ids.back() == tinymind::ByteTokenizer::kEos);

    std::string decoded = tok.decode(ids);
    assert(decoded == "hi");
    printf("test_tokenizer: OK\n");
}

static void test_c_abi_honest_failure() {
    tm_context_t* ctx = tm_create();
    assert(ctx != nullptr);

    int rc = tm_load_model(ctx, "/nonexistent/model.tm");
    assert(rc != TM_OK);  // must not claim success — see runtime.cpp

    tm_generation_config_t config{};
    config.max_new_tokens = 16;
    char buffer[256];
    rc = tm_generate(ctx, "hello", &config, buffer, sizeof(buffer));
    assert(rc == TM_ERROR_NOT_LOADED);  // no model was successfully loaded above

    assert(tm_reset(ctx) == TM_OK);

    tm_free(ctx);
    tm_free(nullptr);  // must be a safe no-op
    printf("test_c_abi_honest_failure: OK\n");
}

int main() {
    test_tensor();
    test_sampler();
    test_kv_cache();
    test_tokenizer();
    test_c_abi_honest_failure();
    printf("ALL NATIVE TESTS PASSED\n");
    return 0;
}
