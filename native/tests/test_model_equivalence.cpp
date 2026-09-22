// test_model_equivalence.cpp — loads a real `.tm` model and runs a
// forward pass, writing the resulting logits to a binary file for
// comparison against the Python reference implementation.
//
// This program does not embed its own "expected" values — it is the
// native half of a cross-language check; the Python half
// (tests/model/test_native_equivalence.py) trains a model, exports it,
// computes its own logits on the same input, runs this program (compiling
// it with g++ first, since no cmake is available in this delivery's
// sandbox — see STATUS.md), and compares the two. See that Python test
// for the actual pass/fail assertion; this program only ever reports "I
// loaded the file and computed logits" or a specific load/inference
// error — it has no notion of whether the numbers are "right."
//
// Usage: test_model_equivalence <model.tm> <output_logits.bin> <token_id>...
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "../src/model.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.tm> <output_logits.bin> <token_id>...\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1];
    std::string output_path = argv[2];
    std::vector<int32_t> input_ids;
    for (int i = 3; i < argc; ++i) {
        input_ids.push_back(std::atoi(argv[i]));
    }

    tinymind::Model model;
    try {
        model = tinymind::Model::load(model_path);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "failed to load %s: %s\n", model_path.c_str(), exc.what());
        return 1;
    }

    std::vector<float> logits;
    try {
        logits = model.forward(input_ids);
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "forward() failed: %s\n", exc.what());
        return 1;
    }

    std::ofstream out(output_path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "could not open %s for writing\n", output_path.c_str());
        return 1;
    }
    out.write(reinterpret_cast<const char*>(logits.data()),
             static_cast<std::streamsize>(logits.size() * sizeof(float)));

    std::fprintf(stderr, "loaded %s (hidden_size=%d, num_layers=%d, %zu params); wrote [%zu, %d] logits to %s\n",
                model_path.c_str(), model.geometry().hidden_size, model.geometry().num_layers,
                model.parameter_count(), input_ids.size(), model.geometry().vocab_size, output_path.c_str());
    return 0;
}
