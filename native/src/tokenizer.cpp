// tokenizer.cpp — see tokenizer.h for the real implementation (header-only;
// see tensor.cpp's comment for why). This translation unit verifies
// tokenizer.h compiles standalone and defines the out-of-line static
// constexpr members required by pre-C++17 ODR rules for any translation
// unit that odr-uses them (harmless, and correct, under C++17 too).
#include "tokenizer.h"

namespace tinymind {

constexpr int32_t ByteTokenizer::kPad;
constexpr int32_t ByteTokenizer::kBos;
constexpr int32_t ByteTokenizer::kEos;
constexpr int32_t ByteTokenizer::kUnk;
constexpr int32_t ByteTokenizer::kNumSpecials;
constexpr int32_t ByteTokenizer::kVocabSize;

}  // namespace tinymind
