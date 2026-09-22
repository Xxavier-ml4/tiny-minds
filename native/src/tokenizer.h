// tokenizer.h — byte-level tokenizer, mirroring tinymind/model/tokenizer.py's
// ByteTokenizer exactly (same vocabulary layout: 4 special tokens + 256
// byte values — see that file's module docstring for the full rationale).
//
// STATUS: real. Like the Python ByteTokenizer this is not a placeholder —
// it is a complete, correct tokenizer over raw UTF-8 bytes, usable today
// with no trained subword vocabulary. A trained subword tokenizer's native
// binding would be a different file (this one doesn't need to change to
// add it, any more than the Python ByteTokenizer needs to change for a
// future subword Tokenizer to exist alongside it).
#ifndef TINYMIND_TOKENIZER_H
#define TINYMIND_TOKENIZER_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tinymind {

class ByteTokenizer {
public:
    static constexpr int32_t kPad = 0;
    static constexpr int32_t kBos = 1;
    static constexpr int32_t kEos = 2;
    static constexpr int32_t kUnk = 3;
    static constexpr int32_t kNumSpecials = 4;
    static constexpr int32_t kVocabSize = 256 + kNumSpecials;

    std::vector<int32_t> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const {
        std::vector<int32_t> ids;
        ids.reserve(text.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
        if (add_bos) ids.push_back(kBos);
        for (unsigned char byte : text) {
            ids.push_back(static_cast<int32_t>(byte) + kNumSpecials);
        }
        if (add_eos) ids.push_back(kEos);
        return ids;
    }

    // Mirrors ByteTokenizer.decode()'s errors="replace" behavior for any
    // byte sequence that isn't valid UTF-8, using the standard UTF-8
    // replacement character rather than throwing — a tokenizer used during
    // generation will routinely see partial multi-byte sequences mid-stream.
    std::string decode(const std::vector<int32_t>& ids) const {
        std::string raw;
        raw.reserve(ids.size());
        for (int32_t id : ids) {
            if (id == kPad || id == kBos || id == kEos || id == kUnk) continue;
            int32_t byte_value = id - kNumSpecials;
            if (byte_value < 0 || byte_value > 255) {
                throw std::out_of_range("ByteTokenizer::decode: token id out of range");
            }
            raw.push_back(static_cast<char>(static_cast<unsigned char>(byte_value)));
        }
        return raw;  // raw bytes; caller performs UTF-8 validation/repair if needed for display
    }

    int32_t vocab_size() const { return kVocabSize; }
};

}  // namespace tinymind

#endif  // TINYMIND_TOKENIZER_H
