// crc32.h — CRC-32 (IEEE 802.3 / zlib-compatible), needed to verify a
// `.tm` file's per-tensor checksums the same way
// tinymind.runtime.format.ModelFile.read_tensor() does on the Python
// side (Python's zlib.crc32 implements this exact, standard algorithm).
//
// STATUS: real. A completely standard, well-documented bit-reflected CRC
// (polynomial 0xEDB88320, initial state 0xFFFFFFFF, final XOR
// 0xFFFFFFFF) — not something specific to this project, just implemented
// locally since no zlib binding is being pulled in for one function.
#ifndef TINYMIND_CRC32_H
#define TINYMIND_CRC32_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace tinymind {

namespace detail {

inline std::array<uint32_t, 256> make_crc32_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    return table;
}

inline const std::array<uint32_t, 256>& crc32_table() {
    static const std::array<uint32_t, 256> table = make_crc32_table();
    return table;
}

}  // namespace detail

inline uint32_t crc32(const uint8_t* data, size_t length) {
    const auto& table = detail::crc32_table();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < length; ++i) {
        c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

}  // namespace tinymind

#endif  // TINYMIND_CRC32_H
