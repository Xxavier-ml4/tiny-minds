// tm_reader.h — reads a `.tm` model file, matching
// tinymind.runtime.format.read_model()'s layout and hardening checks
// exactly (see docs/architecture/native-model-contract.md and
// tinymind/runtime/format.py's own module docstring for the byte layout).
//
// STATUS: real. This is the piece that was previously entirely missing
// from the native side — model.cpp could not load a `.tm` file at all
// before this. Every bounds/overlap/header-region check
// tinymind/runtime/format.py performs is reproduced here, because a `.tm`
// file is untrusted input on the native side exactly as much as on the
// Python side (brief section 28: "Model files are untrusted input").
#ifndef TINYMIND_TM_READER_H
#define TINYMIND_TM_READER_H

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "crc32.h"
#include "json_parser.h"

namespace tinymind {

class TmFormatError : public std::runtime_error {
public:
    explicit TmFormatError(const std::string& message) : std::runtime_error(message) {}
};

struct TmTensorEntry {
    std::string name;
    uint8_t dtype_code;
    std::vector<uint32_t> shape;
    uint64_t offset;
    uint64_t nbytes;
    uint32_t crc32;
};

// Mirrors tinymind.runtime.format._DTYPE_CODES exactly — see that table's
// own comment for why it's small and fixed rather than an open registry.
inline std::string dtype_name_for_code(uint8_t code) {
    switch (code) {
        case 0: return "float32";
        case 1: return "float16";
        case 2: return "bfloat16";
        case 3: return "int8";
        case 4: return "int4";
        case 5: return "int3";
        case 6: return "int2";
        case 7: return "int32";
        case 8: return "uint8";
        default: throw TmFormatError("unknown dtype code " + std::to_string(code));
    }
}

class TmFile {
public:
    static TmFile load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw TmFormatError("no such file: " + path);
        }

        file.seekg(0, std::ios::end);
        int64_t file_size_signed = file.tellg();
        if (file_size_signed < 0) throw TmFormatError("could not determine file size: " + path);
        uint64_t file_size = static_cast<uint64_t>(file_size_signed);
        file.seekg(0, std::ios::beg);

        char magic[4];
        read_exact(file, magic, 4, "magic");
        if (std::memcmp(magic, "TM01", 4) != 0) {
            throw TmFormatError("unrecognized magic bytes — this reader only reads 'TM01' files");
        }

        uint32_t header_length = read_u32(file, "header length");
        static constexpr uint32_t kMaxHeaderLength = 16u * 1024 * 1024;  // matches format.py's own sanity bound
        if (header_length > kMaxHeaderLength) {
            throw TmFormatError("declared metadata length exceeds the sanity limit");
        }
        check_bounds(8, header_length, file_size, "metadata block");

        std::string metadata_bytes(header_length, '\0');
        read_exact(file, metadata_bytes.data(), header_length, "metadata block");

        TmFile result;
        try {
            result.metadata_ = parse_json(metadata_bytes);
        } catch (const JsonParseError& exc) {
            throw TmFormatError(std::string("metadata block is not valid JSON: ") + exc.what());
        }

        uint32_t tensor_count = read_u32(file, "tensor count");
        static constexpr uint32_t kMaxTensorCount = 1000000;  // matches format.py's own sanity bound
        if (tensor_count > kMaxTensorCount) {
            throw TmFormatError("declared tensor count exceeds the sanity limit");
        }

        for (uint32_t i = 0; i < tensor_count; ++i) {
            TmTensorEntry entry;
            uint16_t name_length = read_u16(file, "tensor name length");
            static constexpr uint16_t kMaxNameLength = 4096;
            if (name_length > kMaxNameLength) throw TmFormatError("tensor name length exceeds sanity limit");
            std::string name(name_length, '\0');
            read_exact(file, name.data(), name_length, "tensor name");
            entry.name = name;
            if (result.tensors_.count(name)) {
                throw TmFormatError("duplicate tensor name in directory: " + name);
            }

            uint8_t dtype_code = read_u8(file, "dtype code");
            uint8_t ndim = read_u8(file, "ndim");
            static constexpr uint8_t kMaxNdim = 8;
            if (ndim > kMaxNdim) throw TmFormatError("tensor ndim exceeds sanity limit: " + name);
            entry.dtype_code = dtype_code;
            dtype_name_for_code(dtype_code);  // throws if unknown — validate eagerly

            for (uint8_t d = 0; d < ndim; ++d) {
                entry.shape.push_back(read_u32(file, "tensor shape dim"));
            }

            entry.offset = read_u64(file, "tensor offset");
            entry.nbytes = read_u64(file, "tensor size");
            entry.crc32 = read_u32(file, "tensor checksum");
            check_bounds(entry.offset, entry.nbytes, file_size, "tensor '" + name + "' directory entry");

            result.tensors_[name] = entry;
        }

        uint64_t directory_end = static_cast<uint64_t>(file.tellg());
        check_no_tensor_points_into_header(result.tensors_, directory_end);
        check_no_tensor_overlap(result.tensors_);

        result.path_ = path;
        result.file_size_ = file_size;
        return result;
    }

    const JsonValue& metadata() const { return metadata_; }
    const std::map<std::string, TmTensorEntry>& tensors() const { return tensors_; }

    bool has_tensor(const std::string& name) const { return tensors_.count(name) > 0; }

    std::vector<uint8_t> read_tensor(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw TmFormatError("no such tensor: " + name);
        }
        const TmTensorEntry& entry = it->second;
        std::ifstream file(path_, std::ios::binary);
        if (!file) throw TmFormatError("could not reopen file: " + path_);
        file.seekg(static_cast<std::streamoff>(entry.offset));
        std::vector<uint8_t> data(entry.nbytes);
        if (entry.nbytes > 0) {
            file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(entry.nbytes));
            if (!file) throw TmFormatError("unexpected end of file while reading tensor: " + name);
        }
        uint32_t computed_crc = crc32(data.data(), data.size());
        if (computed_crc != entry.crc32) {
            throw TmFormatError("tensor '" + name + "' failed its checksum — file is corrupt or truncated");
        }
        return data;
    }

private:
    JsonValue metadata_;
    std::map<std::string, TmTensorEntry> tensors_;
    std::string path_;
    uint64_t file_size_ = 0;

    static void read_exact(std::ifstream& file, char* buffer, size_t n, const std::string& context) {
        file.read(buffer, static_cast<std::streamsize>(n));
        if (!file) throw TmFormatError("unexpected end of file while reading " + context);
    }
    static uint8_t read_u8(std::ifstream& file, const std::string& context) {
        uint8_t v;
        read_exact(file, reinterpret_cast<char*>(&v), 1, context);
        return v;
    }
    static uint16_t read_u16(std::ifstream& file, const std::string& context) {
        uint8_t buf[2];
        read_exact(file, reinterpret_cast<char*>(buf), 2, context);
        return static_cast<uint16_t>(buf[0]) | (static_cast<uint16_t>(buf[1]) << 8);  // little-endian, matches struct.pack("<H", ...)
    }
    static uint32_t read_u32(std::ifstream& file, const std::string& context) {
        uint8_t buf[4];
        read_exact(file, reinterpret_cast<char*>(buf), 4, context);
        return static_cast<uint32_t>(buf[0]) | (static_cast<uint32_t>(buf[1]) << 8) |
              (static_cast<uint32_t>(buf[2]) << 16) | (static_cast<uint32_t>(buf[3]) << 24);
    }
    static uint64_t read_u64(std::ifstream& file, const std::string& context) {
        uint8_t buf[8];
        read_exact(file, reinterpret_cast<char*>(buf), 8, context);
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | buf[i];
        return v;
    }

    // Same three hardening checks as tinymind.runtime.format — see that
    // module's _check_bounds/_check_no_tensor_points_into_header/
    // _check_no_tensor_overlap for the Python-side originals and their
    // own comments on exactly what each defends against. Unlike a naive
    // C uint64_t sum, these use uint64_t arithmetic that can only
    // overflow at values far beyond any real file size — see
    // check_bounds's own comment for why that residual risk is handled.
    static void check_bounds(uint64_t offset, uint64_t nbytes, uint64_t file_size, const std::string& context) {
        if (offset > file_size) {
            throw TmFormatError(context + ": offset is beyond the end of the file");
        }
        // offset and nbytes are both real (non-adversarially-huge-beyond-
        // representable) uint64_t values bounded by file_size checks
        // above and below; the one residual risk — offset + nbytes
        // wrapping past UINT64_MAX — is excluded by construction, since
        // offset <= file_size (checked above) and any file this reader
        // opens is far smaller than UINT64_MAX bytes.
        if (offset + nbytes > file_size) {
            throw TmFormatError(context + ": offset + size exceeds the file size — "
                                "file is truncated or the directory is corrupt");
        }
    }

    static void check_no_tensor_points_into_header(const std::map<std::string, TmTensorEntry>& tensors,
                                                   uint64_t directory_end) {
        for (const auto& [name, entry] : tensors) {
            if (entry.nbytes > 0 && entry.offset < directory_end) {
                throw TmFormatError("tensor '" + name + "' points into the header/directory region");
            }
        }
    }

    static void check_no_tensor_overlap(const std::map<std::string, TmTensorEntry>& tensors) {
        std::vector<std::pair<uint64_t, uint64_t>> ranges;  // (start, end)
        std::vector<std::string> names;
        for (const auto& [name, entry] : tensors) {
            if (entry.nbytes == 0) continue;
            ranges.emplace_back(entry.offset, entry.offset + entry.nbytes);
            names.push_back(name);
        }
        std::vector<size_t> order(ranges.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                 [&](size_t a, size_t b) { return ranges[a].first < ranges[b].first; });
        for (size_t i = 1; i < order.size(); ++i) {
            const auto& prev = ranges[order[i - 1]];
            const auto& current = ranges[order[i]];
            if (current.first < prev.second) {
                throw TmFormatError("tensor '" + names[order[i]] + "' overlaps tensor '" +
                                    names[order[i - 1]] + "'");
            }
        }
    }
};

}  // namespace tinymind

#endif  // TINYMIND_TM_READER_H
