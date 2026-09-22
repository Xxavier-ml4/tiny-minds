// tensor.h — a minimal, dtype-tagged tensor buffer.
//
// STATUS: real. Managing a raw byte buffer with a shape and a dtype tag is
// not model-specific — it needs no trained weights and no architecture to
// be correct, so unlike model.cpp/runtime.cpp it is not a stub. This is
// deliberately small: no broadcasting, no views, no operator overloading —
// just what tinymind::Tensor needs to exist as a well-defined bounds-
// checked unit of storage for the pieces that *are* stubs to build on.
#ifndef TINYMIND_TENSOR_H
#define TINYMIND_TENSOR_H

#include <cstdint>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tinymind {

enum class DType : uint8_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
    kInt8 = 3,
    kInt4 = 4,
    kInt3 = 5,
    kInt2 = 6,
    kInt32 = 7,
    kUInt8 = 8,
};

// Bytes per element for the *storage* dtype. Sub-byte types (int4/int3/int2)
// are packed and are not addressable by element here — callers working with
// packed sub-byte tensors go through the raw byte buffer directly, which is
// why this returns 0 for those rather than pretending a fractional byte
// size is meaningful.
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return 4;
        case DType::kFloat16: return 2;
        case DType::kBFloat16: return 2;
        case DType::kInt8: return 1;
        case DType::kUInt8: return 1;
        case DType::kInt32: return 4;
        case DType::kInt4:
        case DType::kInt3:
        case DType::kInt2:
            return 0;  // packed; see comment above
    }
    return 0;
}

// Defined in tensor.cpp — a human-readable name for error messages/logging.
const char* dtype_name(DType dtype);

class Tensor {
public:
    Tensor(std::vector<int64_t> shape, DType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {
        int64_t count = 1;
        for (int64_t dim : shape_) {
            if (dim < 0) {
                throw std::invalid_argument("Tensor: negative dimension in shape");
            }
            count *= dim;
        }
        size_t elem_size = dtype_size(dtype_);
        // Packed sub-byte dtypes: caller is responsible for sizing the
        // buffer correctly (bits-per-element is not a whole byte count) —
        // this constructor only handles the whole-byte-element case.
        num_elements_ = static_cast<size_t>(count);
        data_.resize(elem_size > 0 ? num_elements_ * elem_size : 0);
    }

    const std::vector<int64_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t num_elements() const { return num_elements_; }
    size_t byte_size() const { return data_.size(); }

    uint8_t* data() { return data_.data(); }
    const uint8_t* data() const { return data_.data(); }

    // Bounds-checked float accessor for kFloat32 tensors — the common case
    // for anything that will eventually reach a sampler (native/src/sampler.*)
    // or a KV cache (native/src/kv_cache.*).
    float& at_f32(size_t index) {
        if (dtype_ != DType::kFloat32) {
            throw std::logic_error("Tensor::at_f32 called on a non-float32 tensor");
        }
        if (index >= num_elements_) {
            throw std::out_of_range("Tensor::at_f32 index out of range");
        }
        return reinterpret_cast<float*>(data_.data())[index];
    }

private:
    std::vector<int64_t> shape_;
    DType dtype_;
    size_t num_elements_ = 0;
    std::vector<uint8_t> data_;
};

}  // namespace tinymind

#endif  // TINYMIND_TENSOR_H
