// tensor.cpp — see tensor.h for the real implementation (kept header-only
// since Tensor's methods are small enough that the usual header/impl split
// buys nothing but an extra file to keep in sync). This translation unit
// exists so tensor.h is verified to compile standalone (see
// native/CMakeLists.txt) and to hold the one helper that doesn't belong in
// the header: a human-readable name for a DType, used only in error
// messages/logging, not on any hot path that would care about inlining.
#include "tensor.h"

namespace tinymind {

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return "float32";
        case DType::kFloat16: return "float16";
        case DType::kBFloat16: return "bfloat16";
        case DType::kInt8: return "int8";
        case DType::kInt4: return "int4";
        case DType::kInt3: return "int3";
        case DType::kInt2: return "int2";
        case DType::kInt32: return "int32";
        case DType::kUInt8: return "uint8";
    }
    return "unknown";
}

}  // namespace tinymind
