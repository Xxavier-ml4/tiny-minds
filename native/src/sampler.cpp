// sampler.cpp — see sampler.h for the real implementation (header-only for
// the same reason as tensor.h — see tensor.cpp's comment). This
// translation unit verifies sampler.h compiles standalone and holds a
// small seeded-RNG factory that doesn't belong inline in the header (it's
// a convenience constructor, not part of the sampling algorithm itself).
#include "sampler.h"

namespace tinymind {

std::mt19937 make_rng(uint64_t seed) {
    return std::mt19937(static_cast<std::mt19937::result_type>(seed));
}

}  // namespace tinymind
