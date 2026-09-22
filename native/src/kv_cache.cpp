// kv_cache.cpp — see kv_cache.h for the real implementation (header-only;
// see tensor.cpp's comment for why). This translation unit verifies
// kv_cache.h compiles standalone and holds a small helper — the memory
// footprint of a given configuration in bytes — used for the mobile RAM
// accounting the engineering brief's benchmarking sections (26-27, 50)
// call for, which doesn't belong as a KVCache instance method since it
// doesn't need an instance to compute.
#include "kv_cache.h"

namespace tinymind {

size_t kv_cache_footprint_bytes(int num_layers, int max_seq_len, int num_kv_heads, int head_dim) {
    size_t per_layer = static_cast<size_t>(max_seq_len) * num_kv_heads * head_dim;
    size_t floats_total = static_cast<size_t>(num_layers) * per_layer * 2;  // keys + values
    return floats_total * sizeof(float);
}

}  // namespace tinymind
