/*
 * tinymind.h — the public C ABI for the TinyMind native inference engine.
 *
 * STATUS: interface only. No file under native/src in this delivery
 * implements these functions beyond a stub that reports "not implemented"
 * — see STATUS.md and each .cpp file's own header comment. This header is
 * still real, in the sense that it is the actual contract
 * tinymind/runtime/native.py is written against (see that file's
 * docstring) and that the native/src sources compile against, per
 * tinymind/runtime/native.py's module-level reasoning about *why* a real
 * ABI is worth nailing down before there is an engine behind it: a stable
 * boundary is exactly the kind of thing that is expensive to change later
 * and cheap to get right now, independent of whether the implementation
 * exists yet.
 *
 * Design notes (engineering brief section 16):
 *   - Opaque handle (tm_context_t*), not a struct definition exposed
 *     across the boundary — internals can change without an ABI break.
 *   - Every function returns an int status code; 0 = success, negative =
 *     error. No exceptions cross this boundary (this is a C ABI).
 *   - No function here allocates memory the caller must free with a
 *     *different* library's allocator — tm_free_string() pairs with every
 *     function that hands back a heap-allocated C string.
 *   - This header intentionally does not expose model internals
 *     (architecture, tensor layout) across the ABI — brief section 16:
 *     "Do not expose implementation internals through the public ABI."
 */
#ifndef TINYMIND_H
#define TINYMIND_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* Opaque context: one per session, per docs/architecture/tinymind-design.md
 * section 10 and tinymind/runtime/engine.py's multi-session design — unlike
 * Needle's single process-global engine handle (needle-analysis.md section
 * 17), nothing about this ABI implies there can only be one. */
typedef struct tm_context tm_context_t;

/* Status codes. Never rely on the numeric value beyond zero/negative;
 * always compare against these names. */
#define TM_OK                       0
#define TM_ERROR_INVALID_ARGUMENT  -1
#define TM_ERROR_MODEL_NOT_FOUND   -2
#define TM_ERROR_MODEL_CORRUPT     -3
#define TM_ERROR_UNSUPPORTED_VERSION -4
#define TM_ERROR_OUT_OF_MEMORY     -5
#define TM_ERROR_BUFFER_TOO_SMALL  -6
#define TM_ERROR_GENERATION_FAILED -7
#define TM_ERROR_NOT_LOADED        -8
#define TM_ERROR_INTERNAL          -9

/* Create a new, empty context. Does not load a model — see tm_load_model().
 * Returns NULL on allocation failure. Each context is independent; nothing
 * about creating one affects any other context (see this header's design
 * notes above on multi-session support). */
tm_context_t* tm_create(void);

/* Load a .tm model file (tinymind/runtime/format.py is the Python-side
 * reader/writer for this same format; this function is the native-side
 * counterpart, not yet implemented — see native/src/model.cpp) into
 * `ctx`. `path` is a NUL-terminated UTF-8 path.
 * Returns TM_OK, or TM_ERROR_MODEL_NOT_FOUND / TM_ERROR_MODEL_CORRUPT /
 * TM_ERROR_UNSUPPORTED_VERSION on failure — mirroring the specific
 * exception subclasses tinymind.runtime.format raises on the Python side,
 * deliberately, so an error surfaced here maps cleanly to one of those on
 * the Python binding rather than collapsing into one generic failure. */
int tm_load_model(tm_context_t* ctx, const char* path);

/* Generation parameters. Mirrors tinymind.runtime.generation.GenerationConfig
 * field-for-field — see that module for the Python-side equivalent and its
 * validation rules, which a native caller should apply identically. */
typedef struct {
    int max_new_tokens;
    float temperature;
    int top_k;       /* <= 0 means "unset" */
    float top_p;      /* <= 0 means "unset" */
    uint64_t seed;
} tm_generation_config_t;

/* Run generation for `prompt` (NUL-terminated UTF-8) against the model
 * already loaded into `ctx`. Writes a NUL-terminated UTF-8 JSON response
 * envelope (matching the shape tinymind.model.backend.GenerationResult
 * serializes to) into `out_buffer`, up to `out_buffer_size` bytes
 * including the terminating NUL.
 * Returns TM_OK on success, TM_ERROR_NOT_LOADED if no model is loaded,
 * TM_ERROR_BUFFER_TOO_SMALL if `out_buffer` is too small (the caller
 * should retry with a larger buffer; this function never writes partial,
 * unterminated output on that path), or TM_ERROR_GENERATION_FAILED. */
int tm_generate(tm_context_t* ctx, const char* prompt,
                const tm_generation_config_t* config,
                char* out_buffer, size_t out_buffer_size);

/* Clear per-session state (KV cache, conversation) while keeping the
 * loaded model weights in place. Cheap; safe to call between unrelated
 * requests on the same context. */
int tm_reset(tm_context_t* ctx);

/* Compute an embedding for `text` (NUL-terminated UTF-8) using the model
 * loaded into `ctx`. Writes `out_dim` floats into `out_embedding`, which
 * must have room for at least `out_embedding_capacity` floats; returns
 * TM_ERROR_BUFFER_TOO_SMALL if the model's embedding dimension exceeds
 * `out_embedding_capacity`, writing the required size into `*out_dim`
 * regardless so the caller can reallocate and retry. */
int tm_embed(tm_context_t* ctx, const char* text,
            float* out_embedding, size_t out_embedding_capacity, size_t* out_dim);

/* Release `ctx` and every resource it owns (model weights, KV cache,
 * embedding tables). `ctx` is invalid after this call; using it again is
 * undefined behavior, same as any C ABI free() function. Safe to call with
 * NULL (a no-op), matching the free()/delete convention callers expect. */
void tm_free(tm_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif /* TINYMIND_H */
