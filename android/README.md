# TinyMind on Android

STATUS: plan only. There is no JNI wrapper, no Gradle project, and no
compiled `.so` in this directory — see `STATUS.md` at the repository root.
`native/` doesn't have a working inference engine yet (see
`native/src/model.cpp`), so there's nothing for an Android build to wrap.
What's here is the deployment plan and, specifically, a documented pitfall
worth guarding against from day one of a real Android build.

## Target ABI matrix

| ABI | Priority |
|---|---|
| `arm64-v8a` | Primary |
| `armeabi-v7a` | Secondary |
| `x86_64` | Emulator/dev only |

Per the engineering brief section 32/33, the native runtime should not
require Python on Android — `libtinymind.so` (once `native/` is real) links
against the C ABI in `native/include/tinymind.h` directly from Kotlin via
JNI, with no Python interpreter in the deployed app.

## Known pitfall: Bionic vs. glibc

The engineering brief that scoped this project specifically flagged a
problem hit while field-testing Needle on Android: **a Linux ARM64 binary
is not an Android ARM64 binary.** Both report `aarch64`/`arm64` as their
CPU architecture, so it is easy to build a `linux-arm64` artifact, rename
or re-tag it as `android-arm64`, and have it load fine on a *generic* ARM64
Linux checker while still being wrong for a real device — Android uses
Bionic libc, not glibc, and the two are not ABI-compatible.

Concretely, the failure mode is: a CI matrix or release pipeline builds
`linux-arm64` and `android-arm64` as if they were the same target because
the compiler's `-march`/triple flags look similar, and nothing catches it
until the library fails to load (or, worse, loads and misbehaves) on an
actual phone. `tinymind/runtime/native.py`'s `platform_tag()` exists
specifically to make this distinction a first-class, checkable value rather
than something inferred ad hoc at build time — see that function's
docstring.

**Guard for this once there's a real build**: cross-compile explicitly with
the Android NDK's toolchain file (see `native/CMakeLists.txt`'s comment on
this) rather than a generic `aarch64-linux-gnu` cross-compiler, and add a CI
check that a produced `android-arm64` artifact is actually linked against
Bionic (e.g. `readelf -d` showing `libc.so`/`libdl.so` from the NDK sysroot,
not a glibc path) before it's published — never let "it loaded in an x86_64
Linux container" stand in for "it loaded on a Bionic ARM64 device."

## Offline-first

Per the brief section 34, once a model and the native library are installed
on-device, inference must not require network access. `tinymind model
download`/`install`/`verify` (CLI subcommands — see `tinymind/cli.py`;
network-dependent model fetching itself is not implemented in this
delivery, same status as everything else in this file) are the only
points where a network request should ever happen.

## What would come first

1. A real `native/src/model.cpp` (a working forward pass) and a real
   `.tm` file to load — see the root `STATUS.md`'s phase list. Nothing
   Android-specific is useful before this exists.
2. A minimal JNI wrapper (`android/jni/`) around `native/include/tinymind.h`
   — a thin pass-through, not new logic, since the C ABI is already
   designed not to leak implementation details (see that header's design
   notes).
3. A Kotlin wrapper (`android/kotlin/`) around the JNI layer, plus the NDK
   CMake toolchain wiring in `native/CMakeLists.txt`.
4. The Bionic-vs-glibc CI guard above, before the first `android-arm64`
   artifact is ever published anywhere.
