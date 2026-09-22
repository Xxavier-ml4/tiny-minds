# TinyMind native engine build.
#
# STATUS: this builds a real shared library today — tensor.h, sampler.h,
# kv_cache.h, tokenizer.h, and runtime.cpp's context lifecycle are real
# code (see each file's header comment) — but the resulting
# libtinymind.{so,dylib,dll} cannot load a model or generate text yet,
# because model.cpp's Model::load()/forward() throw unconditionally (see
# that file and STATUS.md). CI (.github/workflows/tests.yml) compiles this
# target so a build regression is caught even though there's no trained
# model to run through it yet.
#
# This delivery's sandbox does not have cmake installed, so this file has
# been written carefully against documented CMake behavior but has only
# been checked by compiling its listed sources directly with g++ (see
# STATUS.md) — run an actual `cmake -B build && cmake --build build` as
# the first thing to verify once cmake is available.

cmake_minimum_required(VERSION 3.16)
project(tinymind_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(tinymind SHARED
    src/runtime.cpp
    src/model.cpp
    src/tokenizer.cpp
    src/sampler.cpp
    src/kv_cache.cpp
    src/tensor.cpp
)

target_include_directories(tinymind
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(tinymind PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra>
)

# Android cross-compilation: pass
#   -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake
#   -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24
# per android/README.md's ABI matrix. Not exercised in this delivery (no
# Android build, no NDK available in this sandbox) — see that file.

enable_testing()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()
