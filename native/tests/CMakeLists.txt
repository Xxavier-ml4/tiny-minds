# See native/CMakeLists.txt's top comment: not verified with an actual
# cmake invocation in this delivery's sandbox (no cmake installed here) —
# verified instead by compiling and running test_native.cpp's sources
# directly with g++ (see STATUS.md for the exact command).
add_executable(tinymind_native_tests test_native.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/kv_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/tokenizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/runtime.cpp
)
target_include_directories(tinymind_native_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)
add_test(NAME tinymind_native_tests COMMAND tinymind_native_tests)
