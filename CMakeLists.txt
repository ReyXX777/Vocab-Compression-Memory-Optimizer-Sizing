
cmake_minimum_required(VERSION 3.20)
project(vocab_sizing VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Need TBB for parallel execution policies on some toolchains
find_package(TBB QUIET)

include_directories(include)

file(GLOB_RECURSE LIB_SOURCES
    src/common/*.cpp
    src/corpus/*.cpp
    src/tokenizer/*.cpp
    src/metrics/*.cpp
    src/optimization/*.cpp
    src/reporting/*.cpp
)

add_library(vocab_lib STATIC ${LIB_SOURCES})
target_include_directories(vocab_lib PUBLIC include)
if(TBB_FOUND)
  target_link_libraries(vocab_lib PUBLIC TBB::tbb)
endif()

add_executable(vocab_sizing src/main.cpp)
target_link_libraries(vocab_sizing PRIVATE vocab_lib)
if(TBB_FOUND)
  target_link_libraries(vocab_sizing PRIVATE TBB::tbb)
endif()

# Tests
add_executable(test_bpe tests/test_bpe.cpp)
target_link_libraries(test_bpe PRIVATE vocab_lib)
if(TBB_FOUND)
  target_link_libraries(test_bpe PRIVATE TBB::tbb)
endif()

add_executable(test_compression tests/test_compression.cpp)
target_link_libraries(test_compression PRIVATE vocab_lib)
if(TBB_FOUND)
  target_link_libraries(test_compression PRIVATE TBB::tbb)
endif()

# CTest
enable_testing()
add_test(NAME BPE COMMAND test_bpe)
add_test(NAME Compression COMMAND test_compression)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
