cmake_minimum_required(VERSION 3.10)
project(CUDA_LEARN LANGUAGES CUDA CXX)
# enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

add_subdirectory(first_try)
add_subdirectory(second_try)
