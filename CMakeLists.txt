cmake_minimum_required(VERSION 3.21)
project(quant_ml_portfolio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(QMP_BUILD_TESTS "Build unit tests (requires Catch2 + Eigen via FetchContent)" ON)
option(QMP_BUILD_BENCHMARKS "Build latency benchmarks (requires Eigen via FetchContent)" ON)
option(QMP_BUILD_PROJECTS "Build applied projects 02-04" ON)

include(cmake/CompilerWarnings.cmake)

# --- Third-party dependencies: test/benchmark oracles only. -----------------
# Eigen is a correctness/performance baseline for libquant's tests and
# benchmarks; it is never linked into libquant's own implementation.
if(QMP_BUILD_TESTS OR QMP_BUILD_BENCHMARKS)
    include(FetchContent)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

if(QMP_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.4
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    include(CTest)
    include(Catch)
    enable_testing()
endif()

add_subdirectory(libs/libquant)
add_subdirectory(libs/libquant_backtest)

if(QMP_BUILD_PROJECTS)
    add_subdirectory(projects/02-rmt-pca-stat-arb)
    add_subdirectory(projects/03-kalman-pairs-control)
    add_subdirectory(projects/04-regime-shootout)
endif()
