add_library(libquant INTERFACE)
target_include_directories(libquant INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(libquant INTERFACE cxx_std_20)

if(QMP_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(QMP_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
