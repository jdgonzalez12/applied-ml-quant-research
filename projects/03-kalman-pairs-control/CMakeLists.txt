add_executable(project03_kalman_pairs src/main.cpp)
target_include_directories(project03_kalman_pairs PRIVATE include)
target_link_libraries(project03_kalman_pairs PRIVATE libquant libquant_backtest)
target_compile_definitions(project03_kalman_pairs PRIVATE
    QMP_DATA_DIR="${CMAKE_SOURCE_DIR}/data/raw/equities"
    QMP_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
)
qmp_set_warnings(project03_kalman_pairs)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(project03_kalman_pairs PRIVATE -O3)
endif()

if(QMP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
