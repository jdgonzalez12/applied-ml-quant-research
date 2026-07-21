add_executable(project02_rmt_pca src/main.cpp)
target_include_directories(project02_rmt_pca PRIVATE include)
target_link_libraries(project02_rmt_pca PRIVATE libquant libquant_backtest)
target_compile_definitions(project02_rmt_pca PRIVATE
    QMP_DATA_DIR="${CMAKE_SOURCE_DIR}/data/raw/equities"
    QMP_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
)
qmp_set_warnings(project02_rmt_pca)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(project02_rmt_pca PRIVATE -O3)
endif()

if(QMP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
