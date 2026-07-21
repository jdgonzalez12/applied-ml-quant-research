add_executable(project04_regime_shootout src/main.cpp)
target_include_directories(project04_regime_shootout PRIVATE include)
target_link_libraries(project04_regime_shootout PRIVATE libquant)
target_compile_definitions(project04_regime_shootout PRIVATE
    QMP_DATA_DIR="${CMAKE_SOURCE_DIR}/data/raw/equities"
    QMP_MACRO_DIR="${CMAKE_SOURCE_DIR}/data/raw/macro"
    QMP_RESULTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/results"
)
qmp_set_warnings(project04_regime_shootout)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(project04_regime_shootout PRIVATE -O3)
endif()

if(QMP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
