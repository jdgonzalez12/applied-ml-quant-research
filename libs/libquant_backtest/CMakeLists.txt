# Shared backtest metrics / walk-forward CV library, used identically by
# Projects 2-4 so cross-project comparison is apples-to-apples.
add_library(libquant_backtest INTERFACE)
target_include_directories(libquant_backtest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(libquant_backtest INTERFACE cxx_std_20)

if(QMP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
