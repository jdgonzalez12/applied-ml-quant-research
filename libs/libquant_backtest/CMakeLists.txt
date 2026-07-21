# Shared backtest metrics / walk-forward CV library (Milestone M5).
# Populated once Project 2 needs it; kept as an empty placeholder target for
# now so the top-level build configures cleanly during earlier milestones.
add_library(libquant_backtest INTERFACE)
target_include_directories(libquant_backtest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(libquant_backtest INTERFACE cxx_std_20)
