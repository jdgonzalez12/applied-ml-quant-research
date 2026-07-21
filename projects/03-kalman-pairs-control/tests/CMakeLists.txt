add_executable(project03_tests test_kalman_adf.cpp)
target_include_directories(project03_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(project03_tests PRIVATE libquant Catch2::Catch2WithMain)
qmp_set_warnings(project03_tests)

include(Catch)
catch_discover_tests(project03_tests)
