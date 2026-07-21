add_executable(project04_tests test_dmd_tda.cpp test_mapper.cpp)
target_include_directories(project04_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(project04_tests PRIVATE libquant Catch2::Catch2WithMain)
qmp_set_warnings(project04_tests)

include(Catch)
catch_discover_tests(project04_tests)
