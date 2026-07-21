add_executable(project02_tests test_mp_ou.cpp)
target_include_directories(project02_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(project02_tests PRIVATE libquant Catch2::Catch2WithMain)
qmp_set_warnings(project02_tests)

include(Catch)
catch_discover_tests(project02_tests)
