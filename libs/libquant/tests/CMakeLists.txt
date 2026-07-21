add_executable(libquant_tests
    test_matrix.cpp
    test_lu.cpp
    test_cholesky.cpp
    test_qr.cpp
    test_symmetric_eigen.cpp
    test_svd.cpp
    test_least_squares.cpp
    test_condition.cpp
)
target_link_libraries(libquant_tests PRIVATE libquant Catch2::Catch2WithMain Eigen3::Eigen)
qmp_set_warnings(libquant_tests)

include(Catch)
catch_discover_tests(libquant_tests)
