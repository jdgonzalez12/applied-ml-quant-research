#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "libquant/matrix.hpp"

using libquant::Matrix;
using Catch::Approx;

TEST_CASE("Matrix basic construction and indexing", "[matrix]") {
    Matrix<double> A(2, 3, 0.0);
    REQUIRE(A.rows() == 2);
    REQUIRE(A.cols() == 3);
    A(0, 0) = 1.0;
    A(1, 2) = 5.0;
    REQUIRE(A(0, 0) == Approx(1.0));
    REQUIRE(A(1, 2) == Approx(5.0));
    REQUIRE(A(0, 1) == Approx(0.0));
}

TEST_CASE("Matrix identity", "[matrix]") {
    auto I = Matrix<double>::identity(3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(I(i, j) == Approx(i == j ? 1.0 : 0.0));
}

TEST_CASE("Matrix multiplication matches hand-computed result", "[matrix]") {
    Matrix<double> A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;

    Matrix<double> B(2, 2);
    B(0, 0) = 5; B(0, 1) = 6;
    B(1, 0) = 7; B(1, 1) = 8;

    Matrix<double> C = A * B;
    REQUIRE(C(0, 0) == Approx(19.0));
    REQUIRE(C(0, 1) == Approx(22.0));
    REQUIRE(C(1, 0) == Approx(43.0));
    REQUIRE(C(1, 1) == Approx(50.0));
}

TEST_CASE("Matrix transpose is involutive", "[matrix]") {
    Matrix<double> A(2, 3);
    double v = 0.0;
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 2; ++i)
            A(i, j) = v++;

    Matrix<double> At = A.transpose();
    REQUIRE(At.rows() == 3);
    REQUIRE(At.cols() == 2);
    Matrix<double> Att = At.transpose();
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 2; ++i)
            REQUIRE(Att(i, j) == Approx(A(i, j)));
}

TEST_CASE("dot and norm2 on column vectors", "[matrix]") {
    Matrix<double> v(3, 1);
    v(0, 0) = 3; v(1, 0) = 4; v(2, 0) = 0;
    REQUIRE(libquant::norm2(v) == Approx(5.0));
    REQUIRE(libquant::dot(v, v) == Approx(25.0));
}
