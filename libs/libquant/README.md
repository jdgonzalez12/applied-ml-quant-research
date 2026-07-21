# Project 1 — `libquant`: Numerical Linear Algebra From Scratch

A from-scratch, header-only C++20 dense linear algebra library implementing
the core algorithms of Golub & Van Loan's *Matrix Computations*: LU with
partial pivoting, Cholesky, Householder QR, a symmetric eigensolver, a full
Golub–Reinsch SVD, SVD-based ridge regression, and both exact and estimated
condition numbers — plus a general (complex) eigensolver for small
matrices. This is the shared foundation every other project in this
portfolio links against; nothing here calls into Eigen, BLAS, or LAPACK.
Eigen is used **only** as a correctness/performance oracle in the test and
benchmark suites.

## 1. What's implemented, and why each algorithm was chosen

| Header | Algorithm | Reference |
|---|---|---|
| [`matrix.hpp`](include/libquant/matrix.hpp) | Dense `Matrix<T>`, column-major contiguous storage | — |
| [`lu.hpp`](include/libquant/lu.hpp) | $PA=LU$, partial pivoting, rank-1-update elimination | Golub & Van Loan §3.2, 3.4 |
| [`cholesky.hpp`](include/libquant/cholesky.hpp) | $A=LL^\top$ for SPD $A$, left-looking (gaxpy) form | Golub & Van Loan §4.2 |
| [`qr.hpp`](include/libquant/qr.hpp) | Householder QR, explicit $Q$ | Golub & Van Loan §5.2 |
| [`symmetric_eigen.hpp`](include/libquant/symmetric_eigen.hpp) | Householder tridiagonalization + implicit-shift QL | Golub & Van Loan §8.3 |
| [`svd.hpp`](include/libquant/svd.hpp) | Golub–Kahan bidiagonalization + implicit-shift bidiagonal QR | Golub & Van Loan §8.6 |
| [`general_eigen.hpp`](include/libquant/general_eigen.hpp) | Faddeev–LeVerrier + Durand–Kerner (small matrices) | Golub & Van Loan §7 (context); see note below |
| [`least_squares.hpp`](include/libquant/least_squares.hpp) | QR-based OLS; SVD-based ridge closed form | Golub & Van Loan §5.5; ESL §3.4 |
| [`condition.hpp`](include/libquant/condition.hpp) | Exact $\kappa_2$ (SVD); Hager's $\kappa_1$ estimator | Golub & Van Loan §2.7, 3.5.4 |

## 2. Mathematical derivations

### 2.1 $PA=LU$ (partial pivoting)

Gaussian elimination expresses $A = LU$ (unit lower-triangular $L$,
upper-triangular $U$) by eliminating column $k$ below the diagonal at each
step, storing the multiplier $l_{ik} = a_{ik}/a_{kk}$. Without pivoting,
this fails whenever a diagonal entry is exactly zero, and is numerically
unstable whenever it is merely small (multipliers blow up, amplifying
rounding error). Partial pivoting — swapping in the largest-magnitude entry
in the current column before eliminating — keeps every multiplier
$|l_{ik}| \le 1$, which bounds the growth factor and is what makes the
algorithm backward stable in practice; the row swaps are tracked as a
permutation $P$ rather than materialized, giving $PA = LU$. Solving $Ax=b$
is then forward substitution ($Ly = Pb$) followed by back substitution
($Ux=y$), each $O(n^2)$.

### 2.2 Cholesky ($A = LL^\top$)

For symmetric positive-definite $A$, no pivoting is needed and the
factorization can be computed in roughly half the flops of LU ($n^3/3$ vs.
$2n^3/3$) because $L$'s $j$-th column is fully determined by $A$'s $j$-th
column and the previously computed columns of $L$: $\ell_{jj} =
\sqrt{a_{jj} - \sum_{k<j}\ell_{jk}^2}$, $\ell_{ij} = (a_{ij} -
\sum_{k<j}\ell_{ik}\ell_{jk})/\ell_{jj}$ for $i>j$.

### 2.3 Householder QR

A Householder reflector $H = I - 2vv^\top$ ($\|v\|=1$) is an orthogonal
matrix that reflects any vector across the hyperplane orthogonal to $v$.
Choosing $v$ so that $H$ maps column $k$'s subvector to a multiple of
$e_1$ zeros out everything below the diagonal in one step; applying $n$
such reflectors triangularizes $A$ into $R$ while accumulating $Q = H_1
H_2 \cdots H_n$. This is preferred over solving the normal equations
$A^\top A x = A^\top b$ for least squares precisely because it never forms
$A^\top A$: the normal-equations approach squares the condition number,
$\kappa(A^\top A) = \kappa(A)^2$, while QR's error sensitivity stays
governed by $\kappa(A)$ itself.

### 2.4 Symmetric eigenproblem

Householder tridiagonalization first reduces symmetric $A$ to tridiagonal
form $T = Q^\top A Q$ (identical reflector idea, applied from both sides).
The implicit-shift QL algorithm then iterates $T \to T'$ via orthogonal
similarity transforms that preserve eigenvalues while driving off-diagonal
entries toward zero, using a Wilkinson shift for cubic local convergence;
eigenvectors are accumulated into $Q$ throughout. This is exactly the
computation Project 2's PCA/RMT pipeline needs: eigendecomposing a
correlation matrix.

### 2.5 Golub–Reinsch SVD

$A = U\Sigma V^\top$ is computed in two phases: Householder bidiagonalization
reduces $A$ to upper-bidiagonal form (the same reflector technique as QR,
applied alternately from the left and right), then an implicit-shift QR
sweep on the bidiagonal form (the "Golub–Kahan SVD step") drives it to
diagonal, accumulating $U$ and $V$ via Givens rotations. This is the
classical Golub–Reinsch/LINPACK `dsvdc` algorithm — deliberately *not* the
simpler route of eigendecomposing $A^\top A$, because that squares
$\kappa(A)$ exactly as the normal equations do in §2.3; bidiagonalizing
$A$ directly avoids that.

### 2.6 General eigensolver — a deliberate scope decision

A full real-Schur Hessenberg-QR eigensolver (handling arbitrary
non-symmetric matrices, with explicit $2\times2$ block deflation for
complex-conjugate pairs) is significantly more delicate than any of the
above. Since this library's only consumer of general eigenvalues is
Project 4's DMD, whose reduced operator is always small ($r \lesssim 10$,
rank-truncated), a lighter-weight approach suffices: Faddeev–LeVerrier
computes the characteristic polynomial from traces of matrix powers (pure
matrix multiplication, no pivoting), and Durand–Kerner's simultaneous
iteration finds all roots directly in complex arithmetic — sidestepping
the need for explicit complex-pair deflation entirely, since the iteration
never assumes real intermediate values. This is a legitimate, fully
from-scratch algorithm, just not the one that would scale to large $n$; a
Hessenberg-QR implementation is noted as an explicit stretch goal, not a
gap.

### 2.7 Condition number: exact and estimated

$\kappa_2(A) = \sigma_{max}/\sigma_{min}$ falls directly out of the SVD.
Computing it exactly costs the full $O(n^3)$ SVD, though — often overkill
just to check whether a solve is trustworthy. Hager's estimator
(`condition_number_estimate`) gets $\kappa_1(A) = \|A\|_1\|A^{-1}\|_1$ to
within a small factor using only a handful of solves against one shared LU
factorization ($O(n^2)$ each), never forming $A^{-1}$ explicitly.

## 3. Correctness: every component cross-checked against Eigen

Every algorithm above has a closed-form or hand-derived test (known small
systems, matrices built as $Q\Lambda Q^\top$ with known $\Lambda$, the
Hilbert matrix as an ill-conditioning stress test) **and** a cross-check
against the equivalent Eigen routine on random matrices — `PartialPivLU`,
`LLT`, `HouseholderQR`, `SelfAdjointEigenSolver`, `JacobiSVD`,
`EigenSolver`. 519 assertions across 26 test cases, all passing
(`libquant_tests`).

## 4. Performance: what the benchmark caught

`benchmarks/bench_linear_algebra.cpp` times LU and Cholesky solves against
Eigen at several problem sizes. The first version of `LU` was **42x
slower than Eigen at n=256** — not an algorithmic error, but a memory
access pattern mismatch: the elimination loop's innermost index walked
across *columns* for a fixed row, which for this library's column-major
storage means striding `rows_` elements per step instead of reading
contiguous memory. Reordering to a rank-1 (outer-product) update — multipliers
computed once per column, then each trailing column updated in a single
contiguous sweep — fixed it:

| n | LU (before) | LU (after) | Eigen | Cholesky (before) | Cholesky (after) | Eigen |
|---|---|---|---|---|---|---|
| 256 | 78.9 ms | **2.16 ms** | 1.74 ms | 10.0 ms | **0.98 ms** | 0.71 ms |

At n=256, LU does $2n^3/3 \approx 11.2$ MFLOP; 2.16 ms implies ≈5.2 GFLOP/s
single-threaded, unvectorized — a believable ceiling for scalar C++
against Eigen's ≈6.4 GFLOP/s with its own internal vectorization. The
same fix applied to Cholesky closed a 13.6x gap to 1.4x. Regenerate with:

```
cmake --build build --target libquant_bench
./build/libs/libquant/benchmarks/libquant_bench
```

## 5. Design notes

`Matrix<T>` uses **column-major** contiguous storage specifically so that
column access — the operation every algorithm above performs in its inner
loop — is a unit-stride sweep. This single decision, and getting the loop
order to actually respect it (§4), was the dominant factor in closing the
gap to a production BLAS-backed library, well before any SIMD or
multi-threading would even be relevant.

## 6. Honesty about limits

The SVD implementation uses standard tolerance-based deflation and has not
been hardened against pathological ill-conditioning at production scale —
stated here as an engineering note, not swept under the rug. The general
eigensolver is scoped to small matrices by design (§2.6). Neither
limitation affects any result reported elsewhere in this portfolio, since
every consumer's matrices are small (≤ a few hundred rows) or explicitly
rank-truncated.
