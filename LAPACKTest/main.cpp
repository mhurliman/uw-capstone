
#include <cstdlib>
#include <stdio.h>

#include "MatrixMath.h"
#include "Random.h"
#include <lapacke.h>

void test_dgeev()
{
    constexpr int N = 5;

    double A[N * N];
    RandomMatrix(A, N, Random<double>(5));

    PrintMatrix(A, N);

    double wr[N];
    double wi[N];

    LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N', N, A, N, wr, wi, NULL, 1, NULL, 1);

    PrintVector(wr, N);
    PrintVector(wi, N);
}

void test_zheev()
{
    constexpr int N = 5;

    c64 A[N * N];
    RandomMatrixReal(A, N, Random<double>(5));

    int lda = N;
    double W[N];

    LAPACKE_zheev(LAPACK_COL_MAJOR, 'N', 'U', N, A, lda, W);
    PrintVector(W, N);
}

int main(int argc, const char** argv)
{
    test_dgeev();

    return 0;
}
