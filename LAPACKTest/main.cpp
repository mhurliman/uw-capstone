
#include <cstdlib>
#include <stdio.h>

#include "MatrixMath.h"
#include "Random.h"

#include <lapacke.h>
#include <cblas.h>


void pdgemm_(char* TRANSA, char* TRANSB,
            int* M, int* N, int* K,
            double* ALPHA,
            double* A, int* IA, int* JA, int* DESCA,
            double* B, int* IB, int* JB, int* DESCB,
            double* BETA,
            double* C, int* IC, int* JC, int* DESCC );

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
    using T = c64;
    constexpr int N = 2;

    auto arena = MemoryArenaPool::GetArena<T>(2 * N * N + 3  * N);
    auto A = arena.Allocate<T>(N * N);
    auto B = arena.Allocate<T>(N * N);
    auto W = arena.Allocate<double>(N);
    auto R = arena.Allocate<T>(N);
    auto V = arena.Allocate<T>(N);


    Random<double> r(5);
    RandomHermitian(A, N, Random<double>(5));
    Copy(B, A, N);
    PrintMatrix(A, N);

    LAPACKE_zheev(LAPACK_COL_MAJOR, 'V', 'U', N, A, N, W);

    PrintMatrix(A, N);
    PrintVector(W, N);

    c64 alpha = 1.0;
    c64 beta = 0.0;

    for (int i = 0; i < N; ++i)
    {
        cblas_zgemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans, 
            N, 1, N,
            &alpha, B, N, A + (N * i), N, &beta, R, N
        );
        Divide(V, R, N, W[i]);
        PrintVector(V, N);
    }
}

int main(int argc, const char** argv)
{
    test_zheev();

    return 0;
}
