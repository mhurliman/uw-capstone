
#include <iostream>
#include <memory>
#include <complex>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <cstring>

#include "MatrixMath.h"
#include "MemoryArena.h"

void QRIterate()
{

}

int main(int argc, char* argv[])
{
    using T = double;

    const int N = 3;

    auto arena = MemoryArenaPool::GetArena<T>(N * N * 6);

    T* A = arena.Allocate<T>(N * N);
    T* At = arena.Allocate<T>(N * N);
    T* AAt = arena.Allocate<T>(N * N);
    T* Q = arena.Allocate<T>(N * N);
    T* R = arena.Allocate<T>(N * N);
    T* RQ = arena.Allocate<T>(N * N);

    A[0] = 1; A[3] = 2; A[6] = 3;
    A[1] = 2; A[4] = 0; A[7] =-3;
    A[2] = 3; A[5] =-3; A[8] = 3;

    Transpose(At, A, N);

    // QR Iteration test
    PrintMatrix(A, N);
    for (int i = 0; i < 20; ++i)
    {
        QRDecompose_Householder(Q, R, A, N);
        MatMul(A, R, Q, N);
    }
    PrintMatrix(A, N);

    Transpose(A, At, N);
    for (int i = 0; i < 20; ++i)
    {
        QRDecompose_GramSchmidt(Q, R, A, N);
        MatMul(A, R, Q, N);
    }
    PrintMatrix(A, N);
    
    // Iterate -1 Dimension
    const int M = N - 1;
    A[0] = 1; A[2] = 2;
    A[1] = 2; A[3] = 0;

    Transpose(At, A, M);

    // QR Iteration test
    PrintMatrix(A, M);
    for (int i = 0; i < 20; ++i)
    {
        QRDecompose_Householder(Q, R, A, M);
        MatMul(A, R, Q, M);
    }
    PrintMatrix(A, M);

    Transpose(A, At, M);
    for (int i = 0; i < 20; ++i)
    {
        QRDecompose_GramSchmidt(Q, R, A, M);
        MatMul(A, R, Q, M);
    }
    PrintMatrix(A, M);

    return 0;
}
