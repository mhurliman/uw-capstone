
#include <iostream>
#include <memory>
#include <complex>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <cstring>

#include "MatrixMath.h"
#include "MemoryArena.h"

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
    A[1] =-1; A[4] = 0; A[7] =-3;
    A[2] = 0; A[5] =-2; A[8] = 3;

    Transpose(At, A, N);
    MatMul(AAt, A, At, N);

    PrintMatrix(AAt, N);

    // QR Iteration test
    PrintMatrix(A, N);
    for (int i = 0; i < 100; ++i)
    {
        QRDecompose_Householder(Q, R, A, N);
        MatMul(A, R, Q, N);
    }

    PrintMatrix(A, N);

    return 0;
}
