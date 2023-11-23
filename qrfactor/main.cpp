
#include <iostream>
#include <memory>
#include <complex>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <cstring>

#include "MatrixMath.h"
#include "MemoryArena.h"

template <typename T>
void PrintMatrix(const T* A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        printf("[");

        for (int j = 0; j < n; ++j)
        {
            printf("%6.2f", A[i + n * j]);
        }

        printf("]\n");
    }
    printf("\n");
}

template <typename T>
int TestOrthonormality(T* A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        // Test normalization
        double m = Magnitude(A + (n * i), n);
        if (!IsEqual(m, 1.0))
        {
            return 0x1;
        }

        // Test linear independence
        for (int j = i + 1; j < n; ++j)
        {
            T d = Dot(A + (n * i), A + (n * j), n);
            if (!IsZero(d))
            {
                return 0x2;
            }
        }
    }

    return 0x0;
}

int main(int argc, char* argv[])
{
    const int N = 3;

    using T = double;
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

    // QR Iteration test
    PrintMatrix(A, N);
    for (int i = 0; i < 10000; ++i)
    {
        QRDecompose_Householder(Q, R, A, N);
        MatMul(A, R, Q, N);
    }

    PrintMatrix(A, N);

    return 0;
}
