
#include <iostream>
#include <memory>
#include <complex>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <cstring>

#include "MatrixMath.h"
#include "MemoryArena.h"
#include "Random.h"

void ComplexMatMulTest()
{
    using T = c64;
    const int N = 2;

    auto arena = MemoryArenaPool::GetArena<T>(3 * N * N);
    T* A = arena.Allocate<T>(N * N);
    T* B = arena.Allocate<T>(N * N);
    T* C = arena.Allocate<T>(N * N);

    A[0] = 5; A[2] = c64{ -2, 1 };
    A[1] = 0; A[3] = c64{ 0, 4 };

    B[0] = 3; B[2] = c64{ 3, -4 };
    B[1] = c64{ -1, 9 }; B[3] = c64{ 2, 1 };

    MatMul(C, A, B, N);
    PrintMatrix(C, N);
}

void ComplexQRIterateTest()
{
    using T = c64;
    const int N = 2;

    auto arena = MemoryArenaPool::GetArena<T>(N * (N + 1));
    T* A = arena.Allocate<T>(N * N);
    T* E = arena.Allocate<T>(N);

    Random<double> r(5);
    RandomHermitian(A, N, r);

    // QR Iteration test
    PrintMatrix(A, N);
    QRIterateHH(E, A, N, 4);
    PrintVector(E, N);
}

void RealQRIterateTest()
{
    using T = double;
    const int N = 5;

    auto arena = MemoryArenaPool::GetArena<T>(N * (N + 1));
    T* A = arena.Allocate<T>(N * N);
    T* E = arena.Allocate<T>(N);

    Random<double> r(5);
    RandomHermitian(A, N, r);

    // QR Iteration test
    PrintMatrix(A, N);
    QRIterateHH(E, A, N, 3);
    PrintVector(E, N);
}

int main(int argc, char* argv[])
{
    ComplexQRIterateTest();

    return 0;
}
