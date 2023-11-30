
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

int main(int argc, char* argv[])
{
    using T = double;
    const int N = 5;

    auto arena = MemoryArenaPool::GetArena<T>(N * (N + 1));
    T* A = arena.Allocate<T>(N * N);
    T* E = arena.Allocate<T>(N);

    Random<T> r(5);
    RandomMatrix(A, N, r);

    // QR Iteration test
    PrintMatrix(A, N);
    QRIterateHH(E, A, N, 20);
    PrintVector(E, N);

    return 0;
}
