#pragma once

#include "VectorMath.h"
#include "MemoryArena.h"

#include <algorithm>

template <typename T>
void Identity(T* A, int n)
{
    assert(A != 0 && n > 0);

    memset(A, 0, n * n * sizeof(T));

    for (int i = 0; i < n; ++i)
    {
        A[i + n * i] = 1.0;
    }
}

template <typename T, typename Generator>
void RandomMatrix(T* A, int n, Generator g)
{
    for (int i = 0; i < n * n; ++i)
    {
        A[i] = g();
    }
}

template <typename Generator>
void RandomMatrix(c64* A, int n, Generator g)
{
    for (int i = 0; i < n * n; ++i)
    {
        A[i] = c64{ g(), g() };
    }
}

template <typename Generator>
void RandomMatrixReal(c64* A, int n, Generator g)
{
    for (int i = 0; i < n * n; ++i)
    {
        A[i] = c64{ g(), 0 };
    }
}

template <typename Generator>
void RandomMatrixImag(c64* A, int n, Generator g)
{
    for (int i = 0; i < n * n; ++i)
    {
        A[i] = c64{ 0, g() };
    }
}

// Implements n x n matrix transpose operation
template <typename T>
void Transpose(T* B, const T* A, int n)
{
    assert(B != NULL && A != NULL && n > 0);

    if (B == A) // In-place transposition
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                std::swap(B[i + n * j], B[j + n * i]); 
            }
        }
    }
    else // General case
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                // Reads in row-major order
                // Writes in column-major order
                B[i + n * j] = A[j + n * i]; 
            }
        }
    }
}

template <typename T>
void MatMul(T* C, const T* B, const T* A, int n)
{
    assert(C != NULL && B != NULL && A != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C[i + n * j] = 0;

            for (int k = 0; k < n; ++k)
            {
                C[i + n * j] += B[i + n * k] * A[k + n * j];
            }
        }
    }
}

// Assume Q, R, & A are n x n matrices
template <typename T>
void QRDecomposeGS(T* Q, T* R, const T* A, int n)
{
    assert(Q != 0 && R != 0 && A != 0 && n > 0);

    memset(R, 0, n * n * sizeof(T));

    // Perform 
    for (int j = 0; j < n; ++j)
    {
        auto Qj = Q + n * j;  // Column j of Q (column-major)
        auto Aj = A + n * j; // Column j of A (column-major)

        auto v = Qj; // Use Qj as working memory for accumulator
        memcpy(v, Aj, n * sizeof(T));

        for (int i = 0; i < j; ++i)
        {
            auto Qi = Q + n * i;

            auto d = Dot(Qi, Aj, n);
            R[i + n * j] = d;

            MultiplyAdd(v, Qi, v, n, -d);
        }

        R[j + n * j] = Magnitude(v, n);
        Normalize(Qj, v, n);
    }
}


template <typename T>
void QRDecomposeHH(T* Q, T* R, const T* A, int n)
{
    assert(Q != 0 && R != 0 && A != 0 && n > 0);

    auto arena = MemoryArenaPool::GetArena<T>(n * 2);

    T* w = arena.template Allocate<T>(n);
    T* t = arena.template Allocate<T>(n);

    Identity(Q, n); // Q is row-major order
    memcpy(R, A, n * n * sizeof(T)); // R is column-major order

    for (int i = 0; i < n; ++i)
    {
        T normx = Magnitude(R + (i + n * i), n - i);
        T s = -Sign(R[i + n * i]);
        T u1 = R[i + n * i] - s * normx;

        // Compute w
        Divide(w, R + (i + n * i), n - i, u1);
        w[0] = 1.0;

        // Compute t
        T tau = -s * u1 / normx;
        Multiply(t, w, n, tau);

        for (int j = 0; j < n; ++j)
        {
            T* Rij = R + (i + n * j); // Column j of R (column-major order)

            T v = Dot(w, Rij, n - i);
            for (int k = 0; k < n - i; ++k)
            {
                Rij[k] -= t[k] * v;
            }
        }

        for (int j = 0; j < n; ++j)
        {
            T* Qji = Q + (i + n * j); // Row j of Q (row-major order)

            T v = Dot(w, Qji, n - i);
            for (int k = 0; k < n - i; ++k)
            {
                Qji[k] -= t[k] * v;
            }
        }
    }

    Transpose(Q, Q, n); // Transpose to column-major order for output
}

template <typename T>
T OneNorm(T* A, int n)
{
    assert(A != 0 && n > 0);

    T max = 0;
    for (int i = 0; i < n; ++i)
    {
        T sum = 0;

        for (int j = 0; j < n; ++j)
            sum += abs(A[j + n * i]);

        if (sum > max)
            max = sum;
    }

    return max;
}

template <typename T>
T FrobeniusNorm(T* A, int n)
{
    assert(A != 0 && n > 0);

    T sum = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            sum += Square(A[j + n * i]);
    }
    return sqrt(sum);
}

template <typename T>
T InfinityNorm(T* A, int n)
{
    assert(A != 0 && n > 0);

    T sum = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
            sum += abs(A[i + n * j]);
    }
    return sqrt(sum);
}

template <typename T>
void PrintMatrix(const T* A, int m, int n)
{
    assert(A != 0 && n > 0);

    for (int i = 0; i < m; ++i)
    {
        printf("[");

        for (int j = 0; j < n; ++j)
        {
            printf("%7.3f", A[i + m * j]);
        }

        printf("]\n");
    }
    printf("\n");
}

template <typename T>
void PrintMatrix(const T* A, int n)
{
    PrintMatrix(A, n, n);
}

template <typename T>
void PrintVector(const T* A, int n)
{
    PrintMatrix(A, 1, n);
}

template <typename T>
int TestOrthonormality(T* A, int n)
{
    assert(A != 0 && n > 0);

    for (int i = 0; i < n; ++i)
    {
        // Test normalized
        double m = Magnitude(A + (n * i), n);
        if (!IsEqual(m, 1.0))
        {
            return 0x1; // Unnormalized vector
        }

        // Test linear independence
        for (int j = i + 1; j < n; ++j)
        {
            T d = Dot(A + (n * i), A + (n * j), n);
            if (!IsZero(d))
            {
                return 0x2; // Dependent basis vectors
            }
        }
    }

    return 0x0;
}

template <typename T, typename QRDecompFunctor>
void QRIterate(T* E, T* A, int N, int iterations)
{
    assert(E != 0 && A != 0);

    // Grab scratch memory
    auto arena = MemoryArenaPool::GetArena<T>(N * N * 2);
    T* Q = arena.Allocate<T>(N * N);
    T* R = arena.Allocate<T>(N * N);

    // Reduce rank at each step
    for (int i = N; i > 2; --i)
    {
        // Iterate over QR decompositions & similar matrix swap
        for (int j = 0; j < iterations; ++j)
        {
            QRDecompFunctor()(Q, R, A, N);
            MatMul(A, R, Q, N);
        }

        // Grab approximated eigenvalue in BR entry
        E[i - 1] = A[i * i - 1];

        for (int j = 1; j < i; ++j)
        {
            memcpy(A + j * (i - 1), A + j * i, (i - 1) * sizeof(T));
        }
    }

    // Snag last two eigenvalues from remaining 2x2 matrix
    E[1] = A[3];
    E[0] = A[0];

    // Sort Eigenvalues from smallest to largest
    std::stable_sort(E, E + N);
}

template <typename T> struct FunctorQRDecompHH { void operator()(T*a, T* b, T* c, int d) { QRDecomposeHH(a, b, c, d); }};
template <typename T> 
constexpr void QRIterateHH(T* Eigenvalues, T* A, int N, int iterations = 15) { QRIterate<T, FunctorQRDecompHH<T>>(Eigenvalues, A, N, iterations); }

template <typename T> struct FunctorQRDecompGS { void operator()(T*a, T* b, T* c, int d) { QRDecomposeGS(a, b, c, d); }};
template <typename T> 
constexpr void QRIterateGS(T* Eigenvalues, T* A, int N, int iterations = 15) { QRIterate<T, FunctorQRDecompGS<T>>(Eigenvalues, A, N, iterations); }
