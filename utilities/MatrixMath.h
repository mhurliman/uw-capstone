#pragma once

#include "VectorMath.h"
#include "MemoryArena.h"

// Implements n x n matrix transpose operation
template <typename T>
void Transpose(T* B, const T* A, int n)
{
    assert(B != NULL && A != NULL);

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
void QRDecompose_GramSchmidt(T* Q, T* R, const T* A, int n)
{
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
void Identity(T* A, int n)
{
    memset(A, 0, n * n * sizeof(T));

    for (int i = 0; i < n; ++i)
    {
        A[i + n * i] = 1.0;
    }
}


template <typename T>
void QRDecompose_Householder(T* Q, T* R, const T* A, int n)
{
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
int Norm(T)
{
    return 0;
}
