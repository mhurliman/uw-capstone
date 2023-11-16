
#include <iostream>
#include <memory>
#include <complex>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

using c32 = std::complex<float>;
using c64 = std::complex<double>;

c64 Conjugate(c64 v)
{
    return c64(v.real(), -v.imag());
}

void Conjugate(c64* B, const c64* A, int n)
{
    assert(A != 0 && B != 0);

    for (int i = 0; i < n; ++i)
    {
        B[i] = Conjugate(A[i]);
    }
}

template <typename T>
inline T Square(T v)
{
    return v * v;
}

template <typename T>
T Dot(const T* a, const T* b, int n)
{
    assert(a != 0 && b != 0);

    double c = 0;
    for (int i = 0; i < n; ++i)
    {
        c += a[i] * b[i];
    }

    return c;
}

template <>
c64 Dot(const c64* a, const c64* b, int n)
{
    assert(a != 0 && b != 0);

    c64 c = 0;
    for (int i = 0; i < n; ++i)
    {
        c += a[i] * Conjugate(b[i]);
    }

    return c;
}

template <typename T>
double Magnitude(const T* A, int n)
{
    assert(A != 0);

    double mag = 0;
    for (int i = 0; i < n; ++i)
    {
        mag += Square(A[i]);
    }
    return sqrt(mag);
}

template <>
double Magnitude<c64>(const c64* A, int n)
{
    assert(A != 0);

    double mag = 0;
    for (int i = 0; i < n; ++i)
    {
        mag += Square(A[i].real()) + Square(A[i].imag());
    }
    return sqrt(mag);
}

template <typename T>
void Normalize(T* B, const T* A, int n)
{
    assert(B != 0 && A != 0);

    double invMag = 1.0 / Magnitude(A, n);
    for (int i = 0; i < n; ++i)
    {
        B[i] = A[i] * invMag;
    }
}

template <typename T>
void Subtract(T* c, const T* a, const T* b, int n)
{
    assert(a != 0 && b != 0 && c != 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] - b[i];
    }
}

template <typename T>
void Add(T* c, const T* a, const T* b, int n)
{
    assert(a != 0 && b != 0 && c != 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void Multiply(T* c, const T* a, const T* b, int n)
{
    assert(a != 0 && b != 0 && c != 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] * b[i];
    }
}

template <typename T, typename U>
void Multiply(T* b, const T* a, int n, U v)
{
    assert(a != 0 && b != 0);

    for (int i = 0; i < n; ++i)
    {
        b[i] = a[i] * v;
    }
}

template <typename T, typename U>
void MultiplyAccum(T* b, const T* a, int n, U v)
{
    assert(a != 0 && b != 0);

    for (int i = 0; i < n; ++i)
    {
        b[i] += a[i] * v;
    }
}

template <typename T>
void Transpose(T* B, const T* A, int n)
{
    assert(B != 0 && A != 0);

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

// Assume Q, R, & A are n x n matrices
template <typename T>
void QRDecompose_GramSchmidt(T* Q, T* R, const T* A, int n)
{
    std::memset(R, 0, n * n * sizeof(T));

    // Perform 
    for (int j = 0; j < n; ++j)
    {
        auto Qj = Q + n * j;  // Column j of Q (column-major)
        auto Aj = A + n * j; // Column j of A (column-major)

        auto v = Qj; // Use Qj as working memory for accumulator
        std::memcpy(v, Aj, n * sizeof(T));

        for (int i = 0; i < j; ++i)
        {
            auto Qi = Q + n * i;

            auto d = Dot(Qi, Aj, n);
            R[i + n * j] = d;

            MultiplyAccum(v, Qi, n, -d);
        }

        R[j + n * j] = Magnitude(v, n);
        Normalize(Qj, v, n);
    }
}

template <typename T>
void QRDecompose_Householder(T* Q, T* R, const T* A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        
    }
}

template <typename T>
void RotationMatrix(T* A, int n, double theta, double psi)
{
    assert(A != 0 && n == 3);
}

template <typename T>
void PrintMatrix(const T* A, int n)
{
    for (int i = 0; i < n; ++i)
    {
        std::cout << '[';

        for (int j = 0; j < n; ++j)
        {
            std::cout << A[i + n * j];

            if (j + 1 < n)
                std::cout << ',';
        }

        std::cout << ']' << std::endl;
    }
}

int main(int argc, char* argv[])
{
    const int N = 3;

    using T = double;
    std::unique_ptr<T[]> A(new T[N * N * sizeof(T)]);
    std::unique_ptr<T[]> Q(new T[N * N * sizeof(T)]);
    std::unique_ptr<T[]> R(new T[N * N * sizeof(T)]);

    A[0] = 1; A[3] = 2; A[6] = 3;
    A[1] =-1; A[4] = 0; A[7] =-3;
    A[2] = 0; A[5] =-2; A[8] = 3;

    QRDecompose_GramSchmidt(Q.get(), R.get(), A.get(), N);

    PrintMatrix(A.get(), N);
    PrintMatrix(Q.get(), N);
    PrintMatrix(R.get(), N);

    return 0;
}
