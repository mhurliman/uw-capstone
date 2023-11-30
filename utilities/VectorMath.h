#pragma once

#include <assert.h>
#include <complex>

using c32 = std::complex<float>;
using c64 = std::complex<double>;

template <typename T>
bool IsEqual(T a, T b, double tolerance = 1e-5)
{
    return abs(a - b) <= tolerance;
}

template <>
bool IsEqual(c64 a, c64 b, double tolerance)
{
    return 
        IsEqual(a.real(), b.real(), tolerance) &&
        IsEqual(a.imag(), b.imag(), tolerance);
}

template <typename T>
bool IsZero(T a, double tolerance = 1e-5)
{
    return IsEqual(a, T(0), tolerance);
}

template <typename T>
T Sign(T v)
{
    return v < 0 ? -1.0 : 1.0;
}

// Square a value
template <typename T>
inline T Square(T v)
{
    return v * v;
}

template <typename T>
double Magnitude(T a)
{
    return abs(a);
}

template <>
double Magnitude(c64 a)
{
    return sqrt(Square(a.real()) + Square(a.imag()));
}

template <typename T>
T Conjugate(T v)
{
    return v; // Trivial case where v is pure real
}

template <>
c64 Conjugate(c64 v)
{
    return c64(v.real(), -v.imag()); // 
}

// Conjugate of complex vector
void Conjugate(c64* b, const c64* a, int n)
{
    assert(a != NULL && b != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        b[i] = Conjugate(a[i]);
    }
}

// Vector dot product
template <typename T>
T Dot(const T* a, const T* b, int n)
{
    assert(a != NULL && b != NULL && n > 0);

    double c = 0;
    for (int i = 0; i < n; ++i)
    {
        c += Conjugate(a[i]) * b[i]; // Decays to a[i] * b[i] in non-complex case
    }

    return c;
}

// Vector magnitude (L2 norm)
template <typename T>
double Magnitude(const T* a, int n)
{
    return sqrt(Dot(a, a, n));
}

// Vector normalization
template <typename T>
void Normalize(T* b, const T* a, int n)
{
    assert(b != NULL && a != NULL && n > 0);

    T invMag = 1.0 / Magnitude(a, n);
    for (int i = 0; i < n; ++i)
    {
        b[i] = a[i] * invMag;
    }
}

// Component-wise subtraction; c = a - b
template <typename T>
void Subtract(T* c, const T* a, const T* b, int n)
{
    assert(a != NULL && b != NULL && c != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] - b[i];
    }
}

// Component-wise addition; c = a + b
template <typename T>
void Add(T* c, const T* a, const T* b, int n)
{
    assert(a != NULL && b != NULL && c != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

// Component-wise multiply; c = a * b
template <typename T>
void Multiply(T* c, const T* a, const T* b, int n)
{
    assert(a != NULL && b != NULL && c != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] * b[i];
    }
}

// Implements b = a * v
template <typename T, typename U>
void Multiply(T* b, const T* a, int n, U v)
{
    assert(a != NULL && b != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        b[i] = a[i] * v;
    }
}

// Implements b = a / v
template <typename T, typename U>
void Divide(T* b, const T* a, int n, U v)
{
    return Multiply(b, a, n, 1.0 / v);
}


// Implements c = a * v + b
template <typename T, typename U>
void MultiplyAdd(T* c, const T* a, const T* b, int n, U v)
{
    assert(c != NULL && a != NULL && b != NULL && n > 0);

    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] * v + b[i];
    }
}
