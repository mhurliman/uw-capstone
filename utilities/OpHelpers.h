#pragma once

#include "pblas.h"

namespace ops
{
    template <typename T>
    void GERV2D(int context, int M, int N, T* A, int lda, int rsrc, int csrc);

    template <typename T>
    void GESD2D(int context, int M, int N, const T* A, int lda, int rdest, int cdest);

    template <typename T>
    void PvTRAN(
        const int* M, const int* N, 
        const ValueType<T>* alpha, const T* A, const int* ia, const int* ja, const int* descA, 
        const ValueType<T>* beta, T* C, const int* ic, const int* jc, const int* descC
    );

    template <typename T>
    void PvTRANU(
        const int* M, const int* N, 
        const ValueType<T>* alpha, const T* A, const int* ia, const int* ja, const int* descA, 
        const ValueType<T>* beta, T* C, const int* ic, const int* jc, const int* descC
    );

    template <typename T>
    void PvTRANC(
        const int* M, const int* N, 
        const ValueType<T>* alpha, const T* A, const int* ia, const int* ja, const int* descA, 
        const ValueType<T>* beta, T* C, const int* ic, const int* jc, const int* descC
    );

    template <typename T>
    void PvHEEV(
        const char* jobz, const char* uplo, const int* n, 
        T* a, const int* ia, const int* ja, const int* desca,
        ValueType<T>* w, T* z, const int* iz, const int* jz, const int* descz, 
        T* work, const int* lwork, ValueType<T>* rwork, const int* lrwork, int* info
    );


    template <typename T>
    void PvGEMM(
        const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const ValueType<T>* alpha,
        const T* a, const int* ia, const int* ja, const int* desca,
        const T* b, const int* ib, const int* jb, const int* descb,
        const ValueType<T>* beta,
        T* c, const int* ic, const int* jc, const int* descc
    );

    template <typename T>
    void PvGEMV(
        const char* trans, const int* M, const int* N, 
        const ValueType<T>* alpha,
        const T* A, const int* ia, const int* ja, const int* descA,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const ValueType<T>* beta,
        T* Y, const int* iy, const int* jy, const int* descY, const int* incY 
    );

    template <typename T>
    void PvDOT(
        const int* n,
        T* dot,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const T* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    template <typename T>
    void PvDOTU(
        const int* n,
        T* dot,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const T* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    template <typename T>
    void PvDOTC(
        const int* n,
        T* dot,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const T* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    template <typename T>
    void PvSCAL(
        const int* n,
        const ValueType<T>* alpha,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    template <typename T>
    void PvCOPY(
        const int* n,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const T* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    template <typename T>
    void PvAXPY(
        const int* n, 
        const ValueType<T>* alpha, 
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const T* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    template <typename T>
    void PvSWAP(
        const int* n,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const T* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    template <typename T>
    void PvNRM2(
        const int* n,
        ValueType<T>* norm2,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    template <typename T>
    void PvASUM(
        const int* n,
        ValueType<T>* asum,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    template <typename T>
    void PvAMAX(
        const int* n,
        T* amax, int* indx,
        const T* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    template <typename T>
    void PvLACPY(
        const char* uplo, const int* M, const int* N, 
        const T* A, const int* ia, const int* ja, const int* descA, 
        T* B, const int* ib, const int* jb, const int* descB
    );
}