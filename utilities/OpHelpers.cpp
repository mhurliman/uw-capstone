
#include "OpHelpers.h"

namespace ops
{
    // -----------------------------
    // GERV2D
    
    template <>
    void GERV2D<float>(int context, int M, int N, float* A, int lda, int rsrc, int csrc)
    {
        Csgerv2d(context, M, N, A, lda, rsrc, csrc);
    }

    template <>
    void GERV2D<double>(int context, int M, int N, double* A, int lda, int rsrc, int csrc)
    {
        Cdgerv2d(context, M, N, A, lda, rsrc, csrc);
    }

    template <>
    void GERV2D<c32>(int context, int M, int N, c32* A, int lda, int rsrc, int csrc)
    {
        Ccgerv2d(context, M, N, A, lda, rsrc, csrc);
    }

    template <>
    void GERV2D<c64>(int context, int M, int N, c64* A, int lda, int rsrc, int csrc)
    {
        Czgerv2d(context, M, N, A, lda, rsrc, csrc);
    }


    // -----------------------------
    // GESD2D

    template <>
    void GESD2D<float>(int context, int M, int N, const float* A, int lda, int rdest, int cdest)
    {
        Csgesd2d(context, M, N, A, lda, rdest, cdest);
    }

    template <>
    void GESD2D<double>(int context, int M, int N, const double* A, int lda, int rdest, int cdest)
    {
        Cdgesd2d(context, M, N, A, lda, rdest, cdest);
    }

    template <>
    void GESD2D<c32>(int context, int M, int N, const c32* A, int lda, int rdest, int cdest)
    {
        Ccgesd2d(context, M, N, A, lda, rdest, cdest);
    }

    template <>
    void GESD2D<c64>(int context, int M, int N, const c64* A, int lda, int rdest, int cdest)
    {
        Czgesd2d(context, M, N, A, lda, rdest, cdest);
    }



    // -----------------------------
    // PvTRAN

    template <>
    void PvTRAN<float>(
        const int* M, const int* N, 
        const float* alpha, const float* A, const int* ia, const int* ja, const int* descA, 
        const float* beta, float* C, const int* ic, const int* jc, const int* descC
    )
    {
        pstran_(M, N, alpha, A, ia, ja, descA, beta, C, ic, jc, descC);
    }

    template <>
    void PvTRAN<double>(
        const int* M, const int* N, 
        const double* alpha, const double* A, const int* ia, const int* ja, const int* descA, 
        const double* beta, double* C, const int* ic, const int* jc, const int* descC
    )
    {
        pdtran_(M, N, alpha, A, ia, ja, descA, beta, C, ic, jc, descC);
    }


    // -----------------------------
    // PvTRANU

    template <>
    void PvTRANU<c32>(
        const int* M, const int* N, 
        const float* alpha, const c32* A, const int* ia, const int* ja, const int* descA, 
        const float* beta, c32* C, const int* ic, const int* jc, const int* descC
    )
    {
        pctranu_(M, N, alpha, A, ia, ja, descA, beta, C, ic, jc, descC);
    }
    
    template <>
    void PvTRANU<c64>(
        const int* M, const int* N, 
        const double* alpha, const c64* A, const int* ia, const int* ja, const int* descA, 
        const double* beta, c64* C, const int* ic, const int* jc, const int* descC
    )
    {
        pztranu_(M, N, alpha, A, ia, ja, descA, beta, C, ic, jc, descC);
    }
    
    
    // -----------------------------
    // PvTRANC

    template <>
    void PvTRANC<c32>(
        const int* M, const int* N, 
        const float* alpha, const c32* A, const int* ia, const int* ja, const int* descA, 
        const float* beta, c32* C, const int* ic, const int* jc, const int* descC
    )
    {
        pctranc_(M, N, alpha, A, ia, ja, descA, beta, C, ic, jc, descC);
    }

    template <>
    void PvTRANC<c64>(
        const int* M, const int* N, 
        const double* alpha, const c64* A, const int* ia, const int* ja, const int* descA, 
        const double* beta, c64* C, const int* ic, const int* jc, const int* descC
    )
    {
        pztranc_(M, N, alpha, A, ia, ja, descA, beta, C, ic, jc, descC);
    }
    

    // -----------------------------
    // PvHEEV

    template <>
    void PvHEEV<c32>(
        const char* jobz, const char* uplo, const int* n, 
        c32* a, const int* ia, const int* ja, const int* desca, 
        float* w, c32* z, const int* iz, const int* jz, const int* descz, 
        c32* work, const int* lwork, float* rwork, const int* lrwork, int* info
    )
    {
        pcheev_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork, rwork, lrwork, info);
    }

    template <>
    void PvHEEV<c64>(
        const char* jobz, const char* uplo, const int* n, 
        c64* a, const int* ia, const int* ja, const int* desca, 
        double* w, c64* z, const int* iz, const int* jz, const int* descz, 
        c64* work, const int* lwork, double* rwork, const int* lrwork, int* info
    )
    {
        pzheev_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork, rwork, lrwork, info);
    }
    

    // -----------------------------
    // PvGERQF

    template <>
    void PvGERQF<float>(
        const int* M, const int* N, 
        float* A, const int* ia, const int* ja, const int* descA,
        float* tau, float* work, const int* lwork, int* info
    )
    {
        psgerqf_(M, N, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvGERQF<double>(
        const int* M, const int* N, 
        double* A, const int* ia, const int* ja, const int* descA,
        double* tau, double* work, const int* lwork, int* info
    )
    {
        pdgerqf_(M, N, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvGERQF<c32>(
        const int* M, const int* N, 
        c32* A, const int* ia, const int* ja, const int* descA,
        c32* tau, c32* work, const int* lwork, int* info
    )
    {
        pcgerqf_(M, N, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvGERQF<c64>(
        const int* M, const int* N, 
        c64* A, const int* ia, const int* ja, const int* descA,
        c64* tau, c64* work, const int* lwork, int* info
    )
    {
        pzgerqf_(M, N, A, ia, ja, descA, tau, work, lwork, info);
    }

    
    // -----------------------------
    // PvORMQR

    template <>
    void PvORMQR<float>(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* tau, 
        float* C, const int* ic, const int* jc, const int* descC,
        float* work, const int* lwork, int* info
    )
    {
        psormqr_(side, trans, M, N, K, A, ia, ja, descA, tau, C, ic, jc, descC, work, lwork, info);
    }

    template <>
    void PvORMQR<double>(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* tau, 
        double* C, const int* ic, const int* jc, const int* descC,
        double* work, const int* lwork, int* info
    )
    {
        pdormqr_(side, trans, M, N, K, A, ia, ja, descA, tau, C, ic, jc, descC, work, lwork, info);
    }


    // -----------------------------
    // PvUNMQR

    template <>
    void PvUNMQR<c32>(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* tau, 
        c32* C, const int* ic, const int* jc, const int* descC,
        c32* work, const int* lwork, int* info
    )
    {
        pcunmqr_(side, trans, M, N, K, A, ia, ja, descA, tau, C, ic, jc, descC, work, lwork, info);
    }

    template <>
    void PvUNMQR<c64>(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* tau, 
        c64* C, const int* ic, const int* jc, const int* descC,
        c64* work, const int* lwork, int* info
    )
    {
        pzunmqr_(side, trans, M, N, K, A, ia, ja, descA, tau, C, ic, jc, descC, work, lwork, info);
    }

    
    // -----------------------------
    // PvORGQR

    template <>
    void PvORGQR<float>(
        const int* M, const int* N, const int* K,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* tau, float* work, const int* lwork, int* info
    )
    {
        psorgqr_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvORGQR<double>(
        const int* M, const int* N, const int* K,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* tau, double* work, const int* lwork, int* info
    )
    {
        pdorgqr_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }


    // -----------------------------
    // PvUNGQR

    template <>
    void PvUNGQR<c32>(
        const int* M, const int* N, const int* K,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* tau, c32* work, const int* lwork, int* info
    )
    {
        pcungqr_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvUNGQR<c64>(
        const int* M, const int* N, const int* K,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* tau, c64* work, const int* lwork, int* info
    )
    {
        pzungqr_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }

    
    // -----------------------------
    // PvORGR2

    template <>
    void PvORGR2<float>(
        const int* M, const int* N, const int* K,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* tau, float* work, const int* lwork, int* info
    )
    {
        psorgr2_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvORGR2<double>(
        const int* M, const int* N, const int* K,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* tau, double* work, const int* lwork, int* info
    )
    {
        pdorgr2_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }


    // -----------------------------
    // PvUNGR2

    template <>
    void PvUNGR2<c32>(
        const int* M, const int* N, const int* K,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* tau, c32* work, const int* lwork, int* info
    )
    {
        pcungr2_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }

    template <>
    void PvUNGR2<c64>(
        const int* M, const int* N, const int* K,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* tau, c64* work, const int* lwork, int* info
    )
    {
        pzungr2_(M, N, K, A, ia, ja, descA, tau, work, lwork, info);
    }


    // -----------------------------
    // PvGEMM

    template <>
    void PvGEMM<float>(
        const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const float* alpha,
        const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb,
        const float* beta,
        float* c, const int* ic, const int* jc, const int* descc
    )
    {
        psgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
    }

    template <>
    void PvGEMM<double>(
        const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        double* c, const int* ic, const int* jc, const int* descc
    )
    {
        pdgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
    }

    template <>
    void PvGEMM<c32>(
        const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const float* alpha,
        const c32* a, const int* ia, const int* ja, const int* desca,
        const c32* b, const int* ib, const int* jb, const int* descb,
        const float* beta,
        c32* c, const int* ic, const int* jc, const int* descc
    )
    {
        pcgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
    }

    template <>
    void PvGEMM<c64>(
        const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const c64* a, const int* ia, const int* ja, const int* desca,
        const c64* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        c64* c, const int* ic, const int* jc, const int* descc
    )
    {
        pzgemm_(transa, transb, m, n, k, alpha, a, ia, ja, desca, b, ib, jb, descb, beta, c, ic, jc, descc);
    }


    // -----------------------------
    // PvGEMV

    template <>
    void PvGEMV<float>(
        const char* trans, const int* M, const int* N, 
        const float* alpha,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* beta,
        float* Y, const int* iy, const int* jy, const int* descY, const int* incY 
    )
    {
        psgemv_(trans, M, N, alpha, A, ia, ja, descA, X, ix, jx, descX, incX, beta, Y, iy, jy, descY, incY);
    }

    template <>
    void PvGEMV<double>(
        const char* trans, const int* M, const int* N, 
        const double* alpha,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* beta,
        double* Y, const int* iy, const int* jy, const int* descY, const int* incY 
    )
    {
        pdgemv_(trans, M, N, alpha, A, ia, ja, descA, X, ix, jx, descX, incX, beta, Y, iy, jy, descY, incY);
    }

    template <>
    void PvGEMV<c32>(
        const char* trans, const int* M, const int* N, 
        const float* alpha,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* beta,
        c32* Y, const int* iy, const int* jy, const int* descY, const int* incY 
    )
    {
        pcgemv_(trans, M, N, alpha, A, ia, ja, descA, X, ix, jx, descX, incX, beta, Y, iy, jy, descY, incY);
    }

    template <>
    void PvGEMV<c64>(
        const char* trans, const int* M, const int* N, 
        const double* alpha,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* beta,
        c64* Y, const int* iy, const int* jy, const int* descY, const int* incY 
    )
    {
        pzgemv_(trans, M, N, alpha, A, ia, ja, descA, X, ix, jx, descX, incX, beta, Y, iy, jy, descY, incY);
    }


    // -----------------------------
    // PvDOT

    template <>
    void PvDOT<float>(
        const int* n,
        float* dot,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
        )
    {
        psdot_(n, dot, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvDOT<double>(
        const int* n,
        double* dot,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
        )
    {
        pddot_(n, dot, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }


    // -----------------------------
    // PvDOTU

    template <>
    void PvDOTU<c32>(
        const int* n,
        c32* dot,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pcdotu_(n, dot, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvDOTU<c64>(
        const int* n,
        c64* dot,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pzdotu_(n, dot, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }


    // -----------------------------
    // PvDOTC
    
    template <>
    void PvDOTC<c32>(
        const int* n,
        c32* dot,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
        )
    {
        pcdotc_(n, dot, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvDOTC<c64>(
        const int* n,
        c64* dot,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
        )
    {
        pzdotc_(n, dot, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }


    // -----------------------------
    // PvSCAL

    template <>
    void PvSCAL<float>(
        const int* n,
        const float* alpha,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        psscal_(n, alpha, X,  ix, jx, descX, incX);
    }

    template <>
    void PvSCAL<double>(
        const int* n,
        const double* alpha,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pdscal_(n, alpha, X,  ix, jx, descX, incX);
    }

    template <>
    void PvSCAL<c32>(
        const int* n,
        const float* alpha,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pcscal_(n, alpha, X,  ix, jx, descX, incX);
    }

    template <>
    void PvSCAL<c64>(
        const int* n,
        const double* alpha,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pzscal_(n, alpha, X,  ix, jx, descX, incX);
    }


    // -----------------------------
    // PvCOPY

    template <>
    void PvCOPY<float>(
        const int* n,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pscopy_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvCOPY<double>(
        const int* n,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pdcopy_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvCOPY<c32>(
        const int* n,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pccopy_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvCOPY<c64>(
        const int* n,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pzcopy_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }
    
    // -----------------------------
    // PvAXPY

    template <>
    void PvAXPY<float>(
        const int* n, 
        const float* alpha, 
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        psaxpy_(n, alpha, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvAXPY<double>(
        const int* n, 
        const double* alpha, 
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pdaxpy_(n, alpha, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvAXPY<c32>(
        const int* n, 
        const float* alpha, 
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pcaxpy_(n, alpha, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvAXPY<c64>(
        const int* n, 
        const double* alpha, 
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pzaxpy_(n, alpha, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }


    // -----------------------------
    // PvSWAP

    template <>
    void PvSWAP<float>(
        const int* n,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        psswap_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvSWAP<double>(
        const int* n,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pdswap_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvSWAP<c32>(
        const int* n,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pcswap_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }

    template <>
    void PvSWAP<c64>(
        const int* n,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    {
        pzswap_(n, X, ix, jx, descX, incX, Y, iy, jy, descY, incY);
    }


    // -----------------------------
    // PvNRM2

    template <>
    void PvNRM2<float>(
        const int* n,
        float* norm2,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        psnrm2_(n, norm2, X, ix, jx, descX, incX);
    }

    template <>
    void PvNRM2<double>(
        const int* n,
        double* norm2,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pdnrm2_(n, norm2, X, ix, jx, descX, incX);
    }

    template <>
    void PvNRM2<c32>(
        const int* n,
        float* norm2,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pscnrm2_(n, norm2, X, ix, jx, descX, incX);
    }

    template <>
    void PvNRM2<c64>(
        const int* n,
        double* norm2,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pdznrm2_(n, norm2, X, ix, jx, descX, incX);
    }


    // -----------------------------
    // PvASUM

    template <>
    void PvASUM<float>(
        const int* n,
        float* asum,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        psasum_(n, asum, X, ix, jx, descX, incX);
    }

    template <>
    void PvASUM<double>(
        const int* n,
        double* asum,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pdasum_(n, asum, X, ix, jx, descX, incX);
    }

    template <>
    void PvASUM<c32>(
        const int* n,
        float* asum,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pscasum_(n, asum, X, ix, jx, descX, incX);
    }

    template <>
    void PvASUM<c64>(
        const int* n,
        double* asum,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pdzasum_(n, asum, X, ix, jx, descX, incX);
    }


    // -----------------------------
    // PvAMAX

    template <>
    void PvAMAX<float>(
        const int* n,
        float* amax, int* indx,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        psamax_(n, amax, indx, X, ix, jx, descX, incX);
    }

    template <>
    void PvAMAX<double>(
        const int* n,
        double* amax, int* indx,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pdamax_(n, amax, indx, X, ix, jx, descX, incX);
    }

    template <>
    void PvAMAX<c32>(
        const int* n,
        c32* amax, int* indx,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pcamax_(n, amax, indx, X, ix, jx, descX, incX);
    }

    template <>
    void PvAMAX<c64>(
        const int* n,
        c64* amax, int* indx,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    )
    {
        pzamax_(n, amax, indx, X, ix, jx, descX, incX);
    }


    // -----------------------------
    // PvLACPY

    template <>
    void PvLACPY<float>(
        const char* uplo, const int* M, const int* N, 
        const float* A, const int* ia, const int* ja, const int* descA, 
        float* B, const int* ib, const int* jb, const int* descB
    )
    {
        pslacpy_(uplo, M, N, A, ia, ja, descA, B, ib, jb, descB);
    }

    template <>
    void PvLACPY<double>(
        const char* uplo, const int* M, const int* N, 
        const double* A, const int* ia, const int* ja, const int* descA, 
        double* B, const int* ib, const int* jb, const int* descB
    )
    {
        pdlacpy_(uplo, M, N, A, ia, ja, descA, B, ib, jb, descB);
    }

    template <>
    void PvLACPY<c32>(
        const char* uplo, const int* M, const int* N, 
        const c32* A, const int* ia, const int* ja, const int* descA, 
        c32* B, const int* ib, const int* jb, const int* descB
    )
    {
        pclacpy_(uplo, M, N, A, ia, ja, descA, B, ib, jb, descB);
    }

    template <>
    void PvLACPY<c64>(
        const char* uplo, const int* M, const int* N, 
        const c64* A, const int* ia, const int* ja, const int* descA, 
        c64* B, const int* ib, const int* jb, const int* descB
    )
    {
        pzlacpy_(uplo, M, N, A, ia, ja, descA, B, ib, jb, descB);
    }
}
