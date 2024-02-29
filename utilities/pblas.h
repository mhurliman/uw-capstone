#pragma once

#include <assert.h>
#include <mpi.h>
#include <complex>
#include <ostream>

#define MPI_CHECK(x) do { \
    int errCode = x; \
    if (errCode != MPI_SUCCESS) { \
        int errLen = 0; \
        char errStr[MPI_MAX_ERROR_STRING] {}; \
        MPI_Error_string(errCode, errStr, &errLen); \
        printf("%s", errStr); \
        assert(x == MPI_SUCCESS); \
    } \
} while(0)

#define ROW_MAJOR_INDEX(row, col, numCols) (col + row * numCols)
#define COL_MAJOR_INDEX(row, col, numRows) (row + col * numRows)

#define RMIDX(row, col, numCols) ROW_MAJOR_INDEX(row, col, numCols)
#define CMIDX(row, col, numRows) COL_MAJOR_INDEX(row, col, numRows)

using c32 = std::complex<float>;
using c64 = std::complex<double>;

std::ostream& operator<<(std::ostream& os, c32 x);
std::ostream& operator<<(std::ostream& os, c64 x);

// -----------------------------
// Templatized ops for convenience

template <typename T, typename Enable = void>
struct TypeTraits 
{ 
    using type = typename T::value_type; 
};

template <typename T>
struct TypeTraits<T, typename std::enable_if_t<std::is_floating_point_v<T>>> 
{ 
    using type = T;
};

template <typename T>
using ValueType = typename TypeTraits<T>::type;

template <typename T>
inline constexpr MPI_Datatype MPI_Type = std::is_same_v<ValueType<T>, double> ? MPI_DOUBLE : MPI_FLOAT;

template <typename T>
auto FormatZeros(T v, ValueType<T> err = 1e-4) -> std::enable_if_t<std::is_floating_point_v<T>, T>
{
    return (std::fabs(v)) > err ? v : T{};
}

template <typename T>
auto FormatZeros(T v, ValueType<T> err = 1e-4) -> std::enable_if_t<std::is_compound_v<T>, T>
{
    return { FormatZeros(std::real(v), err), FormatZeros(std::imag(v), err) };
}

template <typename T>
int ToIndex(T v) { return static_cast<int>(std::real(v)); }


extern "C" {
    // Cblacs declarations
    void Cblacs_pinfo(int* mypnum, int* nprocs);
    void Cblacs_get(int context, int what, int* val);
    void Cblacs_gridinit(int* context, const char* order, int npRow, int npCol);
    void Cblacs_pcoord(int context, int pNum, int* pRow, int* pCol);
    void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
    void Cblacs_gridexit(int context);
    void Cblacs_barrier(int context, const char* scope);

    void Csgerv2d(int context, int M, int N, float* A, int lda, int rsrc, int csrc);
    void Csgesd2d(int context, int M, int N, const float* A, int lda, int rdest, int cdest);

    void Cdgerv2d(int context, int M, int N, double* A, int lda, int rsrc, int csrc);
    void Cdgesd2d(int context, int M, int N, const double* A, int lda, int rdest, int cdest);

    void Ccgerv2d(int context, int M, int N, c32* A, int lda, int rsrc, int csrc);
    void Ccgesd2d(int context, int M, int N, const c32* A, int lda, int rdest, int cdest);
 
    void Czgerv2d(int context, int M, int N, c64* A, int lda, int rsrc, int csrc);
    void Czgesd2d(int context, int M, int N, const c64* A, int lda, int rdest, int cdest);

    // Scalapack tools
    int numroc_(const int* n, const int* nb, const int* iproc, const int* srcproc, const int* nprocs );

    void descinit_(
        int* desc, 
        const int* m, const int* n, 
        const int* mb, const int* nb, 
        const int* irsrc, const int* icsrc, 
        const int* ictxt, const int* lld, int* info);

    int indxg2p_(const int* indxglob, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
    int indxg2l_(const int* indxglob, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);
    int indxl2g_(const int* indxloc, const int* nb, const int* iproc, const int* isrcproc, const int* nprocs);

    // Pblas declarations
    // PvTRAN
    void pstran_(const int* M, const int* N, const float* alpha, const float* A, const int* ia, const int* ja, const int* descA, const float* beta, float* C, const int* ic, const int* jc, const int* descC);
    void pdtran_(const int* M, const int* N, const double* alpha, const double* A, const int* ia, const int* ja, const int* descA, const double* beta, double* C, const int* ic, const int* jc, const int* descC);

    // PvTRANU
    void pctranu_(const int* M, const int* N, const float* alpha, const c32* A, const int* ia, const int* ja, const int* descA, const float* beta, c32* C, const int* ic, const int* jc, const int* descC);
    void pztranu_(const int* M, const int* N, const double* alpha, const c64* A, const int* ia, const int* ja, const int* descA, const double* beta, c64* C, const int* ic, const int* jc, const int* descC);

    // PvTRANC
    void pctranc_(const int* M, const int* N, const float* alpha, const c32* A, const int* ia, const int* ja, const int* descA, const float* beta, c32* C, const int* ic, const int* jc, const int* descC);
    void pztranc_(const int* M, const int* N, const double* alpha, const c64* A, const int* ia, const int* ja, const int* descA, const double* beta, c64* C, const int* ic, const int* jc, const int* descC);


    // PvGEMM
    void psgemm_(
        const char* transA, const char* transB,
        const int* M, const int* N, const int* K,
        const float* alpha,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* B, const int* ib, const int* jb, const int* descB,
        const float* beta,
        float* C, const int* ic, const int* jc, const int* descC
    );

    void pdgemm_(
        const char* transA, const char* transB,
        const int* M, const int* N, const int* K,
        const double* alpha,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* B, const int* ib, const int* jb, const int* descB,
        const double* beta,
        double* C, const int* ic, const int* jc, const int* descC
    );

    void pcgemm_(
        const char* transA, const char* transB,
        const int* M, const int* N, const int* K,
        const float* alpha,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* B, const int* ib, const int* jb, const int* descB,
        const float* beta,
        c32* C, const int* ic, const int* jc, const int* descC
    );

    void pzgemm_(
        const char* transA, const char* transB,
        const int* M, const int* N, const int* K,
        const double* alpha,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* B, const int* ib, const int* jb, const int* descB,
        const double* beta,
        c64* C, const int* ic, const int* jc, const int* descC
    );


    // PvGEMV
    void psgemv_( 
        const char* trans, const int* M, const int* N, 
        const float* alpha,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* beta,
        float* Y, const int* ij, const int* jy, const int* descY, const int* incY 
    );
    
    void pdgemv_( 
        const char* trans, const int* M, const int* N, 
        const double* alpha,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* beta,
        double* Y, const int* ij, const int* jy, const int* descY, const int* incY 
    );

    void pcgemv_( 
        const char* trans, const int* M, const int* N, 
        const float* alpha,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* beta,
        c32* Y, const int* ij, const int* jy, const int* descY, const int* incY 
    );

    void pzgemv_( 
        const char* trans, const int* M, const int* N, 
        const double* alpha,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* beta,
        c64* Y, const int* ij, const int* jy, const int* descY, const int* incY 
    );


    // PvHEEV
    void pcheev_(
        const char* jobz, const char* uplo, const int* N, 
        c32* A, const int* ia, const int* ja, const int* descA,
        float* W,
        c32* Z, const int* iz, const int* jz, const int* descZ,
        c32* work, const int* lwork, float* rwork, const int* lrwork, int* info
    );

    void pzheev_(
        const char* jobz, const char* uplo, const int* N, 
        c64* A, const int* ia, const int* ja, const int* descA,
        double* W,
        c64* Z, const int* iz, const int* jz, const int* descZ,
        c64* work, const int* lwork, double* rwork, const int* lrwork, int* info
    );


    // PvGERQF
    void psgerqf_(
        const int* M, const int* N, 
        float* A, const int* ia, const int* ja, const int* descA,
        float* tau, float* work, const int* lwork, int* info
    );

    void pdgerqf_(
        const int* M, const int* N, 
        double* A, const int* ia, const int* ja, const int* descA,
        double* tau, double* work, const int* lwork, int* info
    );

    void pcgerqf_(
        const int* M, const int* N, 
        c32* A, const int* ia, const int* ja, const int* descA,
        c32* tau, c32* work, const int* lwork, int* info
    );

    void pzgerqf_(
        const int* M, const int* N, 
        c64* A, const int* ia, const int* ja, const int* descA,
        c64* tau, c64* work, const int* lwork, int* info
    );


    // PvORMQR
    void psormqr_(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* tau, 
        float* C, const int* ic, const int* jc, const int* descC,
        float* work, const int* lwork, int* info
    );

    void pdormqr_(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* tau, 
        double* C, const int* ic, const int* jc, const int* descC,
        double* work, const int* lwork, int* info
    );


    // PvUNMQR
    void pcunmqr_(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* tau, 
        c32* C, const int* ic, const int* jc, const int* descC,
        c32* work, const int* lwork, int* info
    );

    void pzunmqr_(
        const char* side, const char* trans,
        const int* M, const int* N, const int* K,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* tau, 
        c64* C, const int* ic, const int* jc, const int* descC,
        c64* work, const int* lwork, int* info
    );


    // PvORGQR
    void psorgqr_(
        const int* M, const int* N, const int* K,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* tau, float* work, const int* lwork, int* info
    );

    void pdorgqr_(
        const int* M, const int* N, const int* K,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* tau, double* work, const int* lwork, int* info
    );


    // PvUNGQR
    void pcungqr_(
        const int* M, const int* N, const int* K,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* tau, c32* work, const int* lwork, int* info
    );

    void pzungqr_(
        const int* M, const int* N, const int* K,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* tau, c64* work, const int* lwork, int* info
    );


    // PvORGR2
    void psorgr2_(
        const int* M, const int* N, const int* K,
        const float* A, const int* ia, const int* ja, const int* descA,
        const float* tau, float* work, const int* lwork, int* info
    );

    void pdorgr2_(
        const int* M, const int* N, const int* K,
        const double* A, const int* ia, const int* ja, const int* descA,
        const double* tau, double* work, const int* lwork, int* info
    );


    // PvUNGQR
    void pcungr2_(
        const int* M, const int* N, const int* K,
        const c32* A, const int* ia, const int* ja, const int* descA,
        const c32* tau, c32* work, const int* lwork, int* info
    );

    void pzungr2_(
        const int* M, const int* N, const int* K,
        const c64* A, const int* ia, const int* ja, const int* descA,
        const c64* tau, c64* work, const int* lwork, int* info
    );


    // PvDOT
    void psdot_(
        const int* n,
        float* dot,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    void pddot_(
        const int* n,
        double* dot,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );


    // PvDOTU
    void pcdotu_(
        const int* n,
        c32* dotu,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    void pzdotu_(
        const int* n,
        c64* dotu,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );


    // PvDOTC
    void pcdotc_(
        const int* n,
        const c32* dotc,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    )
    ;
    void pzdotc_(
        const int* n,
        c64* dotc,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );


    // PvSCAL
    void psscal_(
        const int* n,
        const float* alpha,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pdscal_(
        const int* n,
        const double* alpha,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pcscal_(
        const int* n,
        const float* alpha,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pzscal_(
        const int* n,
        const double* alpha,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    );


    // PvCOPY
    void pscopy_(
        const int* n,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pdcopy_(
        const int* n,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pccopy_(
        const int* n,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pzcopy_(
        const int* n,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );


    // PvAXPY
    void psaxpy_(
        const int* n, 
        const float* alpha, 
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pdaxpy_(
        const int* n, 
        const double* alpha, 
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pcaxpy_(
        const int* n, 
        const float* alpha, 
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pzaxpy_(
        const int* n, 
        const double* alpha, 
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );


    // PvSWAP
    void psswap_(
        const int* n,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const float* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pdswap_(
        const int* n,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const double* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pcswap_(
        const int* n,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c32* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );
    void pzswap_(
        const int* n,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX,
        const c64* Y, const int* iy, const int* jy, const int* descY, const int* incY
    );

    // PvNRM2
    void psnrm2_(
        const int* n,
        float* norm2,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pdnrm2_(
        const int* n,
        double* norm2,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pscnrm2_(
        const int* n,
        float* norm2,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pdznrm2_(
        const int* n,
        double* norm2,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    // PvASUM
    void psasum_(
        const int* n,
        float* asum,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pdasum_(
        const int* n,
        double* asum,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pscasum_(
        const int* n,
        float* asum,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pdzasum_(
        const int* n,
        double* asum,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    // PvAMAX
    void psamax_(
        const int* n,
        float* amax, int* indx,
        const float* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pdamax_(
        const int* n,
        double* amax, int* indx,
        const double* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pcamax_(
        const int* n,
        c32* amax, int* indx,
        const c32* X, const int* ix, const int* jx, const int* descX, const int* incX
    );
    void pzamax_(
        const int* n,
        c64* amax, int* indx,
        const c64* X, const int* ix, const int* jx, const int* descX, const int* incX
    );

    // PvLACPY
    void pslacpy_(
        const char* uplo, const int* M, const int* N, 
        const float* A, const int* ia, const int* ja, const int* descA, 
        float* B, const int* ib, const int* jb, const int* descB
    );
    void pdlacpy_(
        const char* uplo, const int* M, const int* N, 
        const double* A, const int* ia, const int* ja, const int* descA, 
        double* B, const int* ib, const int* jb, const int* descB
    );
    void pclacpy_(
        const char* uplo, const int* M, const int* N, 
        const c32* A, const int* ia, const int* ja, const int* descA, 
        c32* B, const int* ib, const int* jb, const int* descB
    );
    void pzlacpy_(
        const char* uplo, const int* M, const int* N, 
        const c64* A, const int* ia, const int* ja, const int* descA, 
        c64* B, const int* ib, const int* jb, const int* descB
    );
}
