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

std::ostream& operator<<(std::ostream& os, c32 x)
{
    bool pos = x.imag() >= 0.0;
    const char* sign = pos ? " + " : " - ";
    float posImag = (float(pos) * 2 - 1)  * x.imag();
    return os << std::setw(4) << std::setprecision(2) << "(" << x.real() << sign << posImag << "i)";
}

std::ostream& operator<<(std::ostream& os, c64 x)
{
    bool pos = x.imag() >= 0.0;
    const char* sign = pos ? " + " : " - ";
    float posImag = (float(pos) * 2 - 1)  * x.imag();
    return os << std::setw(4) << std::setprecision(2) << "(" << x.real() << sign << posImag << "i)";
}

extern "C" {
    /* Cblacs declarations */
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

    void pdtran_(const int* M, const int* N, const double* alpha, const double* A, const int* ia, const int* ja, const int* descA, const double* beta, double* C, const int* ic, const int* jc, const int* descC);
    void pztranu_(const int* M, const int* N, const double* alpha, const c64* A, const int* ia, const int* ja, const int* descA, const double* beta, c64* C, const int* ic, const int* jc, const int* descC);
    void pztranc_(const int* M, const int* N, const double* alpha, const c64* A, const int* ia, const int* ja, const int* descA, const double* beta, c64* C, const int* ic, const int* jc, const int* descC);

    void pzlacpy_(const char* UPLO, const int* M, const int* N, const c64* A, const int* ia, const int* ja, const int* descA, c64* B, const int* ib, const int* jb, const int* descB);

    /* Scalapack tools */
    int numroc_(
        const int* n, 
        const int* nb, 
        const int* iproc, 
        const int* srcproc, 
        const int* nprocs
    );

    void descinit_(
        int* desc, 
        const int* m, const int* n, 
        const int* mb, const int* nb, 
        const int* irsrc, const int* icsrc, 
        const int* ictxt, const int* lld, int* info);

    void indxg2p(int indxglob, int nb, int iproc, int isrcproc, int nprocs);
    void indxg2l(int indxglob, int nb, int iproc, int isrcproc, int nprocs);
    void indxl2g(int indxloc, int nb, int iproc, int isrcproc, int nprocs);

    // Pblas declarations

    // GEMM
    void psgemm_(const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const float* a, const int* ia, const int* ja, const int* desca,
        const float* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        float* c, const int* ic, const int* jc, const int* descc);

    void pdgemm_(const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

    void pcgemm_(const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const c32* a, const int* ia, const int* ja, const int* desca,
        const c32* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        c32* c, const int* ic, const int* jc, const int* descc);

    void pzgemm_(const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const c64* a, const int* ia, const int* ja, const int* desca,
        const c64* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        c64* c, const int* ic, const int* jc, const int* descc);

    void pcheev_(
        const char* jobz, const char* uplo, const int* n, 
        c32* a, const int* ia, const int* ja, const int* desca,
        float* w,
        c32* z, const int* iz, const int* jz, const int* descz,
        c32* work, const int* lwork, float* rwork, const int* lrwork, 
        int* info
    );

    void pzheev_(
        const char* jobz, const char* uplo, const int* n, 
        c64* a, const int* ia, const int* ja, const int* desca,
        double* w,
        c64* z, const int* iz, const int* jz, const int* descz,
        c64* work, const int* lwork, double* rwork, const int* lrwork, 
        int* info
    );
}
