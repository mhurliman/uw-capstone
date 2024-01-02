#pragma once

#include <assert.h>
#include <mpi.h>

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

#define ROW_MAJOR_INDEX(row, col, numRows) (row + col * numRows)
#define COL_MAJOR_INDEX(row, col, numCols) (col + row * numCols)
#define RMIDX(row, col, numRows) ROW_MAJOR_INDEX(row, col, numRows)
#define CMIDX(row, col, numRows) COL_MAJOR_INDEX(row, col, numRows)

struct c64 { double re, im; };

extern "C" {
    /* Cblacs declarations */
    void Cblacs_pinfo(int* mypnum, int* nprocs);
    void Cblacs_get(int context, int what, int* val);
    void Cblacs_gridinit(int* context, const char* order, int npRow, int npCol);
    void Cblacs_pcoord(int context, int pNum, int* pRow, int* pCol);
    void Cblacs_gridinfo(int context, int* nprow, int* npcol, int* myrow, int* mycol);
    void Cblacs_gridexit(int context);
    void Cblacs_barrier(int context, const char* scope);

    void Cdgerv2d(int context, int M, int N, double* A, int lda, int rsrc, int csrc);
    void Cdgesd2d(int context, int M, int N, const double* A, int lda, int rdest, int cdest);
 
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

    /* Pblas declarations */
    void pdgemm_(const char* transa, const char* transb,
        const int* m, const int* n, const int* k,
        const double* alpha,
        const double* a, const int* ia, const int* ja, const int* desca,
        const double* b, const int* ib, const int* jb, const int* descb,
        const double* beta,
        double* c, const int* ic, const int* jc, const int* descc);

    void pzheev_(
        const char* jobz, const char* uplo, const int* n, 
        c64* a, const int* ia, const int* ja, const int* desca,
        double* w,
        c64* z, const int* iz, const int* jz, const int* descz,
        c64* work, const int* lwork, c64* rwork, const int* lrwork, 
        int* info
    );
}