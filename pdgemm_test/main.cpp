
#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include <mpi.h>
#include "MemoryArena.h"

using namespace std;

#define MPI_CHECK(x) do { \
    int errCode = x; \
    if (errCode != MPI_SUCCESS) { \
        int errLen = 0; \
        char errStr[MPI_MAX_ERROR_STRING] {'\0'}; \
        MPI_Error_string(errCode, errStr, &errLen); \
        printf("%s", errStr); \
        assert(x == MPI_SUCCESS); \
    } \
} while(0)

void pdgemm_(char* TRANSA, char* TRANSB,
            int* M, int* N, int* K,
            double* ALPHA,
            double* A, int* IA, int* JA, int* DESCA,
            double* B, int* IB, int* JB, int* DESCB,
            double* BETA,
            double* C, int* IC, int* JC, int* DESCC );

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int procNum;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &procNum));

    int procRank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &procRank));

    cout << "Hello world, I have  rank " << procRank << " out of " 
         << procNum << endl;

    MPI_Finalize();

    return 0;
}
