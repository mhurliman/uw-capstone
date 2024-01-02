
#include <iostream>
#include <math.h>

#include "DistributedMatrix.h"

int M = 16, N = 20, L = 32;
int MBSIZE = 4, NBSIZE = 4, LBSIZE = 11;

int main(int argc, char* argv[])
{
    // Grab MPI IDs
    MPI_Init(&argc, &argv);

    int p;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));

    int myid;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

    int nameLen;
    char procName[MPI_MAX_PROCESSOR_NAME]{};
    MPI_CHECK(MPI_Get_processor_name(procName, &nameLen));

    std::cout << "Process " << myid << " on " << procName << std::endl;

    int context;
    Cblacs_get(-1, 0, &context);

	int2 procDims;
    procDims.col = ceilf(sqrtf(p));
    for (; p % procDims.col; --procDims.col);

	procDims.row = p / procDims.col;
	if (procDims.row > procDims.col)
	{
		std::swap(procDims.col, procDims.row);
	}

    std::flush(std::cout);

	// Initialize the pr x pc process grid
	int2 id;
	Cblacs_gridinit(&context, "R", procDims.row, procDims.col);
	Cblacs_gridinfo(context, &procDims.row, &procDims.col, &id.row, &id.col);

	// Define block sizes
    auto Af = DistributedMatrix::Initialized(context, procDims, id, M, L, MBSIZE, LBSIZE, [](int2 gid) { return gid.col <= gid.row; });
    auto Bf = DistributedMatrix::Initialized(context, procDims, id, L, N, LBSIZE, NBSIZE, [](int2 gid) { return gid.col; });
    auto Cf = DistributedMatrix::Uninitialized(context, procDims, id, M, N, MBSIZE, NBSIZE);

    std::cout << Af;
    std::cout << Bf;

    DistributedMatrix::GEMM(1.0, Af, Bf, 0.0, Cf);

    std::cout << Cf;

	// Check for correctness
	float err = 0.0;
    Cf.CustomLocalOp([&](int2 gid, double& v) { err += abs(v) - (gid.row + 1) * gid.col; });

	printf("Local error on proc %d = %.2f\n", myid, err);

	Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
