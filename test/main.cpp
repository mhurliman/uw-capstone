
#include <iostream>
#include <math.h>
#include <iomanip>

#include "DistributedMatrix.h"

int2 CalcProcDims(int p)
{
    int d1 = static_cast<int>(ceilf(sqrtf(p)));
    for (; p % d1; --d1);
    int d2 = p / d1;

    return int2{ .col = std::max(d1, d2), .row = std::min(d1, d2) };
}

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

    int2 procDims = CalcProcDims(p);

    // Initialize the process grid
    int context;
    Cblacs_get(-1, 0, &context);

    int2 id;
    Cblacs_gridinit(&context, "R", procDims.row, procDims.col);
    Cblacs_gridinfo(context, &procDims.row, &procDims.col, &id.row, &id.col);

	// Define block sizes
    const int n = 8;
    const int nb = 3;

    auto H = DistributedMatrix<c64>::RandomHermitian(context, procDims, id, n, nb, nb, 1);
    auto Z = DistributedMatrix<c64>::Uninitialized(H);
    auto Wl = LocalMatrix<double>::Uninitialized(n, 1);

    auto M = DistributedMatrix<c64>::Duplicate(H);
    PvHEEV(M, Wl, Z); // Solve for eigenvalues of H

    PvGEMM(1.0, H, Z, 0.0, M); // Compute HZ

    // Compute eZ
    for (int i = 0; i < n; ++i)
    {
        auto Zi = Z.SubmatrixColumn(i);
        PvSCAL(Wl[i], Zi);
    }

    // Find error in HZ - eZ = 0
    auto T = LocalMatrix<double>::Uninitialized(n, 1);
    for (int i = 0; i < n; ++i)
    {
        auto Zi = Z.SubmatrixColumn(i);
        auto Mi = M.SubmatrixColumn(i);
        PvAXPY(-1.0, Zi, Mi);

        T[i] = PvNRM2(Mi);
    }

    // Print error for each vector of Z
    if (myid == 0)
    {
        std::cout << T << std::endl;
    }
    
    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
