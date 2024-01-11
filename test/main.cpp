
#include <iostream>
#include <math.h>
#include <iomanip>

#include <DistributedMatrix.h>

// Start with a square grid (sqrt(p)) and 
int2 CalcProcessorGridDims(int p)
{
    int d1 = static_cast<int>(ceilf(sqrtf(p)));
    for (; p % d1; --d1);

    return int2{ .col = p / d1, .row = d1 };
}

void TestPvHEEV(int context, int pid)
{
    // Define block sizes
    const int n = 100;
    const int nb = 20;

    auto H = DistributedMatrix<c64>::RandomHermitian(context, n, nb, nb, 1);
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
    if (pid == 0)
    {
        std::cout << T << std::endl;
    }
}

void TestPvGESRQ(int context, int pid)
{
    // Define block sizes
    const int n = 10;
    const int nb = 20;

    Cblacs_barrier(context, "All");

    //auto A = DistributedMatrix<double>::UniformRandom(context, n, n, nb, nb, 1);
    auto A = DistributedMatrix<c64>::UniformRandom(context, n, n, nb, nb, 1);
    auto C = DistributedMatrix<c64>::Identity(context, n, n, nb, nb);
    auto Tau = LocalMatrix<c64>::Uninitialized(n, 1);

    PvGERQF(A, Tau);
    PvUNMQR(A, Tau, C);

    auto I = DistributedMatrix<c64>::Identity(context, n, n, nb, nb);
    PvGEMM(GEMM_OPT_NONE, GEMM_OPT_CONJ_TRANS, 1.0, C, C, 0.0, I);

    std::cout << I << std::endl;
}

int main(int argc, char* argv[])
{
    // Grab MPI IDs
    MPI_Init(&argc, &argv);

    int pc;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &pc));

    int pid;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &pid));

    int nameLen;
    char procName[MPI_MAX_PROCESSOR_NAME]{};
    MPI_CHECK(MPI_Get_processor_name(procName, &nameLen));

    std::cout << "Process " << pid << " on " << procName << std::endl;

    // Initialize the process grid
    int context;
    Cblacs_get(-1, 0, &context);

    int2 pgId;
    int2 pgDims = CalcProcessorGridDims(pc);
    Cblacs_gridinit(&context, "R", pgDims.row, pgDims.col);

    //TestPvHEEV(context, pid);
    TestPvGESRQ(context, pid);
    
    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
