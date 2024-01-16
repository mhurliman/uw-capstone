
#include <iostream>
#include <math.h>
#include <iomanip>

#include <DistributedMatrix.h>

// Start with a square grid (sqrt(p)) and 
int2 CalcProcessorGridDims(int p)
{
    int d1 = static_cast<int>(sqrtf(p));
    for (; p % d1; --d1);

    return int2{ .row = d1, .col = p / d1 };
}

void TestPvHEEV(int context, int n, int nb, int seed)
{
    auto H = DistributedMatrix<c64>::RandomHermitian(context, n, { nb, nb }, seed);
    auto Z = DistributedMatrix<c64>::Uninitialized(H);
    auto Wl = LocalMatrix<double>::Uninitialized({n, 1});

    auto M = DistributedMatrix<c64>::Duplicate(H);
    PvHEEV(M, Wl, Z); // Solve for eigenvalues of H

    PvGEMM(1.0, H, Z, 0.0, M); // Compute HZ

    // Compute eZ
    auto Zp = DistributedMatrix<c64>::Duplicate(Z);
    for (int i = 0; i < n; ++i)
    {
        auto Zi = Zp.SubmatrixColumn(i);
        PvSCAL(Wl[i], Zi);
    }

    // Find error in HZ - eZ = 0
    auto T = LocalMatrix<double>::Uninitialized({n, 1});
    for (int i = 0; i < n; ++i)
    {
        auto Zi = Zp.SubmatrixColumn(i);
        auto Mi = M.SubmatrixColumn(i);
        PvAXPY(-1.0, Zi, Mi);

        T[i] = PvNRM2(Mi);
    }

    // Print error for each vector of Z
    if (H.IsRootProcess())
    {
        std::cout << T << std::endl;
    }

    PvGEMM(TRANS_OPT_NONE, TRANS_OPT_CONJ_TRANS, 1.0, Z, Z, 0.0, M);

    std::cout << M;
}

void TestQQtError(int context, int n, int nb, int seed)
{
    auto H = DistributedMatrix<c64>::RandomHermitian(context, n, { nb, nb }, seed);
    auto C = DistributedMatrix<c64>::Uninitialized(context, { n, n }, { nb, nb });

    PvGEMM(TRANS_OPT_NONE, TRANS_OPT_CONJ_TRANS, 1.0, H, H, 0.0, C);

    //Find error in HZ - eZ = 0
    auto T = LocalMatrix<double>::Uninitialized({n, 1});
    for (int i = 0; i < n; ++i)
    {
        auto Ci = C.SubmatrixColumn(i);
        T[i] = PvNRM2(Ci);
    }

    // Print error for each vector of Z
    if (H.IsRootProcess())
    {
        std::cout << T << std::endl;
    }
}

void TestPvGESRQ(int context, int pc, int pid, int n, int nb)
{
    // Define block sizes
    auto A = DistributedMatrix<double>::UniformRandom(context, { n, n }, { nb, nb }, 1);
    // auto A = DistributedMatrix<double>::Initialized(context, {6, 4}, {3, 3}, 
    // { 
    //     -0.57, -1.28, -0.39,  0.25,
    //     -1.93,  1.08, -0.31, -2.14,
    //      2.30,  0.24,  0.40, -0.35,
    //     -1.93,  0.64, -0.66,  0.08,
    //      0.15,  0.30,  0.15, -2.13,
    //     -0.02,  1.03, -1.43,  0.50,
    // });
    auto C = DistributedMatrix<double>::Uninitialized(context, A.Dims(), A.BlockSize());

    //auto D = DistributedMatrix<double>::Duplicate(A);

    auto Tau = DistributedMatrix<double>::Zeros(context, A.Dims().row, A.BlockSize().row);

    PvGERQF(A, Tau);
    //Tau.PrintLocal();

    //D.PrintGlobal();

    PvORGR2(A, Tau);

    A.PrintGlobal();

    // for (int i = 0; i < pc; ++i)
    // {
    //     if (pid == i)
    //     {
    //         std::cout << pid << ": ";
    //         Tau.PrintLocal(std::cout);
    //         std::cout.flush();
    //     }
    //     Cblacs_barrier(context, "All");
    // }
}

int main(int argc, char* argv[])
{
    int n = 7;
    int nb = 4;
    if (argc > 2)
    {
        n = std::atoi(argv[1]);
        nb = std::atoi(argv[2]);
    }

    // Grab MPI IDs
    MPI_Init(&argc, &argv);

    int pc;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &pc));

    int pid;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &pid));

    int nameLen;
    char procName[MPI_MAX_PROCESSOR_NAME]{};
    MPI_CHECK(MPI_Get_processor_name(procName, &nameLen));

    //std::cout << "Process " << pid << " on " << procName << std::endl;

    // Initialize the process grid
    int context;
    Cblacs_get(-1, 0, &context);

    int2 pgId;
    int2 pgDims = CalcProcessorGridDims(pc);
    Cblacs_gridinit(&context, "R", pgDims.row, pgDims.col);

    //TestPvHEEV(context, n, nb, 0);
    //TestQQtError(context, n, nb, 0);
    TestPvGESRQ(context, pc, pid, n, nb);

    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
