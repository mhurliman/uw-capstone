
#include <iostream>
#include <math.h>
#include <iomanip>
#include <fstream>

#include <DistributedMatrix.h>

double TestHZeZError(int context, int n, int nb, int seed)
{
    auto H = DistributedMatrix<c64>::RandomHermitian(context, n, { nb, nb }, seed);
    auto Z = DistributedMatrix<c64>::Uninitialized(H);
    auto Wl = LocalMatrix<double>::Uninitialized({n, 1});

    auto M = DistributedMatrix<c64>::Duplicate(H);

    PvHEEV(M, Wl, Z); // Solve for eigenvalues of H
    PvGEMM(1.0, H, Z, 0.0, M);

    // Find error in HZ - eZ = 0
    auto T = LocalMatrix<double>::Uninitialized({n, 1});
    for (int i = 0; i < n; ++i)
    {
        auto Zi = Z.SubmatrixColumn(i);
        auto Mi = M.SubmatrixColumn(i); 
        PvAXPY(-Wl[i], Zi, Mi);

        T[i] = PvASUM(Mi) / n;
    }

    // Print error for each vector of Z
    double avgErr = T.ASum() / n;
    if (H.IsRootProcess())
    {
        std::cout << "Average error for HZ-eZ: ";
        std::cout << avgErr << std::endl;
    }

    return avgErr;
}

double TestZZhhError(int context, int n, int nb, int seed)
{
    auto H = DistributedMatrix<c64>::RandomHermitian(context, n, { nb, nb }, seed);
    auto Z = DistributedMatrix<c64>::Uninitialized(H);
    auto C = DistributedMatrix<c64>::Identity(context, { n, n }, { nb, nb });

    auto Wl = LocalMatrix<double>::Uninitialized({n, 1});

    PvHEEV(H, Wl, Z);
    PvGEMM(TRANS_OPT_NONE, TRANS_OPT_CONJ_TRANS, 1.0, Z, Z, -1.0, C);

    auto T = LocalMatrix<double>::Uninitialized({n, 1});
    for (int i = 0; i < n; ++i)
    {
        auto Ci = C.SubmatrixColumn(i);
        T[i] = PvASUM(Ci) / n;
    }

    // Print error for each vector of Z
    double avgErr = T.ASum() / n;
    if (H.IsRootProcess())
    {
        std::cout << "Average error for ZZ^H: ";
        std::cout << avgErr << std::endl;
    }

    return avgErr;
}

void TestError(int context, int pc, int pid, int n, int nb, int seed)
{
    const int testPowers = 10;
    const int testCount = testPowers * (testPowers + 1) / 2;

    char* labels[testCount]{};
    double err0[testCount]{};
    double err1[testCount]{};

    int test = 0;
    for (int m = 3; m < testPowers; ++m)
    {
        const int N = 1 << m;

        for (int b = 1; b < m; ++b, ++test)
        {
            const int Nb = 1 << b;

            if (pid == 0)
            {
                char buff[256];
                int cnt = std::snprintf(buff, sizeof(buff), "%d,%d", N, Nb);

                labels[test] = new char[cnt];
                std::strcpy(labels[test], buff);

                std::cout << "Dimension: " << N << " Block size: " << Nb << std::endl;
            }

            err0[test] = TestHZeZError(context, N, Nb, 0);
            err1[test] = TestZZhhError(context, N, Nb, 0);
        }
    }

    if (pid == 0)
    {
        std::ofstream of = std::ofstream("datafile.csv");
        if (of.is_open())
        {
            for (int i = 0; i < test; ++i)
            {
                of << labels[i];
                of << ",";
                of << err0[i];
                of << ",";
                of << err1[i];
                of << "\n";
            }
        }
    }
}

void TestQRFactorization(int context, int pc, int pid, int n, int nb)
{
    // Define block sizes
    auto A = DistributedMatrix<c64>::RandomHermitian(context, n, { nb, nb }, 1);
    auto Tau = DistributedMatrix<c64>::Uninitialized(context, A.Dims().row, A.BlockSize().row);

    PvGERQF(A, Tau);
    PvUNGR2(A, Tau);

    // Test orthonormality
    for (int i = 0; i < n; ++i)
    {
        auto Ai = A.SubmatrixColumn(i);

        // Normality
        double v = PvNRM2(Ai);
        assert((abs(v) - 1.0) < 1e-4);

        if (pid == 0)
        {
            std::cout << "||A_" << i << "|| = " << v << std::endl;
        }

        for (int j = i + 1; j < n; ++j)
        {
            auto Aj = A.SubmatrixColumn(j);

            // Orthogonality
            c64 d = PvDOTC(Ai, Aj);
            assert(std::abs(d) < 1e-4);

            if (pid == 0)
            {
                std::cout << "Dot(A_" << i << ", A_" << j << ") = " << d << std::endl;
            }
        }

        if (pid == 0)
        {
            std::cout << std::endl;
        }
    }
}

void TestOneNorm(int context, int pc, int pid, int n, int nb)
{
    auto A = DistributedMatrix<double>::Initialized(context, {n, n}, {nb, nb}, [](int2 gid){ return gid.col; });

    std::cout << A;

    auto b = A.EuclideanNorm();

    if (pid == 0)
    {
        std::cout << std::setprecision(6) << b << std::endl;
    }
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

    //TestError(context, pc, pid, n, nb, 0);
    //TestQRFactorization(context, pc, pid, n, nb);
    TestOneNorm(context, pc, pid, n, nb);

    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
