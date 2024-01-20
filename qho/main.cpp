
#include <iostream>
#include <math.h>
#include <iomanip>

#include <DistributedMatrix.h>

// Generate as close to a square grid as possible
// Prime values aren't great candidates here...
int2 CalcProcessorGridDims(int p)
{
    int d1 = static_cast<int>(sqrtf(p));
    for (; p % d1; --d1);

    return int2{ .row = d1, .col = p / d1 };
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

    std::cout << "Process " << pid << " on " << procName << std::endl;

    // Initialize the process grid
    int context;
    Cblacs_get(-1, 0, &context);

    int2 pgId;
    int2 pgDims = CalcProcessorGridDims(pc);
    Cblacs_gridinit(&context, "R", pgDims.row, pgDims.col);

    // HERE

    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
