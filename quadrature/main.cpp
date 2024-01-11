
#include <iostream>
#include <math.h>
#include <iomanip>

#include <DistributedMatrix.h>

int2 CalcProcDims(int p)
{
    int d1 = static_cast<int>(ceilf(sqrtf(p)));
    for (; p % d1; --d1);
    int d2 = p / d1;

    return int2{ .col = std::max(d1, d2), .row = std::min(d1, d2) };
}

// Define block sizes
const int n = 100;
const int nb = 10;


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

    // Initialize the process grid
    int context;
    Cblacs_get(-1, 0, &context);

    int2 id;
    int2 procDims = CalcProcDims(p);
    Cblacs_gridinit(&context, "R", procDims.row, procDims.col);
    Cblacs_gridinfo(context, &procDims.row, &procDims.col, &id.row, &id.col);



    Cblacs_gridexit(context);
    MPI_Finalize();

    return 0;
}
