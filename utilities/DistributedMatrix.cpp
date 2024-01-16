#include "DistributedMatrix.h"


// 1D-BC Local to Global Index Transform
int LocalToGlobal(int localIndex, int pgDim, int pgId, int blockSize)
{
    return (localIndex % blockSize) + (localIndex / blockSize) * pgDim * blockSize + (pgId * blockSize);
}

// 2D-BC Local to Global Index Transform
int2 LocalToGlobal(int localRowIdx, int localColIdx, int2 pgDims, int2 pgId, const MatDesc& desc)
{
    return int2 {
        .row = LocalToGlobal(localRowIdx, pgDims.row, pgId.row, desc.Mb),
        .col = LocalToGlobal(localColIdx, pgDims.col, pgId.col, desc.Nb)
    };
}

// 1D-BC Global to Local Index Transform
void GlobalToLocal(int globalIndex, int pgDim, int blockSize, int& pgId, int& localIndex)
{
    pgId = (globalIndex / blockSize) % pgDim;
    localIndex = (globalIndex / (pgDim * blockSize)) * blockSize + globalIndex % blockSize;
}

// 2D-BC Global to Local Index Transform
void GlobalToLocal(int globalRowIdx, int globalColIdx, int2 pgDims, const MatDesc& desc, int2& pgId, int2& localId)
{
    GlobalToLocal(globalRowIdx, pgDims.row, desc.Mb, pgId.row, localId.row);
    GlobalToLocal(globalColIdx, pgDims.col, desc.Nb, pgId.col, localId.col);
}


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

std::ostream& operator<<(std::ostream& os, const int2& d)
{
    return os <<"(" << d.row << ", " << d.col << ")";
}

std::ostream& operator<<(std::ostream& os, const MatDesc& d)
{
    return os <<"MatDesc - " << d.M << " x " << d.N << ", LLD = " << d.lld;
}