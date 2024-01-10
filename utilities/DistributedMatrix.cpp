#include "DistributedMatrix.h"

// 2D-BC Local to Global Index Transform
int2 LocalToGlobal(int localRowIdx, int localColIdx, int2 pgDims, int2 pgId, const MatDesc& desc)
{
    const int zero = 0;
    return int2{
        .col = indxl2g_(&localColIdx, &desc.Nb, &pgId.col, &zero, &pgDims.col),
        .row = indxl2g_(&localRowIdx, &desc.Mb, &pgId.row, &zero, &pgDims.row)
    };
}

// 2D-BC Global to Local Index Transform
void GlobalToLocal(int globalRowIdx, int globalColIdx, int2 pgDims, const MatDesc& desc, int2& pgId, int2& localId)
{
    const int zero = 0;
    pgId = int2 {
        .col = indxg2p_(&globalColIdx, &desc.Nb, &zero, &zero, &pgDims.col),
        .row = indxg2p_(&globalRowIdx, &desc.Mb, &zero, &zero, &pgDims.row),
    };

    localId = int2 {
        .col = indxg2l_(&globalColIdx, &desc.Nb, &zero, &zero, &pgDims.col),
        .row = indxg2l_(&globalRowIdx, &desc.Mb, &zero, &zero, &pgDims.row),
    };
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