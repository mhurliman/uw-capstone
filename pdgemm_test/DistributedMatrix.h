#pragma once

#include <functional>
#include <ostream>

#include "pblas.h"

#define MIN(a, b) (a < b ? a : b)

union int2 
{
    struct { int x, y; };
    struct { int col, row; };

    bool isRoot() const { return col == 0 && row == 0; }
    int totalProcs() const { return col * row; }
    int flat(int lld) const { return RMIDX(row, col, lld); }
};

struct MatDesc
{
    int dtype;      // Descriptor type
    int ctxt;       // Cblas context
    int M, N;       // Row/column dimensions
    int Mb, Nb;     // Row/column block size
    int rsrc, crsc; // Row/column of process grid over which global matrix is distributed.
    int lld;        // Leading dimension of the local array

    operator int*(void) { return &dtype; }
    operator int const*(void) const { return &dtype; }
};

void ScatterMatrix(int2 id, int2 pgDims, const double* global, const MatDesc& desc, double* local, int2 localDims);
void GatherMatrix(int2 id, int2 pgDims, double* global, const MatDesc& desc, const double* local, int2 localDims);


// 1D-BC Local to Global Index Transform
inline int LocalToGlobal(int localIndex, int pgDim, int pgId, int blockSize)
{
    return (localIndex % blockSize) + (localIndex / blockSize) * pgDim * blockSize + (pgId * blockSize);
}

// 2D-BC Local to Global Index Transform
inline int2 LocalToGlobal(int localRowIdx, int localColIdx, int2 pgDims, int2 pgId, const MatDesc& desc)
{
    return {
        LocalToGlobal(localColIdx, pgDims.col, pgId.col, desc.Nb),
        LocalToGlobal(localRowIdx, pgDims.row, pgId.row, desc.Mb)
    };
}

// 1D-BC Global to Local Index Transform
inline void GlobalToLocal(int globalIndex, int pgDim, int blockSize, int& pgId, int& localIndex)
{
    pgId = (globalIndex / blockSize) % pgDim;
    localIndex = (globalIndex / (pgDim * blockSize)) * blockSize + globalIndex % blockSize;
}

// 2D-BC Global to Local Index Transform
inline void GlobalToLocal(int globalRowIdx, int globalColIdx, int2 pgDims, const MatDesc& desc, int2& pgId, int2& localId)
{
    GlobalToLocal(globalColIdx, pgDims.col, desc.Nb, pgId.col, localId.col);
    GlobalToLocal(globalRowIdx, pgDims.row, desc.Mb, pgId.row, localId.row);
}

class DistributedMatrix
{
public:

    // Delete copy semantics
    DistributedMatrix(const DistributedMatrix&) = delete;
    DistributedMatrix(DistributedMatrix&&) = default;
    DistributedMatrix& operator=(const DistributedMatrix&) = delete;
    DistributedMatrix& operator=(DistributedMatrix&&) = default;

    virtual ~DistributedMatrix(void)
    {
        Free(LocalData);
    }

    const double* Data(void) const { return LocalData; }
    double* Data(void) { return LocalData; }
    const MatDesc& Desc(void) const { return m_desc; }

    bool IsRootProcess() const { return PGridId.isRoot(); }

    using InitializerFunc = std::function<double(int2)>;
    void SetElements(InitializerFunc f);

    using CustomOpFunc = std::function<void(int2, double&)>;
    void CustomLocalOp(CustomOpFunc f);

    void PrintLocal(std::ostream& os) const;
    void PrintGlobal(std::ostream& os) const;

public:
    static DistributedMatrix Uninitialized(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );

    static DistributedMatrix Initialized(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize,
        InitializerFunc f
    );

    static DistributedMatrix Identity(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );

    static DistributedMatrix UniformRandom(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize,
        int seed, float range
    );

    static DistributedMatrix UniformRandomHermitian(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize,
        int seed
    );

    static void GEMM(double alpha, const DistributedMatrix& A, const DistributedMatrix& B, double beta, DistributedMatrix& C);

private:
    DistributedMatrix() = default;

    void Init(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );

    template <typename T>
    static T* Allocate(int numRows, int numCols)
    {
        return static_cast<T*>(calloc(numRows * numCols, sizeof(T)));
    }

    template <typename T>
    static void Free(T* localData)
    {
        free(localData);
    }

private:
    int2 PGridDims;
    int2 PGridId;
    MatDesc m_desc;
    double* LocalData;
    int2 LocalDims;

    static const int s_Zero = 0;
};

std::ostream& operator<<(std::ostream& os, const DistributedMatrix& m);
