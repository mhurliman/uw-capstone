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

template <typename T>
void ScatterMatrix(int2 id, int2 pgDims, const T* global, const MatDesc& desc, T* local, int2 localDims);

template <typename T>
void GatherMatrix(int2 id, int2 pgDims, T* global, const MatDesc& desc, const T* local, int2 localDims);


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


template <typename T>
class DistributedMatrix
{
public:
    virtual ~DistributedMatrix(void)
    {
        Free(LocalData);
    }

    T* Data(void) { return static_cast<T*>(LocalData); }
    const T* Data(void) const { return static_cast<T*>(LocalData); }

    int NumRows(void) const { return m_desc.M; }
    int NumCols(void) const { return m_desc.N; }
    bool IsSquare(void) const { return NumRows() == NumCols(); }
    const MatDesc& Desc(void) const { return m_desc; }

    int2 ProcGridDimensions(void) const { return PGridDims; }
    int2 ProcGridId(void) const { return PGridId; }

    bool IsRootProcess() const { return PGridId.isRoot(); }

    using InitializerFunc = std::function<T(int2)>;
    using ModifierFunc = std::function<T(int2, T&)>;
    using CustomOpFunc = std::function<void(int2, T&)>;

    void SetElements(InitializerFunc f);
    void ModifyElements(ModifierFunc f);
    void CustomLocalOp(CustomOpFunc f);

    void PrintLocal(std::ostream& os) const;
    void PrintGlobal(std::ostream& os) const;

    void PrintLocal() const { PrintLocal(std::cout); }
    void PrintGlobal() const { PrintGlobal(std::cout); }

public:
    static DistributedMatrix<T> Uninitialized(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );
    
    template <typename U>
    static DistributedMatrix<T> Uninitialized(const DistributedMatrix<U>& copyDims);

    static DistributedMatrix<T> Initialized(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize,
        InitializerFunc f
    );

    static DistributedMatrix<T> Identity(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );

    static DistributedMatrix<T> UniformRandom(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize,
        int seed, double range = 1.0f
    );

    static DistributedMatrix<T> RandomHermitian(
        int context, int2 pgDims, int2 pgId,
        int N, int rowBlockSize, int colBlockSize, int seed
    );
    
    static DistributedMatrix<T> Duplicate(const DistributedMatrix<T>& A);
    static void Duplicate(const DistributedMatrix<T>& A, DistributedMatrix<T>& B);

    // Operations
    static void GEMM(double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, double beta, DistributedMatrix<T>& C);
    
    template <typename U>
    static void HEEV(DistributedMatrix<T>& A, DistributedMatrix<U>& W, DistributedMatrix<T>& Z);

    template <typename U>
    static void HEEV(DistributedMatrix<T>& A, DistributedMatrix<U>& W);

    // Delete copy semantics
    DistributedMatrix(const DistributedMatrix&) = delete;
    DistributedMatrix& operator=(const DistributedMatrix&) = delete;

    // Enable move semantics
    DistributedMatrix(DistributedMatrix&&) = default;
    DistributedMatrix& operator=(DistributedMatrix&&) = default;


    void Init(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );
private:
    DistributedMatrix(void) : LocalData{} {}

    template <typename U>
    static U* Allocate(int count)
    {
        return static_cast<U*>(calloc(count, sizeof(U)));
    }

    template <typename U>
    static U* Allocate(int numRows, int numCols)
    {
        return static_cast<U*>(calloc(numRows * numCols, sizeof(U)));
    }

    template <typename U>
    static void Free(U* localData)
    {
        free(localData);
    }

private:
    int2     PGridDims;
    int2     PGridId;
    MatDesc  m_desc;
    T*       LocalData;
    int2     LocalDims;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const DistributedMatrix<T>& m)
{
    m.PrintGlobal(os);
    return os;
}

#include "DistributedMatrix.inl"