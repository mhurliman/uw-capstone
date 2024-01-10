#pragma once

#include <assert.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory>
#include <ostream>

#include "pblas.h"

#include "Random.h"

struct int2 
{
    int col, row;

    bool IsRoot(void) const { return col == 0 && row == 0; }
    int NumProcs(void) const { return col * row; }
    int Flat(int lld) const { return RMIDX(row, col, lld); }

    static int2 One(void) { return int2 { 1, 1 }; }
    static int2 OneBased(int col, int row) { return int2 { col + 1, row + 1 }; }
};

static int2 operator+(const int2& a, const int2& b) 
{
    return int2{ .col = a.col + b.col, .row = a.row + b.row };
}

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


int2 LocalToGlobal(int localRowIdx, int localColIdx, int2 pgDims, int2 pgId, const MatDesc& desc);
void GlobalToLocal(int globalRowIdx, int globalColIdx, int2 pgDims, const MatDesc& desc, int2& pgId, int2& localId);

template <typename T>
class LocalMatrix
{
public:
    T* Data(void) { return m_data.get(); }
    const T* Data(void) const { return m_data.get(); }

    int2 Dims(void) const { return m_dims; }

    inline const int& NumRows(void) const { return m_dims.row; }
    inline const int& NumCols(void) const { return m_dims.col; }

    void Print(std::ostream& os) const;

    using InitializerFunc = std::function<T(int2)>;
    void SetElements(InitializerFunc f);

    using CustomOpFunc = std::function<void(int2, T&)>;
    void CustomOp(CustomOpFunc f);

    template <typename U>
    static LocalMatrix<T> Uninitialized(const LocalMatrix<U>& dimsSpec);
    static LocalMatrix<T> Uninitialized(int numRows, int numCols);

    template <typename U>
    static LocalMatrix<T> Initialized(const LocalMatrix<U>& data);
    static LocalMatrix<T> Initialized(int numRows, int numCols, InitializerFunc f);

    inline T& operator[](int index) { return m_data[index]; }
    inline T operator[](int index) const { return m_data[index]; }

private:
    LocalMatrix(void) = default;

    void Init(int numRows, int numCols);

private:
    std::unique_ptr<T[]> m_data;
    int2                 m_dims;

    template <typename U>
    friend class LocalMatrix;

    template <typename U>
    friend class DistributedMatrix;
};


template <typename T>
class DistributedMatrix
{
public:
    inline T* Data(void) { return m_localData.get(); }
    inline const T* Data(void) const { return m_localData.get(); }

    inline const int& NumRows(void) const { return m_subDims.row; }
    inline const int& NumCols(void) const { return m_subDims.col; }

    inline const int& IndexRow(void) const { return m_subIndex.row; }
    inline const int& IndexCol(void) const { return m_subIndex.col; }

    inline const int2& Dims(void) const { return m_subDims; }
    inline const int2& Index(void) const { return m_subIndex; }

    inline bool IsSquare(void) const { return NumRows() == NumCols(); }
    inline const MatDesc& Desc(void) const { return m_desc; }

    inline int2 ProcGridDimensions(void) const { return m_PGridDims; }
    inline int2 ProcGridId(void) const { return m_PGridId; }

    inline bool IsRootProcess() const { return m_PGridId.IsRoot(); }

    LocalMatrix<T> ToLocalMatrix(void) const;

    DistributedMatrix<T> Submatrix(int numRows, int numCols, int rowIndex, int colIndex) const;
    DistributedMatrix<T> SubmatrixRow(int rowIndex) const;
    DistributedMatrix<T> SubmatrixColumn(int colIndex) const;

    using InitializerFunc = std::function<T(int2)>;
    void SetElements(InitializerFunc f);

    using CustomOpFunc = std::function<void(int2, T&)>;
    void CustomLocalOp(CustomOpFunc f);

    void PrintLocal(std::ostream& os) const;
    void PrintGlobal(std::ostream& os) const;

    void PrintLocal(void) const { PrintLocal(std::cout); }
    void PrintGlobal(void) const { PrintGlobal(std::cout); }

    static DistributedMatrix<T> Uninitialized(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );

    template <typename U>
    static DistributedMatrix<T> Uninitialized(
        int context, int2 pgDims, int2 pgId, 
        int rowBlockSize, int colBlockSize, const LocalMatrix<U>& data
    );

    template <typename U>
    static DistributedMatrix<T> Uninitialized(const DistributedMatrix<U>& dimsSpec);

    static DistributedMatrix<T> Initialized(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize,
        InitializerFunc f
    );

    template <typename U>
    static DistributedMatrix<T> Initialized(
        int context, int2 pgDims, int2 pgId, 
        int rowBlockSize, int colBlockSize, const LocalMatrix<U>& data
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

    template <typename U>
    static DistributedMatrix<T> Diagonal(
        int context, int2 pgDims, int2 pgId,
        int rowBlockSize, int colBlockSize, const LocalMatrix<U>& X
    );

    static DistributedMatrix<T> Duplicate(const DistributedMatrix<T>& A);
    static void Duplicate(const DistributedMatrix<T>& A, DistributedMatrix<T>& B);

private:
    DistributedMatrix(void) = default;

    void Init(
        int context, int2 pgDims, int2 pgId, 
        int numRows, int numCols, int rowBlockSize, int colBlockSize
    );

private:
    int2                 m_PGridDims;
    int2                 m_PGridId;

    MatDesc              m_desc;
    std::shared_ptr<T[]> m_localData;
    int2                 m_localDims;

    int2                 m_subDims;
    int2                 m_subIndex;

    template <typename U>
    friend class DistributedMatrix;
};

// Templated operations
template <typename T>
void PvGEMM(
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, 
    double beta, DistributedMatrix<T>& C
);

template <typename T>
void PvGEMV(
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& X, 
    double beta, DistributedMatrix<T>& Y
);

template <typename T>
void PvHEEV(DistributedMatrix<T>& A, LocalMatrix<ValueType<T>>& W, DistributedMatrix<T>& Z);

template <typename T>
void PvHEEV(DistributedMatrix<T>& A, LocalMatrix<ValueType<T>>& W);

template <typename T>
void PvTRAN(double a, const DistributedMatrix<T>& A, double b, DistributedMatrix<T>& C);

template <typename T>
void PvTRANU(double a, const DistributedMatrix<T>& A, double b, DistributedMatrix<T>& C);

template <typename T>
void PvTRANC(double a, const DistributedMatrix<T>& A, double b, DistributedMatrix<T>& C);

template <typename T>
float PvDOT(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y);

template <typename T>
T PvDOTU(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y);

template <typename T>
T PvDOTC(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y);

template <typename T>
void PvSCAL(ValueType<T> a, const DistributedMatrix<T>& X);

template <typename T>
void PvSWAP(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y);

template <typename T>
void PvLACPY(const DistributedMatrix<T>& A, DistributedMatrix<T>& B);

template <typename T>
ValueType<T> PvNRM2(const DistributedMatrix<T>& X);

template <typename T>
ValueType<T> PvASUM(const DistributedMatrix<T>& X);

template <typename T>
void PvAMAX(T& max, int& index, const DistributedMatrix<T>& X);

template <typename T>
void PvCOPY(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y);

template <typename T>
void PvAXPY(ValueType<T> a, const DistributedMatrix<T>& X, DistributedMatrix<T>& Y);


// Output Stream Helpers

std::ostream& operator<<(std::ostream& os, const int2& d);
std::ostream& operator<<(std::ostream& os, const MatDesc& d);

template <typename T>
std::ostream& operator<<(std::ostream& os, const LocalMatrix<T>& m);
template <typename T>
std::ostream& operator<<(std::ostream& os, const DistributedMatrix<T>& m);

#include "DistributedMatrix.inl"