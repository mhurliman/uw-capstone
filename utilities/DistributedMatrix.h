#pragma once

#include <assert.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory>
#include <ostream>
#include <cstring>

#include "pblas.h"

#include "Random.h"

struct int2 
{
    int row;
    int col;

    bool IsRoot(void) const { return col == 0 && row == 0; }
    bool ValidIndex(void) const { return row > 0 && col > 0; }
    bool ValidDims(void) const { return row > 0 && col > 0; }

    int Count(void) const { return col * row; }
    int FlatCM(int lld) const { return CMIDX(row, col, lld); }
    int FlatRM(int lld) const { return RMIDX(row, col, lld); }

    int2 ToOneBased(void) const { return { .row = row + 1, .col = col + 1 }; }

    static int2 One(void) { return int2 { 1, 1 }; }
    static int2 OneBased(int2 dims) { return dims.ToOneBased(); }

    inline int operator[](int index) const { return (index & 0x1) == 0 ? row : col; };
    inline int2 operator-(void) const { return { .row = -row, .col = -col }; }
    inline bool operator==(int2 rhs) const { return std::memcmp(this, &rhs, sizeof(int2)) == 0; }
    inline int2 operator+(int2 rhs) const { return { .row = row + rhs.row, .col = col + rhs.col }; }
    inline int2 operator-(int2 rhs) const { return operator+(-rhs); }
};

// Generate as close to a square grid as possible
// Prime values aren't great candidates here...
inline int2 CalcProcessorGridDims(int p)
{
    int d1 = static_cast<int>(sqrtf(p));
    for (; p % d1; --d1);

    return int2{ .row = d1, .col = p / d1 };
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
    int Bytes(void) const { return NumElements() * sizeof(T); }

    inline const int& NumRows(void) const { return m_dims.row; }
    inline const int& NumCols(void) const { return m_dims.col; }
    inline int NumElements(void) const { return m_dims.Count(); }

    void Print(std::ostream& os) const;

    using InitializerFunc = std::function<T(int2)>;
    void SetElements(InitializerFunc f);

    using CustomOpFunc = std::function<void(int2, T&)>;
    void CustomOp(CustomOpFunc f);

    template <typename U>
    static LocalMatrix<T> Uninitialized(const LocalMatrix<U>& dimsSpec);
    static LocalMatrix<T> Uninitialized(int2 dims);

    template <typename U>
    static LocalMatrix<T> Initialized(const LocalMatrix<U>& data);
    static LocalMatrix<T> Initialized(int2 dims, InitializerFunc f);
    static LocalMatrix<T> Initialized(int2 dims, const std::initializer_list<T>& rowMajorList);

    inline ValueType<T> ASum(void) const;

    inline T& operator[](int index) { return m_data[index]; }
    inline T operator[](int index) const { return m_data[index]; }

    static LocalMatrix<T> Transpose(const LocalMatrix<T>& A) 
    { 
        auto B = LocalMatrix<T>::Initialized(A);
        std::swap(B.m_dims.row, B.m_dims.col); 
        return B;
    }

private:
    LocalMatrix(void) = default;

    void Init(int2 dims);

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

    inline int2 Dims(void) const { return m_subDims; }
    inline int2 Index(void) const { return m_subIndex; }
    inline int2 BlockSize(void) const { return { m_desc.Mb, m_desc.Nb };}
    inline int Bytes(void) const { return m_localDims.Count() * sizeof(T);}

    inline const MatDesc& Desc(void) const { return m_desc; }

    inline int2 ProcGridDims(void) const { return m_PGridDims; }
    inline int2 ProcGridId(void) const { return m_PGridId; }

    inline bool IsRootProcess() const { return m_PGridId.IsRoot(); }
    inline bool IsSquare(void) const { return NumRows() == NumCols(); }

    LocalMatrix<T> ToLocalMatrix(void) const;

    DistributedMatrix<T> Submatrix(int2 dims, int2 indices) const;
    DistributedMatrix<T> SubmatrixRow(int rowIndex) const;
    DistributedMatrix<T> SubmatrixColumn(int colIndex) const;

    ValueType<T> OneNorm(void) const;
    ValueType<T> InfinityNorm(void) const;
    ValueType<T> EuclideanNorm(void) const;

    using InitializerFunc = std::function<T(int2)>;
    void SetElements(InitializerFunc f);

    using CustomOpFunc = std::function<void(int2, T&)>;
    void CustomLocalOp(CustomOpFunc f);

    void PrintLocal(std::ostream& os) const;
    void PrintGlobal(std::ostream& os) const;

    void PrintLocal(void) const { PrintLocal(std::cout); }
    void PrintGlobal(void) const { PrintGlobal(std::cout); }

    static DistributedMatrix<T> Uninitialized(int context, int numRows, int blockSize);
    static DistributedMatrix<T> Uninitialized(int context, int2 dims, int2 blockSize);

    template <typename U>
    static DistributedMatrix<T> Uninitialized(int context, int2 blockSize, const LocalMatrix<U>& data);

    template <typename U>
    static DistributedMatrix<T> Uninitialized(const DistributedMatrix<U>& dimsSpec);
    
    static DistributedMatrix<T> Zeros(int context, int numRows, int blockSize);
    static DistributedMatrix<T> Zeros(int context, int2 dims, int2 blockSize);

    template <typename U>
    static DistributedMatrix<T> Initialized(int context, int2 blockSize, const LocalMatrix<U>& data);
    static DistributedMatrix<T> Initialized(int context, int2 dims, int2 blockSize, InitializerFunc f);
    static DistributedMatrix<T> Initialized(int context, int2 dims, int2 blockSize, const std::initializer_list<T>& rowMajorList);

    static DistributedMatrix<T> Identity(int context, int2 dims, int2 blockSize);
    static DistributedMatrix<T> UniformRandom(int context, int2 dims, int2 blockSize, int seed, double range = 1.0f);
    static DistributedMatrix<T> RandomHermitian(int context, int dim, int2 blockSize, int seed);

    template <typename U>
    static DistributedMatrix<T> Diagonal(int context, int2 blockSize, const LocalMatrix<U>& X);

    static DistributedMatrix<T> Duplicate(const DistributedMatrix<T>& A);
    static void Duplicate(const DistributedMatrix<T>& A, DistributedMatrix<T>& B);

private:
    DistributedMatrix(void) = default;

    void Init(int context, int2 dims, int2 blockSize);

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
enum TRANS_OPT {
    TRANS_OPT_NONE = 0,
    TRANS_OPT_TRANS,
    TRANS_OPT_CONJ_TRANS
};
template <typename T>
void PvGEMM(
    TRANS_OPT transA, TRANS_OPT transB,
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, 
    double beta, DistributedMatrix<T>& C
);

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
void PvGERQF(DistributedMatrix<T>& A, DistributedMatrix<T>& Tau);

template <typename T>
void PvORMQR(const DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau, DistributedMatrix<T>& C);

template <typename T>
void PvUNMQR(const DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau, DistributedMatrix<T>& C);

template <typename T>
void PvORGQR(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau);

template <typename T>
void PvUNGQR(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau);

template <typename T>
void PvORGR2(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau);

template <typename T>
void PvUNGR2(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau);

template <typename T>
void PvHEEV(DistributedMatrix<T>& A, LocalMatrix<ValueType<T>>& W, DistributedMatrix<T>& Z);

template <typename T>
void PvHEEV(DistributedMatrix<T>& A, LocalMatrix<ValueType<T>>& W);

template <typename T>
void PvTRAN(ValueType<T> a, const DistributedMatrix<T>& A, ValueType<T> b, DistributedMatrix<T>& C);

template <typename T>
void PvTRANU(ValueType<T> a, const DistributedMatrix<T>& A, ValueType<T> b, DistributedMatrix<T>& C);

template <typename T>
void PvTRANC(ValueType<T> a, const DistributedMatrix<T>& A, ValueType<T> b, DistributedMatrix<T>& C);

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

// sub( Y ) := sub( Y ) + alpha * sub( X )
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