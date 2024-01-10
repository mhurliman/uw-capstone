
#include "OpHelpers.h"

template <typename T>
std::ostream& operator<<(std::ostream& os, const LocalMatrix<T>& m)
{
    m.Print(os);
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const DistributedMatrix<T>& m)
{
    m.PrintGlobal(os);
    return os;
}

template <typename T>
void ScatterMatrix(int2 id, int2 pgDims, const T* global, const MatDesc& desc, T* local, int2 localDims)
{
    const int M = desc.M;
    const int N = desc.N;
    const int Mb = desc.Mb;
    const int Nb = desc.Nb;
    const int ctxt = desc.ctxt;

    int sendr = 0;
    int sendc = 0;
    int recvr = 0; 
    int recvc = 0;

    for (int r = 0; r < M; r += Mb, sendr = (sendr + 1) % pgDims.row) 
    {
        sendc = 0;

        int nr = std::min(Mb, M - r);

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % pgDims.col) 
        {
            int nc = std::min(Nb, N - c);

            if (id.IsRoot()) 
            {
                ops::GESD2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }

            if (id.row == sendr && id.col == sendc) 
            {
                ops::GERV2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}

template <typename T>
void GatherMatrix(int2 id, int2 pgDims, T* global, const MatDesc& desc, const T* local, int2 localDims)
{
    const int M = desc.M;
    const int N = desc.N;
    const int Mb = desc.Mb;
    const int Nb = desc.Nb;
    const int ctxt = desc.ctxt;

    int sendr = 0;
    int sendc = 0;
    int recvr = 0; 
    int recvc = 0;

    for (int r = 0; r < M; r += Mb, sendr = (sendr + 1) % pgDims.row) 
    {
        sendc = 0;

        int nr = std::min(Mb, M - r);

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % pgDims.col) 
        {
            int nc = std::min(Nb, N - c);

            if (id.row == sendr && id.col == sendc) 
            {
                ops::GESD2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }
            
            if (id.IsRoot()) 
            {
                ops::GERV2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}


template <typename T>
void LocalMatrix<T>::Print(std::ostream& os) const
{
    for (int iRow = 0; iRow < m_dims.row; ++iRow)
    {
        for (int iCol = 0; iCol < m_dims.col; ++iCol)
        {
            os << std::setw(4) << std::setprecision(2) << m_data[CMIDX(iRow, iCol, m_dims.row)] << " ";
        }
        os << std::endl;
    }
}

template <typename T>
void LocalMatrix<T>::SetElements(InitializerFunc f)
{
    for (int iCol = 0; iCol < m_dims.col; ++iCol)
    {
        for (int iRow = 0; iRow < m_dims.row; ++iRow)
        {
            m_data[CMIDX(iRow, iCol, m_dims.row)] = f(int2{iCol, iRow});
        }
    }
}

template <typename T>
void LocalMatrix<T>::CustomOp(CustomOpFunc f)
{
    for (int iCol = 0; iCol < m_dims.col; ++iCol)
    {
        for (int iRow = 0; iRow < m_dims.row; ++iRow)
        {
            f(int2{iCol, iRow}, m_data[CMIDX(iRow, iCol, m_dims.row)]);
        }
    }
}

template <typename T>
LocalMatrix<T> LocalMatrix<T>::Uninitialized(int numRows, int numCols)
{
    LocalMatrix<T> A;
    A.Init(numRows, numCols);
    return A;
}

template <typename T>
template <typename U>
LocalMatrix<T> LocalMatrix<T>::Uninitialized(const LocalMatrix<U>& dimsSpec)
{
    return Uninitialized(dimsSpec.m_dims.row, dimsSpec.m_dims.col);
}

template <typename T>
LocalMatrix<T> LocalMatrix<T>::Initialized(int numRows, int numCols, InitializerFunc f)
{
    auto A = Uninitialized(numRows, numCols);
    A.SetElements(f);
    return A;
}

template <typename T>
template <typename U>
LocalMatrix<T> LocalMatrix<T>::Initialized(const LocalMatrix<U>& data)
{
    auto A = Uninitialized(data);
    A.SetElements([&](int2 gid)
    {
        return T(data.m_data[CMIDX(gid.row, gid.col, data.m_dims.row)]); // Constructor conversion
    });
    return A;
}

template <typename T>
void LocalMatrix<T>::Init(int numRows, int numCols)
{
    assert(numRows > 0 & numCols > 0);

    m_data.reset(new T[numRows * numCols]);
    m_dims = { numCols, numRows };
}


template <typename T>
LocalMatrix<T> DistributedMatrix<T>::ToLocalMatrix(void) const
{
    LocalMatrix<T> m;
    if (IsRootProcess())
    {
        m = LocalMatrix<T>::Uninitialized(m_desc.M, m_desc.N);
    }

    GatherMatrix<T>(m_PGridId, m_PGridDims, m.Data(), m_desc, m_localData.get(), m_localDims);
    return std::move(m);
}


template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Submatrix(int numRows, int numCols, int rowIndex, int colIndex) const
{
    DistributedMatrix<T> A = *this;
    A.m_subDims = int2 { .col = numCols, .row = numRows };
    A.m_subIndex = int2::OneBased(colIndex, rowIndex);

    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::SubmatrixRow(int rowIndex) const
{
    assert(false); // Unimplemented - requires proper handling of 'transpose' & 'inc'
    return Submatrix(1, m_desc.N, rowIndex, 0);
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::SubmatrixColumn(int colIndex) const
{
    return Submatrix(m_desc.M, 1, 0, colIndex);
}

template <typename T>
void DistributedMatrix<T>::SetElements(InitializerFunc f)
{
    for (int iCol = 0; iCol < m_localDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < m_localDims.row; ++iRow)
        {
            int2 gid = LocalToGlobal(iRow, iCol, m_PGridDims, m_PGridId, m_desc);

            m_localData[CMIDX(iRow, iCol, m_localDims.row)] = f(gid);
        }
    }
}

template <typename T>
void DistributedMatrix<T>::CustomLocalOp(CustomOpFunc f)
{
    for (int iCol = 0; iCol < m_localDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < m_localDims.row; ++iRow)
        {
            int2 gid = LocalToGlobal(iRow, iCol, m_PGridDims, m_PGridId, m_desc);

            f(gid, m_localData[CMIDX(iRow, iCol, m_localDims.row)]);
        }
    }
}

template <typename T>
void DistributedMatrix<T>::Init(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    m_PGridDims = pgDims;
    m_PGridId = pgId;

    const int zero = 0;
    m_localDims.row = numroc_(&numRows, &rowBlockSize, &pgId.row, &zero, &pgDims.row);
    m_localDims.col = numroc_(&numCols, &colBlockSize, &pgId.col, &zero, &pgDims.col);

    m_subDims = int2{ .col = numCols, .row = numRows };
    m_subIndex = int2::One();

    if (m_localDims.row > 0 && m_localDims.col > 0)
    {
        m_localData.reset(new T[m_localDims.row * m_localDims.col]);
    }
    else
    {
        m_localData.reset(new T[1]);
    }
    assert(m_localData.get() != nullptr);

    int info;
    descinit_(m_desc, &numRows, &numCols, &rowBlockSize, &colBlockSize, &zero, &zero, &context, &m_localDims.row, &info);
    assert(info >= 0);
}

template <typename T>
void DistributedMatrix<T>::PrintLocal(std::ostream& os) const
{
    for (int i = 0; i < m_PGridDims.NumProcs(); ++i)
    {
        if (m_PGridId.Flat(m_PGridDims.row) == i)
        {
            for (int iCol = 0; iCol < m_localDims.col; ++iCol)
            {
                for (int iRow = 0; iRow < m_localDims.row; ++iRow)
                {
                    os << std::setw(4) << std::setprecision(2) << m_localData[CMIDX(iRow, iCol, m_localDims.row)] << " ";
                }
                os << std::endl;
            }
        }

        Cblacs_barrier(m_desc.ctxt, "All");
    }
}

template <typename T>
void DistributedMatrix<T>::PrintGlobal(std::ostream& os) const
{
    auto m = ToLocalMatrix();

    if (IsRootProcess())
    {
        for (int r = 0; r < m_subDims.row; ++r) 
        {
            for (int c = 0; c < m_subDims.col; ++c) 
            {
                os << std::setw(4) << std::setprecision(2) << m.m_data[CMIDX(r, c, m_desc.M)] << " ";
            }
            os << std::endl;
        }
    }
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    DistributedMatrix<T> A;
    A.Init(context, pgDims, pgId, numRows, numCols, rowBlockSize, colBlockSize);
    return A;
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(const DistributedMatrix<U>& dimsSpec)
{
    auto& desc = dimsSpec.Desc();

    DistributedMatrix<T> A;
    A.Init(desc.ctxt, dimsSpec.ProcGridDimensions(), dimsSpec.ProcGridId(), desc.M, desc.N, desc.Mb, desc.Nb);
    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize,
    InitializerFunc f
)
{
    auto A = Uninitialized(context, pgDims, pgId, numRows, numCols, rowBlockSize, colBlockSize);
    A.SetElements(f);
    return A;
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(
    int context, int2 pgDims, int2 pgId, 
    int rowBlockSize, int colBlockSize, const LocalMatrix<U>& data
)
{
    return Uninitialized(context, pgDims, pgId, data.m_dims.row, data.m_dims.col, rowBlockSize, colBlockSize);
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(
    int context, int2 pgDims, int2 pgId, 
    int rowBlockSize, int colBlockSize, const LocalMatrix<U>& data
)
{
    auto A = Uninitialized<U>(context, pgDims, pgId, rowBlockSize, colBlockSize, data);

    LocalMatrix<T> m = LocalMatrix<T>::Initialized(data);
    ScatterMatrix<T>(pgId, pgDims, m.Data(), A.Desc(), A.Data(), A.m_localDims);

    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Identity(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    return Initialized(
        context, pgDims, pgId, 
        numRows, numCols, rowBlockSize, colBlockSize, 
        [](int2 gid) { return double(gid.col == gid.row); }
    );
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::UniformRandom(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize,
    int seed, double range
)
{
    int state = 0;
    Random<T> r(seed);

    return Initialized(
        context, pgDims, pgId, 
        numRows, numCols, rowBlockSize, colBlockSize, 
        [&](int2 gid) 
        {
            // Discard up to this global index
            for (; state < gid.Flat(numRows); ++state)
            {
                r();
            }

            ++state;
            return r() * range;
        }
    );
}


template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Diagonal(
    int context, int2 pgDims, int2 pgId,
    int rowBlockSize, int colBlockSize, const LocalMatrix<U>& X
)
{
    assert(X.m_dims.col == 1);

    DistributedMatrix<T> A;
    A.Init(context, pgDims, pgId, X.m_dims.row, X.m_dims.row, rowBlockSize, colBlockSize);

    A.SetElements([&](int2 gid)
    { 
        return gid.col == gid.row ? X.m_data[gid.col] : 0; 
    });

    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Duplicate(const DistributedMatrix<T>& A)
{
    DistributedMatrix<T> B;
    B.Init(A.Desc().ctxt, A.m_PGridDims, A.m_PGridId, A.Desc().M, A.Desc().N, A.Desc().Mb, A.Desc().Nb);

    Duplicate(A, B);
    return B;
}

template <typename T>
void DistributedMatrix<T>::Duplicate(const DistributedMatrix<T>& src, DistributedMatrix<T>& dest)
{
    assert(
        src.NumCols() == dest.NumCols() &&
        src.NumRows() == dest.NumRows()
    );

    const int one = 1;
    ops::PvLACPY("A", &src.Desc().M, &src.Desc().M, src.Data(), &one, &one, src.Desc(), dest.Data(), &one, &one, dest.Desc());
}


template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::RandomHermitian(
    int context, int2 pgDims, int2 pgId, 
    int N, int rowBlockSize, int colBlockSize,
    int seed
)
{
    int state = 0;
    Random<T> r(seed);

    auto A = Initialized(
        context, pgDims, pgId, 
        N, N, rowBlockSize, colBlockSize, 
        [&](int2 gid) 
        {
            if (gid.col > gid.row)
            {
                return T{};
            }

            // Discard up to this global index
            for (; state < gid.Flat(N); ++state)
            {
                r();
            }

            ++state;
            if (gid.col < gid.row) // Complex in lower-triangulars
            {
                return r();
            }
            else  // Reals along diagonal; multiply by half because our tranpose
            {
                return r.GenerateReal() * 0.5;
            }
        }
    );

    // Duplicate to create final dest matrix
    auto B = Duplicate(A);

    // Tranpose conjugate 
    const double one = 1.0;
    ops::PvTRANC<T>(
        &A.NumRows(), &A.NumCols(), 
        &one, A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        &one, B.Data(), &B.IndexRow(), &B.IndexCol(), B.Desc()
    );

    return B;
}

template <typename T>
void PvGEMM(
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, 
    double beta, DistributedMatrix<T>& C
)
{
    assert(
        A.NumCols() == B.NumRows() &&
        A.NumRows() == C.NumRows() &&
        B.NumCols() == C.NumCols()
    );

    const int one = 1;
    ops::PvGEMM<T>(
        "N", "N", 
        &A.Desc().M, &B.Desc().N, &A.Desc().N, 
        &alpha,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        B.Data(), &B.IndexRow(), &B.IndexCol(), B.Desc(), 
        &beta,
        C.Data(), &C.IndexRow(), &C.IndexCol(), C.Desc()
    );
}

template <typename T>
void PvGEMV(
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& X, 
    double beta, DistributedMatrix<T>& Y
)
{
    assert(
        A.NumCols() == X.NumRows() &&
        X.NumRows() == Y.NumRows()
    );

	const int one = 1;
	ops::PvGEMV<T>(
        "N", &A.Desc().M, &A.Desc().N, 
        &alpha,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(),
		X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &one,
        &beta,
		Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &one
        );
}

template <typename T>
void PvHEEV(DistributedMatrix<T>& A, LocalMatrix<ValueType<T>>& W, DistributedMatrix<T>& Z)
{
    using U = ValueType<T>;
    assert(
        A.IsSquare() && 
        A.NumRows() == W.NumRows() &&
        A.NumRows() == Z.NumRows() &&
        A.NumCols() == Z.NumCols()
    );

    T workSize;
    U rworkSize;

	const int one = 1.0;
    int lwork = -1;
    int lrwork = -1;
    int info = 0;

    ops::PvHEEV<T>(
        "V", "U", &A.NumRows(), 
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, 
        nullptr, &Z.IndexRow(), &Z.IndexCol(), Z.Desc(),
        &workSize, &lwork, &rworkSize, &lrwork, &info);

    lwork = static_cast<int>(real(workSize));
    lrwork = static_cast<int>(rworkSize);

    std::unique_ptr<T[]> work;
    std::unique_ptr<U[]> rwork;
    work.reset(new T[lwork]);
    rwork.reset(new U[lrwork]);

    ops::PvHEEV(
        "V", "U", &A.NumRows(), 
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        W.Data(), 
        Z.Data(), &Z.IndexRow(), &Z.IndexCol(), Z.Desc(), 
        work.get(), &lwork, rwork.get(), &lrwork, &info);
}

template <typename T>
void PvHEEV(DistributedMatrix<T>& A, LocalMatrix<ValueType<T>>& W)
{
    using U = ValueType<T>;
    assert(A.IsSquare() && A.NumRows() == W.NumRows());

    T workSize;
    U rworkSize;

    int lwork = -1;
    int lrwork = -1;
    int info = 0;

    ops::PvHEEV<T>(
        "N", "U", &A.NumRows(), 
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, 
        nullptr, nullptr, nullptr, nullptr,
        &workSize, &lwork, &rworkSize, &lrwork, &info
    );

    lwork = static_cast<int>(real(workSize));
    lrwork = static_cast<int>(rworkSize);

    std::unique_ptr<T[]> work;
    std::unique_ptr<U[]> rwork; 
    work.reset(new T[lwork]);
    rwork.reset(new U[lrwork]);

    ops::PvHEEV(
        "N", "U", &A.NumRows(), 
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        W.Data(), 
        nullptr, nullptr, nullptr, nullptr, 
        work.get(), &lwork, rwork.get(), &lrwork, &info);
}

template <typename T>
void PvTRAN(double a, const DistributedMatrix<T>& A, double b, DistributedMatrix<T>& C)
{
    assert(
        A.NumRows() == C.NumRows() &&
        A.NumCols() == C.NumCols()
    );

    ops::PvTRAN(
        &A.NumRows(), &A.NumCols(),
        &a, A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(),
        &b, C.Data(), &C.IndexRow(), &C.IndexCol(), C.Desc()
    );
}

template <typename T>
void PvTRANU(double a, const DistributedMatrix<T>& A, double b, DistributedMatrix<T>& C)
{
    assert(
        A.NumRows() == C.NumRows() &&
        A.NumCols() == C.NumCols()
    );

    ops::PvTRANU(
        &A.NumRows(), &A.NumCols(),
        &a, A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(),
        &b, C.Data(), &C.IndexRow(), &C.IndexCol(), C.Desc()
    );
}

template <typename T>
void PvTRANC(double a, const DistributedMatrix<T>& A, double b, DistributedMatrix<T>& C)
{
    assert(
        A.NumRows() == C.NumRows() &&
        A.NumCols() == C.NumCols()
    );

    ops::PvTRANC(
        &A.NumRows(), &A.NumCols(),
        &a, A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(),
        &b, C.Data(), &C.IndexRow(), &C.IndexCol(), C.Desc()
    );
}

template <typename T>
float PvDOT(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y)
{
    assert(
        X.NumRows() == Y.NumRows() &&
        X.NumCols() == 1 &&
        Y.NumCols() == 1
    );
    
    ValueType<T> result{};

    const int inc = 1;
    ops::PvDOT(
        &X.NumRows(),
        &result,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
    );

    return result;
}

template <typename T>
T PvDOTU(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y)
{
    assert(
        X.NumRows() == Y.NumRows() &&
        X.NumCols() == 1 &&
        Y.NumCols() == 1
    );

    ValueType<T> result{};

    const int inc = 1;
    ops::PvDOTU(
        &X.NumRows(),
        &result,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
    );

    return result;
}

template <typename T>
T PvDOTC(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y)
{
    assert(
        X.NumRows() == Y.NumRows() &&
        X.NumCols() == 1 &&
        Y.NumCols() == 1
    );

    ValueType<T> result{};

    const int inc = 1;
    ops::PvDOTC(
        &X.NumRows(),
        &result,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
    );

    return result;
}

template <typename T>
void PvSCAL(ValueType<T> a, const DistributedMatrix<T>& X)
{
    const int inc = 1;
    ops::PvSCAL(
        &X.NumRows(),
        &a,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc
    );
}

template <typename T>
void PvSWAP(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y)
{
    assert(X.NumRows() == Y.NumRows());

    const int inc = 1;
    ops::PvSWAP(
        &X.NumRows(),
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
    );
}

template <typename T>
void PvLACPY(const DistributedMatrix<T>& A, DistributedMatrix<T>& B)
{
    assert(
        A.NumRows() == B.NumRows() &&
        A.NumCols() == B.NumCols()
    );

    const int inc = 1;
    ops::PvLACPY(
        "N", // 'U' - copy upper-tri, 'L' - copy lower-tri, 'N' - copy all
        &A.NumRows(), &A.NumCols(),
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), &inc,
        B.Data(), &B.IndexRow(), &B.IndexCol(), B.Desc(), &inc
    );
}

template <typename T>
ValueType<T> PvNRM2(const DistributedMatrix<T>& X)
{
    ValueType<T> result{};

    const int inc = 1;
    ops::PvNRM2(
        &X.NumRows(),
        &result,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc
    );

    constexpr auto dtype = std::is_same_v<float, ValueType<T>> ? MPI_FLOAT : MPI_DOUBLE;
    int root = ((X.IndexCol() - 1) / X.Desc().Mb) % X.ProcGridDimensions().col;

    MPI_Bcast(&result, 1, dtype, root, MPI_COMM_WORLD);

    return result;
}

template <typename T>
ValueType<T> PvASUM(const DistributedMatrix<T>& X)
{
    ValueType<T> result{};

    const int inc = 1;
    ops::PvASUM(
        &X.NumRows(),
        &result,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc
    );

    return result;
}

template <typename T>
void PvAMAX(T& max, int& index, const DistributedMatrix<T>& X)
{
    const int inc = 1;
    ops::PvASUM(
        &X.NumRows(),
        &max, &index,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc
    );
}

template <typename T>
void PvCOPY(const DistributedMatrix<T>& X, DistributedMatrix<T>& Y)
{
    assert(
        X.NumRows() == Y.NumRows() &&
        X.NumCols() == Y.NumCols()
    );

    const int inc = 1;
    ops::PvCOPY(
        &X.NumRows(),
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
    );
}

template <typename T>
void PvAXPY(ValueType<T> a, const DistributedMatrix<T>& X, DistributedMatrix<T>& Y)
{
    assert(
        X.NumRows() == Y.NumRows() &&
        X.NumCols() == Y.NumCols()
    );

    const int inc = 1;
    ops::PvAXPY(
        &X.NumRows(),
        &a,
        X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
    );
}
