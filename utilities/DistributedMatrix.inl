
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
                ops::CvGESD2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }

            if (id.row == sendr && id.col == sendc) 
            {
                ops::CvGERV2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
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
                ops::CvGESD2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }

            if (id.IsRoot()) 
            {
                ops::CvGERV2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}

template <typename T>
ValueType<T> DistributedMatrix<T>::OneNorm(void) const
{
    // Compute local sum of each column
    auto sum = std::vector<ValueType<T>>(m_localDims.col, 0);

    for (int iCol = 0; iCol < m_localDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < m_localDims.row; ++iRow)
        {
            sum[iCol] += std::abs(m_localData[CMIDX(iRow, iCol, m_localDims.row)]);
        }
    }

    // Sum reduction along rows
    auto dest = std::vector<ValueType<T>>(m_localDims.col);

    for (int order = 1; order < m_PGridDims.row; order <<= 1)
    {
        // Swizzle the LSB to get the exchange process id
        int xRowId = m_PGridId.row ^ order;

        // Sit out this exchange if the sending thread is out of range of the grid dimensions
        if (xRowId >= m_PGridDims.row)
            continue;

        // Threads with '1' LSB send to lower id threads
        if ((m_PGridId.row & order) != 0)
        {
            ops::CvGESD2D(m_desc.ctxt, 1, m_localDims.col, sum.data(), 1, xRowId, m_PGridId.col);
            break;
        }
        else // Threads with '0' LSB receive from higher threads
        {
            ops::CvGERV2D(m_desc.ctxt, 1, m_localDims.col, dest.data(), 1, xRowId, m_PGridId.col);

            // Accumulate exchanged values
            for (int i = 0; i < m_localDims.col; ++i)
            {
                sum[i] += dest[i];
            }
        }
    }

    ValueType<T> max = *std::max_element(sum.begin(), sum.end());
    ValueType<T> recv;

    // Max reduction along columns of row 0
    if (m_PGridId.row == 0)
    {
        for (int order = 1; order < m_PGridDims.col; order <<= 1)
        {
            int xColId = m_PGridId.col ^ order;
            if (xColId >= m_PGridDims.col)
                continue;

            if ((m_PGridId.col & order) != 0)
            {
                ops::CvGESD2D(m_desc.ctxt, 1, 1, &max, 1, 0, xColId);
                break;
            }
            else
            {
                ops::CvGERV2D(m_desc.ctxt, 1, 1, &recv, 1, 0, xColId);
                max = std::max(max, recv);
            }

        }
    }

    // Optional step, but broadcast the final result back sto all processes
    MPI_Bcast(&max, 1, MPI_Type<T>, 0, MPI_COMM_WORLD);

    return max;
}

template <typename T>
ValueType<T> DistributedMatrix<T>::InfinityNorm(void) const
{
    // Compute local sum of each row
    auto sum = std::vector<ValueType<T>>(m_localDims.row, 0);

    for (int iCol = 0; iCol < m_localDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < m_localDims.row; ++iRow)
        {
            sum[iRow] += std::abs(m_localData[CMIDX(iRow, iCol, m_localDims.row)]);
        }
    }

    // Sum reduction along rows
    auto dest = std::vector<ValueType<T>>(m_localDims.row);

    for (int order = 1; order < m_PGridDims.col; order <<= 1)
    {
        // Swizzle the LSB of the column id to get the exchange process id
        int xColId = m_PGridId.col ^ order;

        // Sit out this exchange if the sending thread is out of range of the grid dimensions
        if (xColId >= m_PGridDims.col)
            continue;

        // Processes with '1' LSB send to lower id threads
        if ((m_PGridId.col & order) != 0)
        {
            ops::CvGESD2D(m_desc.ctxt, 1, m_localDims.row, sum.data(), 1, m_PGridId.row, xColId);
            break;
        }
        else // Threads with '0' LSB receive from higher threads
        {
            ops::CvGERV2D(m_desc.ctxt, 1, m_localDims.row, dest.data(), 1, m_PGridId.row, xColId);

            // Accumulate exchanged values
            for (int i = 0; i < m_localDims.row; ++i)
            {
                sum[i] += dest[i];
            }
        }
    }

    ValueType<T> max = *std::max_element(sum.begin(), sum.end());
    ValueType<T> recv;

    // Max reduction along columns of row 0
    if (m_PGridId.col == 0)
    {
        for (int order = 1; order < m_PGridDims.row; order <<= 1)
        {
            int xRowId = m_PGridId.row ^ order;
            
            if (xRowId >= m_PGridDims.row)
                continue;

            // Processes with '1' LSB send to lower id threads
            if ((m_PGridId.row & order) != 0)
            {
                ops::CvGESD2D(m_desc.ctxt, 1, 1, &max, 1, xRowId, 0);
                break;
            }
            else
            {
                ops::CvGERV2D(m_desc.ctxt, 1, 1, &recv, 1, xRowId, 0);
                max = std::max(max, recv);
            }

        }
    }

    // Optional step, but broadcast the final result back sto all processes
    MPI_Bcast(&max, 1, MPI_Type<T>, 0, MPI_COMM_WORLD);

    return max;
}

template <typename T>
ValueType<T> DistributedMatrix<T>::EuclideanNorm(void) const
{
    // Compute local absolute sum
    ValueType<T> sum = std::accumulate(
        m_localData.get(), 
        m_localData.get() + m_localDims.Count(),
        0,
        [](auto x, auto y) { return x + std::real(std::conj(y) * y); } // Must account for complex dot product
    );

    // Sum reduction along rows
    ValueType<T> dest;

    for (int order = 1; order < m_PGridDims.row; order <<= 1)
    {
        // Swizzle the LSB to get the exchange process id
        int xRowId = m_PGridId.row ^ order;

        // Sit out this exchange if the sending thread is out of range of the grid dimensions
        if (xRowId >= m_PGridDims.row)
            continue;

        // Threads with '1' LSB send to lower id threads
        if ((m_PGridId.row & order) != 0)
        {
            ops::CvGESD2D(m_desc.ctxt, 1, 1, &sum, 1, xRowId, m_PGridId.col);
            break;
        }
        else // Threads with '0' LSB receive from higher threads
        {
            ops::CvGERV2D(m_desc.ctxt, 1, 1, &dest, 1, xRowId, m_PGridId.col);

            sum += dest;
        }
    }

    // Sum reduction along columns of row 0
    if (m_PGridId.row == 0)
    {
        for (int order = 1; order < m_PGridDims.col; order <<= 1)
        {
            int xColId = m_PGridId.col ^ order;
            
            // Sit out this exchange if the sending thread is out of range of the grid dimensions
            if (xColId >= m_PGridDims.col)
                continue;

            if ((m_PGridId.col & order) != 0)
            {
                ops::CvGESD2D(m_desc.ctxt, 1, 1, &sum, 1, 0, xColId);
                break;
            }
            else
            {
                ops::CvGERV2D(m_desc.ctxt, 1, 1, &dest, 1, 0, xColId);
                sum += dest;
            }
        }
    }

    // Norm that baby
    sum = sqrt(sum);

    // Optional step, but broadcast the final result back sto all processes
    MPI_Bcast(&sum, 1, MPI_Type<T>, 0, MPI_COMM_WORLD);

    return sum;
}

template <typename T>
void PrintValue(std::ostream& os, T v)
{
    os << std::setw(8) << std::setprecision(2) << FormatZeros(v, 1e-4) << " ";
}

template <typename T>
void LocalMatrix<T>::Print(std::ostream& os) const
{
    for (int iRow = 0; iRow < m_dims.row; ++iRow)
    {
        for (int iCol = 0; iCol < m_dims.col; ++iCol)
        {
            PrintValue(os, m_data[CMIDX(iRow, iCol, m_dims.row)] );
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
            m_data[CMIDX(iRow, iCol, m_dims.row)] = f({ .row = iRow, .col = iCol });
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
            f({ .row = iRow, .col = iCol }, m_data[CMIDX(iRow, iCol, m_dims.row)]);
        }
    }
}

template <typename T>
LocalMatrix<T> LocalMatrix<T>::Uninitialized(int2 dims)
{
    LocalMatrix<T> A;
    A.Init(dims);
    return A;
}

template <typename T>
template <typename U>
LocalMatrix<T> LocalMatrix<T>::Uninitialized(const LocalMatrix<U>& dimsSpec)
{
    return Uninitialized(dimsSpec.m_dims);
}

template <typename T>
LocalMatrix<T> LocalMatrix<T>::Initialized(int2 dims, InitializerFunc f)
{
    auto A = Uninitialized(dims);
    A.SetElements(f);
    return A;
}


template <typename T>
LocalMatrix<T> LocalMatrix<T>::Initialized(int2 dims, const std::initializer_list<T>& rowMajorList)
{
    assert(dims.Count() == rowMajorList.size());

    auto A = Uninitialized(dims);

    int i = 0;
    for (auto it = rowMajorList.begin(); it < rowMajorList.end(); ++it, ++i)
    {
        int row = i / dims.col;
        int col = i - row * dims.col;

        A.m_data[CMIDX(row, col, dims.row)] = *it;
    }

    return A;
}

template <typename T>
template <typename U>
LocalMatrix<T> LocalMatrix<T>::Initialized(const LocalMatrix<U>& data)
{
    auto A = Uninitialized(data);
    std::memcpy(A.m_data.get(), data.m_data.get(), A.Bytes());
    return A;
}

template <typename T>
ValueType<T> LocalMatrix<T>::ASum(void) const
{
    ValueType<T> acc = 0;
    for (int i = 0; i < NumElements(); ++i)
    {
        acc += std::abs(m_data[i]);
    }
    return acc;
}

template <typename T>
void LocalMatrix<T>::Init(int2 dims)
{
    assert(dims.ValidDims());

    m_data.reset(new T[dims.Count()]);
    m_dims = dims;
}


template <typename T>
LocalMatrix<T> DistributedMatrix<T>::ToLocalMatrix(void) const
{
    LocalMatrix<T> m;
    if (IsRootProcess())
    {
        m = LocalMatrix<T>::Uninitialized(m_subDims);
    }

    GatherMatrix<T>(m_PGridId, m_PGridDims, m.Data(), m_desc, m_localData.get(), m_localDims);
    return std::move(m);
}


template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Submatrix(int2 dims, int2 indices) const
{
    DistributedMatrix<T> A = *this;
    A.m_subDims = dims;
    A.m_subIndex = indices.ToOneBased();

    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::SubmatrixRow(int rowIndex) const
{
    assert(false); // Unimplemented - requires proper handling of 'transpose' & 'inc'
    return Submatrix({ 1, m_desc.N }, { rowIndex, 0 });
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::SubmatrixColumn(int colIndex) const
{
    return Submatrix({ m_desc.M, 1 }, { 0, colIndex });
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
void DistributedMatrix<T>::Init(int context, int2 dims, int2 blockSize)
{
    assert(dims.row != blockSize.row && dims.col != blockSize.col);

    Cblacs_gridinfo(context, &m_PGridDims.row, &m_PGridDims.col, &m_PGridId.row, &m_PGridId.col);

    const int zero = 0;
    m_localDims.row = numroc_(&dims.row, &blockSize.row, &m_PGridId.row, &zero, &m_PGridDims.row);
    m_localDims.col = numroc_(&dims.col, &blockSize.col, &m_PGridId.col, &zero, &m_PGridDims.col);

    m_subDims = dims;
    m_subIndex = int2::One(); // A horrifically base-one system

    int2 allocDims = m_localDims;
    if (allocDims.row == 1)
    {
        allocDims.col = dims.col;
    }
    else if (allocDims.col == 1)
    {
        allocDims.row = dims.row;
    }

    m_localData.reset(new T[allocDims.Count()]);

    assert(m_localData.get() != nullptr);

    int info;
    descinit_(
        m_desc, 
        &m_subDims.row, &m_subDims.col, 
        &blockSize.row, &blockSize.col, 
        &zero, &zero, 
        &context, 
        &m_localDims.row,
        &info
    );
    assert(info >= 0);
}

template <typename T>
void DistributedMatrix<T>::PrintLocal(std::ostream& os) const
{
#if 1
    int pid = m_PGridId.FlatRM(m_PGridDims.col);

    for (int i = 0; i < m_PGridDims.Count(); ++i)
    {
        if (pid == i)
        {
            std::cout << "PID " << pid << ":  " << m_localDims << std::endl;

            for (int iRow = 0; iRow < m_localDims.row; ++iRow)
            {
                for (int iCol = 0; iCol < m_localDims.col; ++iCol)
                {
                    PrintValue(os, m_localData[CMIDX(iRow, iCol, m_localDims.row)]);
                }
                os << std::endl;
            }
        }

        os.flush();
        Cblacs_barrier(m_desc.ctxt, "All");
    }
#else

    Cblacs_barrier(m_desc.ctxt, "All");
    for (int iRow = 0; iRow < m_desc.M; ++iRow)
    {
        for (int iCol = 0; iCol < m_desc.N; iCol += m_desc.Nb)
        {
            int2 pgid;
            int2 id;
            GlobalToLocal(iRow, iCol, m_PGridDims, m_desc, pgid, id);

            if (pgid == m_PGridId)
            {
                int iEnd = std::min(m_desc.N, iCol + m_desc.Nb);
                for (int iColl = iCol; iColl < iEnd; ++iColl, ++id.col)
                {
                    T z = m_localData[CMIDX(id.row, id.col, m_localDims.row)];
                    // T v {id.row, id.col};
                    // T m {pgid.row, pgid.col};
                    PrintValue(os, z);
                    os.flush();
                }
            }
            
            Cblacs_barrier(m_desc.ctxt, "All");
        }

        if (IsRootProcess())
        {
            os << std::endl;
            
            os.flush();
        }

        Cblacs_barrier(m_desc.ctxt, "All");
    }

    if (IsRootProcess())
    {
        os << std::endl;
        os.flush();
    }

    Cblacs_barrier(m_desc.ctxt, "All");
#endif
}

template <typename T>
void DistributedMatrix<T>::PrintGlobal(std::ostream& os) const
{
    auto m = ToLocalMatrix();

    if (IsRootProcess())
    {
        os << m << std::endl;
    }
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(int context, int numRows, int blockSize)
{
    DistributedMatrix<T> A;
    A.Init(context, { numRows, 1 }, { blockSize, 1 });
    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(int context, int2 dims, int2 blockSize)
{
    DistributedMatrix<T> A;
    A.Init(context, dims, blockSize);
    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Zeros(int context, int numRows, int blockSize)
{
    DistributedMatrix<T> A;
    A.Init(context, { numRows, 1 }, { blockSize, 1 });
    std::memset(A.Data(), 0, A.Bytes());
    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Zeros(int context, int2 dims, int2 blockSize)
{
    DistributedMatrix<T> A;
    A.Init(context, dims, blockSize);
    std::memset(A.Data(), 0, A.Bytes());
    return A;
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(const DistributedMatrix<U>& dimsSpec)
{
    auto& desc = dimsSpec.Desc();

    DistributedMatrix<T> A;
    A.Init(desc.ctxt, { desc.M, desc.N }, { desc.Mb, desc.Nb });
    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(int context, int2 dims, int2 blockSize, InitializerFunc f)
{
    auto A = Uninitialized(context, dims, blockSize);
    A.SetElements(f);
    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(int context, int2 dims, int2 blockSize, const std::initializer_list<T>& rowMajorList)
{
    auto m = LocalMatrix<T>::Initialized(dims, rowMajorList);

    auto A = Uninitialized(context, dims, blockSize);
    A.SetElements([&](int2 gid) { return m[gid.FlatCM(dims.row)]; });
    return A;
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(int context, int2 blockSize, const LocalMatrix<U>& data)
{
    return Uninitialized(context, data.m_dims, blockSize);
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Initialized(int context, int2 blockSize, const LocalMatrix<U>& data)
{
    auto A = Uninitialized<U>(context, blockSize, data);

    LocalMatrix<T> m = LocalMatrix<T>::Initialized(data);
    ScatterMatrix<T>(A.ProcGridId(), A.ProcGridDims(), m.Data(), A.Desc(), A.Data(), A.m_localDims);

    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Identity(int context, int2 dims, int2 blockSize)
{
    return Initialized(context, dims, blockSize, [](int2 gid) { return double(gid.col == gid.row); });
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::UniformRandom(int context, int2 dims, int2 blockSize, int seed, double range)
{
    int state = 0;
    Random<T> r(seed);

    return Initialized(
        context, dims, blockSize, 
        [&](int2 gid) 
        {
            // Discard up to this global index
            for (; state < gid.FlatCM(dims.row); ++state)
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
DistributedMatrix<T> DistributedMatrix<T>::Diagonal(int context, int2 blockSize, const LocalMatrix<U>& X)
{
    assert(X.m_dims.col == 1);

    DistributedMatrix<T> A;
    A.Init(context, X.m_dims, blockSize);

    A.SetElements([&](int2 gid) { return gid.col == gid.row ? X.m_data[gid.col] : 0; });

    return A;
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Duplicate(const DistributedMatrix<T>& A)
{
    DistributedMatrix<T> B;
    B.Init(A.Desc().ctxt, A.Dims(), A.BlockSize());

    Duplicate(A, B);
    return B;
}

template <typename T>
void DistributedMatrix<T>::Duplicate(const DistributedMatrix<T>& src, DistributedMatrix<T>& dst)
{
    assert(
        src.NumCols() == dst.NumCols() &&
        src.NumRows() == dst.NumRows()
    );

    ops::PvLACPY(
        "A", &src.NumRows(), &src.NumCols(), 
        src.Data(), &src.IndexRow(), &src.IndexCol(), src.Desc(), 
        dst.Data(), &dst.IndexRow(), &dst.IndexCol(), dst.Desc()
    );
}


template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::RandomHermitian(int context, int dim, int2 blockSize, int seed)
{
    int state = 0;
    Random<T> r(seed);

    auto A = Initialized(
        context, { dim, dim }, blockSize, 
        [&](int2 gid) 
        {
            if (gid.col > gid.row)
            {
                return T{};
            }

            // Discard up to this global index
            for (; state < gid.FlatCM(dim); ++state)
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
    PvTRANC(1.0, A, 1.0, B);

    return B;
}

template <typename T>
void PvGEMM(
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, 
    double beta, DistributedMatrix<T>& C
)
{
    PvGEMM(TRANS_OPT_NONE, TRANS_OPT_NONE, alpha, A, B, beta, C);
}

template <typename T>
void PvGEMM(
    TRANS_OPT transA, TRANS_OPT transB,
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, 
    double beta, DistributedMatrix<T>& C
)
{
    assert(
        A.NumCols() == B.NumRows() &&
        A.NumRows() == C.NumRows() &&
        B.NumCols() == C.NumCols()
    );

    if (std::is_fundamental_v<T>)
    {
        transA = std::min(TRANS_OPT_TRANS, transA);
        transB = std::min(TRANS_OPT_TRANS, transB);
    }

    const char* opts = "NTC"; // N - no-op; T - transpose; C - conjugate transpose

    ops::PvGEMM<T>(
        &opts[transA], &opts[transB], 
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

    const int inc = 1;
	ops::PvGEMV<T>(
        "N", &A.Desc().M, &A.Desc().N, 
        &alpha,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(),
		X.Data(), &X.IndexRow(), &X.IndexCol(), X.Desc(), &inc,
        &beta,
		Y.Data(), &Y.IndexRow(), &Y.IndexCol(), Y.Desc(), &inc
        );
}

template <typename T>
void PvGERQF(DistributedMatrix<T>& A, DistributedMatrix<T>& Tau)
{
    assert(A.NumRows() <= Tau.NumRows());

    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvGERQF<T>(
        &A.NumRows(), &A.NumCols(),
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, 
        &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvGERQF(
        &A.NumRows(), &A.NumCols(),
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), 
        work.get(), &lwork, &info
    );
}

template <typename T>
void PvORMQR(const DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau, DistributedMatrix<T>& C)
{
    static_assert(std::is_floating_point_v<T>);

    // Assume result is C * Q for now (operation 'R')
    // So N_C == M_Q, and final matrix is M_C x N_Q
    assert(
        A.NumRows() <= Tau.NumRows() &&
        A.NumRows() == C.NumCols()
    );

    int numRefl = std::min(A.NumRows(), A.NumCols());
    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvORMQR<T>(
        "R", "N",
        &A.NumRows(), &A.NumCols(), &numRefl,
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, 
        nullptr, &C.IndexRow(), &C.IndexCol(), C.Desc(), 
        &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvORMQR(
        "R", "N",
        &A.NumRows(), &A.NumCols(), &numRefl,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), 
        C.Data(), &C.IndexRow(), &C.IndexCol(), C.Desc(), 
        work.get(), &lwork, &info
    );
}

template <typename T>
void PvUNMQR(const DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau, DistributedMatrix<T>& C)
{
    static_assert(std::is_compound_v<T> && std::is_fundamental_v<ValueType<T>>);

    // Assume result is C * Q for now (operation 'R')
    // So N_C == M_Q, and final matrix is M_C x N_Q
    assert(
        A.NumRows() == Tau.NumRows() &&
        A.NumRows() == C.NumCols()
    );

    int numRefl = std::min(A.NumRows(), A.NumCols());
    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvUNMQR<T>(
        "R", "N",
        &A.NumRows(), &A.NumCols(), &numRefl,
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, 
        nullptr, &C.IndexRow(), &C.IndexCol(), C.Desc(), 
        &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvUNMQR(
        "R", "N",
        &A.NumRows(), &A.NumCols(), &numRefl,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), 
        C.Data(), &C.IndexRow(), &C.IndexCol(), C.Desc(), 
        work.get(), &lwork, &info
    );
}

template <typename T>
void PvORGQR(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau)
{
    static_assert(std::is_floating_point_v<T>);
    assert(A.NumRows() <= Tau.NumRows());

    int numRefl = std::min(A.NumRows(), A.NumCols());

    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvORGQR<T>(
        &A.NumRows(), &A.NumCols(), &numRefl,
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvORGQR<T>(
        &A.NumRows(), &A.NumCols(), &numRefl,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), work.get(), &lwork, &info
    );
}

template <typename T>
void PvUNGQR(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau)
{
    static_assert(std::is_compound_v<T> && std::is_fundamental_v<ValueType<T>>);

    assert(A.NumRows() == Tau.NumRows());

    int numRefl = std::min(A.NumRows(), A.NumCols());
    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvUNGQR<T>(
        &A.NumRows(), &A.NumCols(), &numRefl,
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvUNGQR(
        &A.NumRows(), &A.NumCols(), &numRefl,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), work.get(), &lwork, &info
    );
}

template <typename T>
void PvORGR2(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau)
{
    static_assert(std::is_floating_point_v<T>);
    assert(A.NumRows() <= Tau.NumRows());

    int numRefl = std::min(A.NumRows(), A.NumCols());

    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvORGR2<T>(
        &A.NumRows(), &A.NumCols(), &numRefl,
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvORGR2<T>(
        &A.NumRows(), &A.NumCols(), &numRefl,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), work.get(), &lwork, &info
    );
}

template <typename T>
void PvUNGR2(DistributedMatrix<T>& A, const DistributedMatrix<T>& Tau)
{
    static_assert(std::is_compound_v<T> && std::is_fundamental_v<ValueType<T>>);

    assert(A.NumRows() == Tau.NumRows());

    int numRefl = std::min(A.NumRows(), A.NumCols());
    T workSize;
    int lwork = -1;
    int info = 0;

    ops::PvUNGR2<T>(
        &A.NumRows(), &A.NumCols(), &numRefl,
        nullptr, &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        nullptr, &workSize, &lwork, &info
    );

    lwork = ToIndex(workSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);

    ops::PvUNGR2(
        &A.NumRows(), &A.NumCols(), &numRefl,
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        Tau.Data(), work.get(), &lwork, &info
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
        &workSize, &lwork, &rworkSize, &lrwork, &info
    );

    lwork = ToIndex(workSize);
    lrwork = ToIndex(rworkSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);
    auto rwork = std::unique_ptr<U[]>(new U[lrwork]);

    ops::PvHEEV(
        "V", "U", &A.NumRows(), 
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        W.Data(), 
        Z.Data(), &Z.IndexRow(), &Z.IndexCol(), Z.Desc(), 
        work.get(), &lwork, rwork.get(), &lrwork, &info
    );
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

    lwork = ToIndex(workSize);
    lrwork = ToIndex(rworkSize);
    auto work = std::unique_ptr<T[]>(new T[lwork]);
    auto rwork = std::unique_ptr<U[]>(new U[lrwork]);

    ops::PvHEEV(
        "N", "U", &A.NumRows(), 
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(), 
        W.Data(), 
        nullptr, nullptr, nullptr, nullptr, 
        work.get(), &lwork, rwork.get(), &lrwork, &info
    );
}

template <typename T>
void PvTRAN(ValueType<T> a, const DistributedMatrix<T>& A, ValueType<T> b, DistributedMatrix<T>& C)
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
void PvTRANU(ValueType<T> a, const DistributedMatrix<T>& A, ValueType<T> b, DistributedMatrix<T>& C)
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
void PvTRANC(ValueType<T> a, const DistributedMatrix<T>& A, ValueType<T> b, DistributedMatrix<T>& C)
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

    T result{};

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

    T result{};

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
        A.Data(), &A.IndexRow(), &A.IndexCol(), A.Desc(),
        B.Data(), &B.IndexRow(), &B.IndexCol(), B.Desc()
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

    // Broadcast owning value to all threads
    constexpr auto dtype = std::is_same_v<float, ValueType<T>> ? MPI_FLOAT : MPI_DOUBLE;
    int root = ((X.IndexCol() - 1) / X.Desc().Mb) % X.ProcGridDims().col;

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
