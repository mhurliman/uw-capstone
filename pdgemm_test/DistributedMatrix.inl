
#include <assert.h>
#include <iostream>
#include <math.h>
#include <iomanip>

#include <Random.h>

// GERV2D
template <typename T>
void GERV2D(int context, int M, int N, T* A, int lda, int rsrc, int csrc);

template <>
void GERV2D<float>(int context, int M, int N, float* A, int lda, int rsrc, int csrc)
{
    Csgerv2d(context, M, N, A, lda, rsrc, csrc);
}

template <>
void GERV2D<double>(int context, int M, int N, double* A, int lda, int rsrc, int csrc)
{
    Cdgerv2d(context, M, N, A, lda, rsrc, csrc);
}

template <>
void GERV2D<c32>(int context, int M, int N, c32* A, int lda, int rsrc, int csrc)
{
    Ccgerv2d(context, M, N, A, lda, rsrc, csrc);
}

template <>
void GERV2D<c64>(int context, int M, int N, c64* A, int lda, int rsrc, int csrc)
{
    Czgerv2d(context, M, N, A, lda, rsrc, csrc);
}

// GESD2D
template <typename T>
void GESD2D(int context, int M, int N, const T* A, int lda, int rdest, int cdest);

template <>
void GESD2D<float>(int context, int M, int N, const float* A, int lda, int rdest, int cdest)
{
    Csgesd2d(context, M, N, A, lda, rdest, cdest);
}

template <>
void GESD2D<double>(int context, int M, int N, const double* A, int lda, int rdest, int cdest)
{
    Cdgesd2d(context, M, N, A, lda, rdest, cdest);
}

template <>
void GESD2D<c32>(int context, int M, int N, const c32* A, int lda, int rdest, int cdest)
{
    Ccgesd2d(context, M, N, A, lda, rdest, cdest);
}

template <>
void GESD2D<c64>(int context, int M, int N, const c64* A, int lda, int rdest, int cdest)
{
    Czgesd2d(context, M, N, A, lda, rdest, cdest);
}

// HEEV
template <typename T, typename U>
void HEEV(
    const char* jobz, const char* uplo, const int* n, 
    T* a, const int* ia, const int* ja, const int* desca,
    U* w, T* z, const int* iz, const int* jz, const int* descz, 
    T* work, const int* lwork, U* rwork, const int* lrwork, int* info
);

template <>
void HEEV<c32, float>(
    const char* jobz, const char* uplo, const int* n, 
    c32* a, const int* ia, const int* ja, const int* desca, 
    float* w, c32* z, const int* iz, const int* jz, const int* descz, 
    c32* work, const int* lwork, float* rwork, const int* lrwork, int* info
)
{
    pcheev_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork, rwork, lrwork, info);
}

template <>
void HEEV<c64, double>(
    const char* jobz, const char* uplo, const int* n, 
    c64* a, const int* ia, const int* ja, const int* desca, 
    double* w, c64* z, const int* iz, const int* jz, const int* descz, 
    c64* work, const int* lwork, double* rwork, const int* lrwork, int* info
)
{
    pzheev_(jobz, uplo, n, a, ia, ja, desca, w, z, iz, jz, descz, work, lwork, rwork, lrwork, info);
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

        int nr = MIN(Mb, M - r);

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % pgDims.col) 
        {
            int nc = MIN(Nb, N - c);
            
            if (id.isRoot()) 
            {
                GESD2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }

            if (id.row == sendr && id.col == sendc) 
            {
                GERV2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
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

        int nr = MIN(Mb, M - r);

        for (int c = 0; c < N; c += Nb, sendc = (sendc + 1) % pgDims.col) 
        {
            int nc = MIN(Nb, N - c);

            if (id.row == sendr && id.col == sendc) 
            {
                GESD2D(ctxt, nr, nc, &local[CMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }
            
            if (id.isRoot()) 
            {
                GERV2D(ctxt, nr, nc, &global[CMIDX(r, c, M)], M, sendr, sendc);
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}

template <typename T>
void DistributedMatrix<T>::SetElements(InitializerFunc f)
{
    for (int iCol = 0; iCol < LocalDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < LocalDims.row; ++iRow)
        {
            int2 gid = LocalToGlobal(iRow, iCol, PGridDims, PGridId, m_desc);

            LocalData[CMIDX(iRow, iCol, LocalDims.row)] = f(gid);
        }
    }
}

template <typename T>
void DistributedMatrix<T>::CustomLocalOp(CustomOpFunc f)
{
    for (int iCol = 0; iCol < LocalDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < LocalDims.row; ++iRow)
        {
            int2 gid = LocalToGlobal(iRow, iCol, PGridDims, PGridId, m_desc);

            f(gid, LocalData[CMIDX(iRow, iCol, LocalDims.row)]);
        }
    }
}

template <typename T>
void DistributedMatrix<T>::Init(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    Free(LocalData);

    PGridDims = pgDims;
    PGridId = pgId;

    const int zero = 0;
    LocalDims.row = numroc_(&numRows, &rowBlockSize, &pgId.row, &zero, &pgDims.row);
    LocalDims.col = numroc_(&numCols, &colBlockSize, &pgId.col, &zero, &pgDims.col);

    LocalData = Allocate<T>(LocalDims.row, LocalDims.col);
    assert(LocalData != nullptr);

    int info;
    descinit_(m_desc, &numRows, &numCols, &rowBlockSize, &colBlockSize, &zero, &zero, &context, &LocalDims.row, &info);
    assert(info >= 0);
}

template <typename T>
void DistributedMatrix<T>::PrintLocal(std::ostream& os) const
{
    for (int i = 0; i < PGridDims.totalProcs(); ++i)
    {
        if (PGridId.flat(PGridDims.row) == i)
        {
            for (int iCol = 0; iCol < LocalDims.col; ++iCol)
            {
                for (int iRow = 0; iRow < LocalDims.row; ++iRow)
                {
                    os << std::setw(4) << std::setprecision(2) << LocalData[CMIDX(iRow, iCol, LocalDims.row)] << " ";
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
    T* global = nullptr;
    if (IsRootProcess())
    {
        global = Allocate<T>(m_desc.M, m_desc.N);
    }

    GatherMatrix(PGridId, PGridDims, global, m_desc, LocalData, LocalDims);

    if (IsRootProcess())
    {
        for (int r = 0; r < m_desc.M; ++r) 
        {
            for (int c = 0; c < m_desc.N; ++c) 
            {
                os << std::setw(4) << std::setprecision(2) << global[CMIDX(r, c, m_desc.M)] << " ";
            }
            os << std::endl;
        }
    }

    Free(global);
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    DistributedMatrix A;
    A.Init(context, pgDims, pgId, numRows, numCols, rowBlockSize, colBlockSize);
    return A;
}

template <typename T>
template <typename U>
DistributedMatrix<T> DistributedMatrix<T>::Uninitialized(const DistributedMatrix<U>& copyDims)
{
    auto& desc = copyDims.Desc();

    DistributedMatrix A;
    A.Init(desc.ctxt, copyDims.ProcGridDimensions(), copyDims.ProcGridId(), desc.M, desc.N, desc.Mb, desc.Nb);
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
            for (int numDiscards = (gid.flat(numRows) - state); numDiscards > 0; --numDiscards)
            {
                r();
            }

            state = gid.flat(numRows) + 1;
            return r() * range;
        }
    );
}

template <typename T>
DistributedMatrix<T> DistributedMatrix<T>::Duplicate(const DistributedMatrix<T>& A)
{
    DistributedMatrix<T> B;
    B.Init(A.Desc().ctxt, A.PGridDims, A.PGridId, A.Desc().M, A.Desc().N, A.Desc().Mb, A.Desc().Nb);

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
    pzlacpy_("A", &src.Desc().M, &src.Desc().M, src.Data(), &one, &one, src.Desc(), dest.Data(), &one, &one, dest.Desc());
}

struct SequenceGenerator
{
    int state = 1;

    c64 GenerateReal() { return operator()(); }
    c64 operator()() { return c64(state++, 0); }
};

template <>
DistributedMatrix<c64> DistributedMatrix<c64>::RandomHermitian(
    int context, int2 pgDims, int2 pgId, 
    int N, int rowBlockSize, int colBlockSize,
    int seed
)
{
    int state = 0;
    Random<c64> r(seed);

    auto A = Initialized(
        context, pgDims, pgId, 
        N, N, rowBlockSize, colBlockSize, 
        [&](int2 gid) 
        {
            if (gid.col > gid.row)
            {
                return c64{};
            }

            // Discard up to this global index
            for (; state < gid.flat(N); ++state)
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
    const double fone = 1.0;
    const int ione = 1;
    pztranc_(&A.Desc().M, &A.Desc().M, &fone, A.Data(), &ione, &ione, A.Desc(), &fone, B.Data(), &ione, &ione, B.Desc());

    return B;
}

template <typename T>
void DistributedMatrix<T>::GEMM(
    double alpha, const DistributedMatrix<T>& A, const DistributedMatrix<T>& B, 
    double beta, DistributedMatrix<T>& C)
{
    assert(
        A.NumCols() == B.NumRows() &&
        A.NumRows() == C.NumRows() &&
        B.NumCols() == C.NumCols()
    );

	const int one = 1;
	pdgemm_("N", "N", 
		&A.Desc().M, &B.Desc().N, &A.Desc().N, &alpha,
		A.Data(), &one, &one, A.Desc(), B.Data(), &one, &one, B.Desc(), &beta,
		C.Data(), &one, &one, C.Desc());
}

template <typename T>
template <typename U>
void DistributedMatrix<T>::HEEV(DistributedMatrix<T>& A, DistributedMatrix<U>& W, DistributedMatrix<T>& Z)
{
    assert(A.IsSquare());

    T workSize;
    U rworkSize;

	const int one = 1.0;
    int lwork = -1;
    int lrwork = -1;
    int info = 0;

    ::HEEV<T, U>(
        "V", "U", &A.Desc().N, 
        nullptr, &one, &one, A.Desc(), 
        nullptr, nullptr, &one, &one, Z.Desc(),
        &workSize, &lwork, &rworkSize, &lrwork, &info);

    lwork = static_cast<int>(real(workSize));
    lrwork = static_cast<int>(rworkSize);

    T* work = Allocate<T>(lwork);
    U* rwork = Allocate<U>(lrwork);

    ::HEEV<T, U>(
        "V", "U", &A.Desc().N, 
        A.Data(), &one, &one, A.Desc(), 
        W.Data(), Z.Data(), &one, &one, Z.Desc(), 
        work, &lwork, rwork, &lrwork, &info);

    Free(work);
    Free(rwork);
}

template <typename T>
template <typename U>
void DistributedMatrix<T>::HEEV(DistributedMatrix<T>& A, DistributedMatrix<U>& W)
{
    assert(A.IsSquare());

	const int one = 1.0;

    T workSize;
    U rworkSize;

    int lwork = -1;
    int lrwork = -1;
    int info = 0;

    ::HEEV<T, U>(
        "N", "U", &A.Desc().N, 
        nullptr, &one, &one, A.Desc(), 
        nullptr, nullptr, nullptr, nullptr, nullptr,
        &workSize, &lwork, &rworkSize, &lrwork, &info
    );

    lwork = static_cast<int>(real(workSize));
    lrwork = static_cast<int>(rworkSize);

    T* work = Allocate<T>(lwork);
    U* rwork = Allocate<U>(lrwork);

    ::HEEV<T, U>(
        "N", "U", &A.Desc().N, 
        A.Data(), &one, &one, A.Desc(), 
        W.Data(), nullptr, nullptr, nullptr, nullptr, 
        work, &lwork, rwork, &lrwork, &info);

    Free(work);
    Free(rwork);
}
