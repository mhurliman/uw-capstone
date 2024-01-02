#include "DistributedMatrix.h"

#include <assert.h>
#include <iostream>
#include <math.h>
#include <iomanip>

const int DistributedMatrix::s_Zero;

void ScatterMatrix(int2 id, int2 pgDims, const double* globalA, const MatDesc& descA, double* localA, int2 localDims)
{
    int M = descA.M;
    int N = descA.N;
    int Mb = descA.Mb;
    int Nb = descA.Nb;
    int ctxt = descA.ctxt;

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
                Cdgesd2d(ctxt, nr, nc, &globalA[RMIDX(r, c, M)], M, sendr, sendc);
            }

            if (id.row == sendr && id.col == sendc) 
            {
                Cdgerv2d(ctxt, nr, nc, &localA[RMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}

void GatherMatrix(int2 id, int2 pgDims, double* globalA, const MatDesc& descA, const double* localA, int2 localDims)
{
    int M = descA.M;
    int N = descA.N;
    int Mb = descA.Mb;
    int Nb = descA.Nb;
    int ctxt = descA.ctxt;

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
                Cdgesd2d(ctxt, nr, nc, &localA[RMIDX(recvr, recvc, localDims.row)], localDims.row, 0, 0);
                recvc = (recvc + nc) % localDims.col;
            }
            
            if (id.isRoot()) 
            {
                Cdgerv2d(ctxt, nr, nc, &globalA[RMIDX(r, c, M)], M, sendr, sendc);
            }
        }

        if (id.row == sendr)
            recvr = (recvr + nr) % localDims.row;
    }
}

void DistributedMatrix::SetElements(InitializerFunc f)
{
    for (int iCol = 0; iCol < LocalDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < LocalDims.row; ++iRow)
        {
            int2 gid = LocalToGlobal(iRow, iCol, PGridDims, PGridId, m_desc);

            LocalData[RMIDX(iRow, iCol, LocalDims.row)] = f(gid);
        }
    }
}

void DistributedMatrix::CustomLocalOp(CustomOpFunc f)
{
    for (int iCol = 0; iCol < LocalDims.col; ++iCol)
    {
        for (int iRow = 0; iRow < LocalDims.row; ++iRow)
        {
            int2 gid = LocalToGlobal(iRow, iCol, PGridDims, PGridId, m_desc);

            f(gid, LocalData[RMIDX(iRow, iCol, LocalDims.row)]);
        }
    }
}

void DistributedMatrix::Init(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    PGridDims = pgDims;
    PGridId = pgId;

    LocalDims.row = numroc_(&numRows, &rowBlockSize, &pgId.row, &s_Zero, &pgDims.row);
    LocalDims.col = numroc_(&numCols, &colBlockSize, &pgId.col, &s_Zero, &pgDims.col);

    LocalData = Allocate<double>(LocalDims.row, LocalDims.col);
    assert(LocalData != nullptr);

    int info;
    descinit_(m_desc, &numRows, &numCols, &rowBlockSize, &colBlockSize, &s_Zero, &s_Zero, &context, &LocalDims.row, &info);
    assert(info >= 0);
}

void DistributedMatrix::PrintLocal(std::ostream& os) const
{
    for (int i = 0; i < PGridDims.totalProcs(); ++i)
    {
        if (PGridId.flat(PGridDims.row) == i)
        {
            for (int iCol = 0; iCol < LocalDims.col; ++iCol)
            {
                for (int iRow = 0; iRow < LocalDims.row; ++iRow)
                {
                    os << std::setw(3) << round(LocalData[RMIDX(iRow, iCol, LocalDims.row)]) << " ";
                }
                os << std::endl;
            }
        }

        Cblacs_barrier(m_desc.ctxt, "All");
    }
}

void DistributedMatrix::PrintGlobal(std::ostream& os) const
{
    double* global = nullptr;
    if (IsRootProcess())
    {
        global = Allocate<double>(m_desc.M, m_desc.N);
    }

    GatherMatrix(PGridId, PGridDims, global, m_desc, LocalData, LocalDims);

    if (IsRootProcess())
    {
        for (int r = 0; r < m_desc.M; ++r) 
        {
            for (int c = 0; c < m_desc.N; ++c) 
            {
                os << std::setw(3) << round(global[RMIDX(r, c, m_desc.M)]) << " ";
            }
            os << std::endl;
        }
    }

    Free(global);
}

DistributedMatrix DistributedMatrix::Uninitialized(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize
)
{
    DistributedMatrix A;
    A.Init(context, pgDims, pgId, numRows, numCols, rowBlockSize, colBlockSize);
    return A;
}

DistributedMatrix DistributedMatrix::Initialized(
    int context, int2 pgDims, int2 pgId, 
    int numRows, int numCols, int rowBlockSize, int colBlockSize,
    InitializerFunc f
)
{
    auto A = Uninitialized(context, pgDims, pgId, numRows, numCols, rowBlockSize, colBlockSize);
    A.SetElements(f);
    return A;
}

DistributedMatrix DistributedMatrix::Identity(
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

void DistributedMatrix::GEMM(double alpha, const DistributedMatrix& A, const DistributedMatrix& B, double beta, DistributedMatrix& C)
{
	const int one = 1.0;
	pdgemm_("No Transpose", "No Transpose", 
		&A.Desc().M, &B.Desc().N, &A.Desc().N, &alpha,
		A.Data(), &one, &one, A.Desc(), B.Data(), &one, &one, B.Desc(), &beta,
		C.Data(), &one, &one, C.Desc());
}

std::ostream& operator<<(std::ostream& os, const DistributedMatrix& m)
{
    m.PrintGlobal(os);
    return os;
}