
#include <cstdlib>
#include <stdio.h>

#include <lapack.h>

void test_dgesv()
{
    double A[9] = {76, 27, 18, 25, 89, 60, 11, 51, 32};
    double b[3] = {10, 7, 43};

    int N = 3;
    int nrhs = 1;
    int lda = 3;
    int ipiv[3];
    int ldb = 3;
    int info;

    dgesv_(&N, &nrhs, A, &lda, ipiv, b, &ldb, &info);

    if(info == 0) /* succeed */
	printf("The solution is %lf %lf %lf\n", b[0], b[1], b[2]);
    else
	fprintf(stderr, "dgesv_ fails %d\n", info);
}

void test_zheev()
{
    std::complex<double> A[9];

    A[0] = {1, 0};  A[3] = {2, 0};  A[6] = {3, 0};
    A[1] = {2, 0};  A[4] = {0, 0};  A[7] = {-3, 0};
    A[2] = {3, 0};  A[5] = {-3, 0}; A[8] = {3, 0};

    constexpr int N = 3;
    int lda = 3;
    int info;

    const int lwork = 2*N-1;
    double W[N];
    std::complex<double> work[lwork];
    double rwork[3*N-2];

    zheev_("N", "U", &N, A, &lda, W, work, &lwork, rwork, &info);

    if(info == 0) /* succeed */
	printf("The solution is %lf %lf %lf\n", W[0], W[1], W[2]);
    else
	fprintf(stderr, "zheev_ fails %d\n", info);
}

void test_dgeev()
{
    double A[9];
    double b[3] = {10, 7, 43};

    A[0] = 1; A[3] = 2; A[6] = 3;
    A[1] = -1; A[4] = 0; A[7] = -3;
    A[2] = 0; A[5] = -2; A[8] = 3;

    constexpr int N = 3;
    int nrhs = 1;
    int lda = 3;
    int ldvl = 1;
    int ldvr = 1;
    int info;

    const int lwork = 3*N;
    double work[lwork];
    double wr[N];
    double wi[N];

    dgeev_("N", "N", &N, A, &lda, wr, wi, NULL, &ldvl, NULL, &ldvr, work, &lwork, &info);

    if(info == 0) /* succeed */
	printf("The solution is %lf %lf %lf\n", wi[0], wi[1], wi[2]);
    else
	fprintf(stderr, "zheev_ fails %d\n", info);
}

int main(int argc, const char** argv)
{
    test_zheev();

    return 0;
}
