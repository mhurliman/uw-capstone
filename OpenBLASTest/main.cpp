
#include <iostream>
#include <cblas.h>

int main(int argc, char* args[])
{
    double A[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};         
    double B[6] = {1.0,2.0,1.0,-3.0,4.0,-1.0};  
    double C[9] = {.5,.5,.5,.5,.5,.5,.5,.5,.5}; 
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,3,3,2,1,A, 3, B, 3,2,C,3);

    for(int i = 0; i < 9; ++i)
    {
        printf("%lf ", C[i]);
        if ((i + 1) % 3 == 0)
            printf("\n");
    }
    printf("\n");

    printf("Hello world");

    return 0;
}
