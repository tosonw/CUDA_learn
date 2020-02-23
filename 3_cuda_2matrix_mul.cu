// CUDA编程之快速入门
// https://www.cnblogs.com/skyfsm/p/9673960.html

// 最后一个例子我们将计算一个更加复杂的任务，矩阵乘法
//
//回顾一下矩阵乘法：两矩阵相乘，左矩阵第一行乘以右矩阵第一列（分别相乘，第一个数乘第一个数），
// 乘完之后相加，即为结果的第一行第一列的数，依次往下算，直到计算完所有矩阵元素。
//
// 梳理一下CUDA求解矩阵乘法的思路：因为C=A×B，我们利用每个线程求解C矩阵每个(x, y)的元素，
// 每个线程载入A的一行和B的一列，遍历各自行列元素，对A、B对应的元素做一次乘法和一次加法。
//
// 从这个矩阵乘法任务可以看出，我们通过GPU进行并行计算的方式仅花费了0.5秒，
// 但是CPU串行计算方式却花费了7.6秒，计算速度提升了十多倍，可见并行计算的威力！

#define CPU

#ifdef GPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#define Row 1024
#define Col 1024


__global__ void matrix_mul_gpu(int *M, int* N, int* P, int width)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    int sum = 0;
    for(int k=0;k<width;k++)
    {
        int a = M[j*width+k];
        int b = N[k*width+i];
        sum += a*b;
    }
    P[j*width+i] = sum;
}

int main()
{
    struct timeval start, end;
    gettimeofday( &start, NULL );

    int *A = (int *)malloc(sizeof(int) * Row * Col);
    int *B = (int *)malloc(sizeof(int) * Row * Col);
    int *C = (int *)malloc(sizeof(int) * Row * Col);
    //malloc device memory
    int *d_dataA, *d_dataB, *d_dataC;
    cudaMalloc((void**)&d_dataA, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataB, sizeof(int) *Row*Col);
    cudaMalloc((void**)&d_dataC, sizeof(int) *Row*Col);
    //set value
    for (int i = 0; i < Row*Col; i++) {
        A[i] = 90;
        B[i] = 10;
    }

    cudaMemcpy(d_dataA, A, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataB, B, sizeof(int) * Row * Col, cudaMemcpyHostToDevice);
    dim3 threadPerBlock(16, 16);
    dim3 blockNumber((Col+threadPerBlock.x-1)/ threadPerBlock.x, (Row+threadPerBlock.y-1)/ threadPerBlock.y );
    printf("Block(%d,%d)   Grid(%d,%d).\n", threadPerBlock.x, threadPerBlock.y, blockNumber.x, blockNumber.y);
    matrix_mul_gpu << <blockNumber, threadPerBlock >> > (d_dataA, d_dataB, d_dataC, Col);
    //拷贝计算数据-一级数据指针
    cudaMemcpy(C, d_dataC, sizeof(int) * Row * Col, cudaMemcpyDeviceToHost);

    //释放内存
    free(A);
    free(B);
    free(C);
    cudaFree(d_dataA);
    cudaFree(d_dataB);
    cudaFree(d_dataC);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    printf("total time is %d ms\n", timeuse/1000);

    return 0;
}

//Block(16,16)   Grid(64,64).
//total time is 256 ms

#endif
#ifdef CPU

#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#define ROWS 1024
#define COLS 1024

using namespace std;

void matrix_mul_cpu(float* M, float* N, float* P, int width)
{
    for(int i=0;i<width;i++)
        for(int j=0;j<width;j++)
        {
            float sum = 0.0;
            for(int k=0;k<width;k++)
            {
                float a = M[i*width+k];
                float b = N[k*width+j];
                sum += a*b;
            }
            P[i*width+j] = sum;
        }
}

int main()
{
    struct timeval start, end;
    gettimeofday( &start, NULL );
    float *A, *B, *C;
    int total_size = ROWS*COLS*sizeof(float);
    A = (float*)malloc(total_size);
    B = (float*)malloc(total_size);
    C = (float*)malloc(total_size);

    //CPU一维数组初始化
    for(int i=0;i<ROWS*COLS;i++)
    {
        A[i] = 80.0;
        B[i] = 20.0;
    }

    matrix_mul_cpu(A, B, C, COLS);

    gettimeofday( &end, NULL );
    int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
    cout << "total time is " << timeuse/1000 << "ms" <<endl;

    return 0;
}

//total time is 18087ms

#endif

