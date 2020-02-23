/***************************************************************************
*1.将数据从主机内存数据复制到设备显存
*2.写好核函数
*3.CUDA编译器执行核函数 在GPU上完成计算操作
*4.把显存数据复制到主机内存
*5.释放显存空间
/***************************************************************************/

#include <stdio.h>
#include <stdlib.h>

//CUDA RunTime API
#include <cuda_runtime.h>
#define DATA_SIZE 1048576
int data[DATA_SIZE];

//产生大量0-9之间的随机数
void GenerateNumbers(int *number, int size)
{
	for (int i = 0; i < size; i++) {
		number[i] = rand() % 10;
	}
}

//CUDA 初始化
bool InitCUDA()
{
	int count;
	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

// __global__ 函数 (GPU上执行) 计算立方和 
//核函数是不可以有返回值类型的
__global__ static void sumOfSquares( int *num, int* result)
{
	int sum = 0;
	int i;
	for (i = 0; i < DATA_SIZE; i++) {
		sum += num[i] * num[i] * num[i];
	}
	*result = sum;
}

int main()
{
	//CUDA 初始化
	if (!InitCUDA()) {
		return 0;
	}
	//生成随机数
	GenerateNumbers(data, DATA_SIZE);
	/*把数据复制到显卡内存中*/
	int* gpudata, *result;
	//cudaMalloc 取得一块显卡内存 ( 其中result用来存储计算结果 )
	cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
	cudaMalloc((void**)&result, sizeof(int));
	//cudaMemcpy 将产生的随机数复制到显卡内存中 
	//cudaMemcpyHostToDevice - 从内存复制到显卡内存
	//cudaMemcpyDeviceToHost - 从显卡内存复制到内存
	cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);
	// 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
	sumOfSquares << <1, 1, 0 >> >(gpudata, result);
	/*把结果从显示芯片复制回主内存*/
	int sum;
	//cudaMemcpy 将结果从显存中复制回内存
	cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
	//Free
	cudaFree(gpudata);
	cudaFree(result);
	printf("GPUsum: %d \n", sum);
	sum = 0;
	for (int i = 0; i < DATA_SIZE; i++) {
		sum += data[i] * data[i] * data[i];
	}
	printf("CPUsum: %d \n", sum);
	return 0;
}
