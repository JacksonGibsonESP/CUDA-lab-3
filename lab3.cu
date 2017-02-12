#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
//#include <chrono>

using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define LIMIT(a,b,c) (MIN(MAX((a),(b)),(c)))

#define CSC(call) {														\
    cudaError err = call;												\
    if(err != cudaSuccess) {											\
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",	\
            __FILE__, __LINE__, cudaGetErrorString(err));				\
        exit(1);														\
				    }													\
} while (0)

__global__ void kernel_histogram(unsigned char *src, int length, unsigned int *histogram)
{
	__shared__ unsigned int local_histogram[256];
	if (threadIdx.x == 0)
	{
		memset(&local_histogram, 0, sizeof(unsigned int) * 256);
	}
	__syncthreads();
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	while (tid < length)
	{
		atomicAdd(&local_histogram[src[tid]], 1);
		tid += gridDim.x * blockDim.x;
	}
	__syncthreads();
	atomicAdd(&histogram[threadIdx.x], local_histogram[threadIdx.x]);
}

__global__ void kernel_scan(unsigned int *histogram)
{
	__shared__ unsigned int local_histogram[256];
	if (threadIdx.x == 0)
	{
		memcpy(&local_histogram, histogram, sizeof(unsigned int) * 256);
	}
	__syncthreads();
	int pow = 2;
	while (pow <= 256)
	{
		if (threadIdx.x * pow + pow - 1 < 256)
		{
			local_histogram[threadIdx.x * pow + pow - 1] += local_histogram[threadIdx.x * pow + pow - pow / 2 - 1];
		}
		__syncthreads();
		pow *= 2;
	}
	if (threadIdx.x == 0)
	{
		local_histogram[255] = 0;
	}
	__syncthreads();

	pow = 256;
	while (pow > 1)
	{
		if (threadIdx.x * pow + pow - 1 < 256)
		{
			unsigned int sum = local_histogram[threadIdx.x * pow + pow - 1] + local_histogram[threadIdx.x * pow + pow - pow / 2 - 1];
			local_histogram[threadIdx.x * pow + pow - pow / 2 - 1] = local_histogram[threadIdx.x * pow + pow - 1];
			local_histogram[threadIdx.x * pow + pow - 1] = sum;
		}
		__syncthreads();
		pow /= 2;
	}
	histogram[threadIdx.x] += local_histogram[threadIdx.x];
}

int main()
{
	int length = 0;
	fread(&length, sizeof(int), 1, stdin); 
	
	if (length != 0)
	{
		unsigned char *src = (unsigned char *)malloc(sizeof(unsigned char) * length);
		unsigned char *dst = (unsigned char *)malloc(sizeof(unsigned char) * length);
		fread(src, sizeof(unsigned char), length, stdin);

		unsigned char *src_dev;
		CSC(cudaMalloc(&src_dev, sizeof(unsigned char) * length));
		CSC(cudaMemcpy(src_dev, src, sizeof(unsigned char) * length, cudaMemcpyHostToDevice));

		unsigned int *histogram_dev;
		CSC(cudaMalloc(&histogram_dev, sizeof(unsigned int) * 256));
		cudaMemset(histogram_dev, 0, sizeof(unsigned int) * 256);
		//auto start_time = chrono::high_resolution_clock::now();
		kernel_histogram << <32, 256 >> >(src_dev, length, histogram_dev);

		kernel_scan << <1, 256 >> >(histogram_dev);

		unsigned int histogram[256];
		CSC(cudaMemcpy(&histogram, histogram_dev, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost));

		for (int i = 0; i < length; i++)
		{
			dst[histogram[src[i]] - 1] = src[i];
			histogram[src[i]]--;
		}
		//auto end_time = chrono::high_resolution_clock::now();
		//cout << '\n' << chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count() << "ms\n";
		fwrite(dst, sizeof(unsigned char), length, stdout);

		free(dst);
		free(src);
	}
	return 0;
}