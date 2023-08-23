#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define memSharing 0
// const int N = 1e1;

__global__ void testing(){

	printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d, blockIdx.y = %d, blockDim.y = %d, threadIdx.y = %d, gridDim.x = %d, gridDim.y = %d\n", blockIdx.x, blockDim.x, threadIdx.x, blockIdx.y, blockDim.y, threadIdx.y, gridDim.x, gridDim.y);

	__syncthreads();
    return;
}

int main(){

    // int B = 2, T = N/B;
	dim3 B(2, 2);
	dim3 T(2, 4);

    testing<<<1,T>>>();

    return 0;
}