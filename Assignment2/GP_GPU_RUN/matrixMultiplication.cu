#include <cassert>
#include <iostream>
#include <time.h>
using namespace std;

const bool SHARED = true;

// Matrix dimensions
const int M = 1e2 + 7;
const int N = 1e2 + 9;
const int K = 1e2 + 11;

// Threads per CTA dimension
const int THREADS = 128;

// Padded matrix dimensions
const int M_padded = M + THREADS - M % THREADS;
const int N_padded = N + THREADS - N % THREADS;
const int K_padded = K + THREADS - K % THREADS;

// Size of shared memory per TB
const int SHMEM_SIZE = THREADS * THREADS;

__global__ void matrixMul(const int *a, const int *b, int *c) {

    //Shared Memory Approach
    if(SHARED){
        // Compute each thread's global row and column index
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Statically allocated shared memory
        __shared__ int s_a[SHMEM_SIZE];
        __shared__ int s_b[SHMEM_SIZE];

        int tmp = 0;

        // Sweep tile across matrix
        for (int i = 0; i < K_padded; i += blockDim.x) {
            // Load in elements for this tile
            s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * K + i + threadIdx.x];
            s_b[threadIdx.y * blockDim.x + threadIdx.x] = b[i * N + threadIdx.y * N + col];

            // Wait for both tiles to be loaded in before doing computation
            __syncthreads();

            // Do matrix multiplication on the small matrix
            for (int j = 0; j < blockDim.x; j++) {
            tmp += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
            }

            // Wait for all threads to finish using current tiles before loading in new
            // ones
            __syncthreads();
        }

        // Write back results
        if (row < M && col < N) c[row * N + col] = tmp;
    }
    
    //Naive(Global Memory) Approach
    else{
        // Compute each thread's global row and column index
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        // Iterate over row, and down column
        int temp = 0;
        for (int k = 0; k < K; k++) {
            // Accumulate results for a single element
            temp += a[row * K + k] * b[k * N + col];
        }
        if (row < M && col < N) c[row * N + col] = temp;
    }
    return;
}

// Checking result on the CPU
void verify_result(int* a, int* b, int* c) {
    clock_t start_time, end_time;
    printf("CPU verification has started\n");
    start_time = clock();

    for (int row = 0; row < M_padded; row++) {
        if (row >= M) continue;
        for (int col = 0; col < N_padded; col++) {
            if (col >= N) continue;
            int tmp = 0;
            for (int i = 0; i < K_padded; i++) {
                tmp += a[row * K + i] * b[i * N + col];
            }

            assert(tmp == c[row * N + col]);
        }
    }
    end_time = clock();
    cout << "Result verified by CPU\n";
    printf("Time taken by CPU : %f ms\n", (((double)end_time-start_time)/CLOCKS_PER_SEC)*100);

}

int main() {

    if(SHARED){
        cout<<"Shared Memory Approach"<<endl;
    }
    else{
        cout<<"Naive Approach"<<endl;
    }

    // Size (in bytes) of matrix
    // MxN = MxK * KxN
    size_t bytes_a = M_padded * K_padded * sizeof(int);
    size_t bytes_b = K_padded * N_padded * sizeof(int);
    size_t bytes_c = M * N * sizeof(int);

    int *h_a = (int *)malloc(bytes_a);
    int *h_b = (int *)malloc(bytes_b);
    int *h_c = (int *)malloc(bytes_c);

    srand(time(NULL));
    // Initialize matrices
    for (int i = 0; i < M_padded; i++) {
    for (int j = 0; j < K_padded; j++) {
        if (i < M && j < K) h_a[i * K + j] = rand() % 100;
    }
    }

    for (int i = 0; i < K_padded; i++) {
    for (int j = 0; j < N_padded; j++) {
        if (i < K && j < N) h_b[i * N + j] = rand() % 100;
    }
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c, bytes_c);

    // Copy data to the device
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    // Blocks per grid dimension (assumes THREADS divides M and N evenly)
    int BLOCKS_X = N_padded / THREADS;
    int BLOCKS_Y = M_padded / THREADS;

    // Use dim3 structs for block  and grid dimensions
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    cudaEvent_t start, stop;
    float time_taken;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cout << "GPU code has started"<<endl;
    cudaEventRecord(start,0);

    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    cout << "GPU code has finished"<<endl;
    cout<<"Time taken: "<<time_taken<<"ms"<<endl;

    // Copy back to the host
    cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost);

    // Check result
    verify_result(h_a, h_b, h_c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
