#include <stdio.h>
#include <cuda.h>
#include <time.h>

const bool memSharing = false;
#define N (int)1e2

__global__ void vectAdd(int *a, int *b, int *c){

        int tID = blockIdx.x * blockDim.x + threadIdx.x;
        if(tID < N){
            c[tID] = a[tID] + b[tID];
        }
        __syncthreads();
    
    return;
}

int main(){
    // printf("Enter the number of vectors: ");
    // scanf("%d", &N);

    int *a, *b, *c;
    size_t n = N*sizeof(int);

    a = (int*)malloc(n);
    b = (int*)malloc(n);
    c = (int*)malloc(n);

    srand(time(NULL));
    srand(time(NULL));
    for(int i=0; i<N; i++){
        a[i] = rand()%100;
        b[i] = rand()%100;
    }

    // clock_t start_time, end_time;
    // printf("CPU code has started\n");
    // start_time = clock();
    // for(int i=0; i<N; i++){
    //     c[i] = a[i] + b[i];
    // }
    // end_time = clock();
    // printf("Time taken by CPU : %f\n", ((double)end_time-start_time)/CLOCKS_PER_SEC);

    printf("Preparing for GPU code\n");

    int *cudaA, *cudaB, *cudaC;

    cudaMalloc(&cudaA, n);
    cudaMalloc(&cudaB, n);
    cudaMalloc(&cudaC, n);

    cudaMemcpy(cudaA, a, n, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, n, cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaC, c, n, cudaMemcpyHostToDevice);
    // cudaMemcpy(cudaC, c, n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time_taken;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    int B = 10, T = N/B;
    printf("GPU code has started\n");
    cudaEventRecord(start,0);

    vectAdd<<<B,T>>>(cudaA, cudaB, cudaC);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_taken, start, stop);
    printf("GPU code has ended\n");

    cudaMemcpy(c, cudaC, n, cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++){
    //     printf("%d+%d=%d\n",a[i],b[i],c[i]);
    // }

    printf("%d+%d=%d\n",a[N-1],b[N-1],c[N-1]);

    printf("Time taken by GPU : %f\n", time_taken);

    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
