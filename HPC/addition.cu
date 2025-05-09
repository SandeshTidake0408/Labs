#include<iostream>
#include<math.h>
#include<time.h>
#include "cuda_runtime.h" // For CUDA API functions
using namespace std;

void cpuSum(int* A, int*B, int* C, int N){
    for (int i=0; i<N; ++i){
        C[i] = A[i] + B[i];
    }
}
    
//The important part is the __global__ keyword, which tells the CUDA compiler that this function is a kernel (i.e., it runs on the GPU
__global__ void kernel(int* A, int* B, int* C, int N){ //
    
    //Computes the global index i of the thread:
    // blockIdx.x: Current block number
    // blockDim.x: Threads per block
    // threadIdx.x: Thread number within the block
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N){
        C[i] = A[i] + B[i];
    }
}
    
void gpuSum(int* A, int* B, int* C, int N) {

    int threadsPerBlock = min(1024, N); //CUDA allows max 1024 threads per block.
    int blocksPerGrid = ceil(double(N)/double(threadsPerBlock)); //Calculates how many blocks are needed to cover all N elements.
    kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N); //Launches the kernel on the GPU.

}
    
bool isVectorEqual(int* A, int* B, int N) {
    for (int i=0; i<N; ++i){
        if (A[i] != B[i]){
            return false;
        }
    }
    return true;
}
   

int main()
    {
    int N = 2e7; //20 million
    int *A, *B, *C, *D, *a, *b, *c;
    int size = N * sizeof(int); //Total memory needed per array in bytes.
    
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    D = (int*)malloc(size);
    
    for (int i=0; i<N; ++i) {
        A[i] = rand()%1000;
        B[i] = rand()%1000;
    }
        
    clock_t start, end;
    
    start = clock();
    cpuSum(A, B, C, N);
    end = clock();
    float timeTakenCPU = ((float)(end-start))/CLOCKS_PER_SEC; // (end-start)give clock ticks and convertd in to sec by dividing
    
    cudaMalloc(&a, size);
    cudaMalloc(&b, size);
    cudaMalloc(&c, size);
    
    cudaMemcpy(a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b, B, size, cudaMemcpyHostToDevice);
    
    start = clock();
    gpuSum(a, b, c, N);
    cudaDeviceSynchronize(); // Wait for GPU to finish
    cudaMemcpy(D, c, size, cudaMemcpyDeviceToHost);
    end = clock();
    float timeTakenGPU = ((float)(end-start))/CLOCKS_PER_SEC;
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    
    bool success = isVectorEqual(C, D, N);
    
    printf("Vector Addition\n");
    printf("--------------------\n");
    printf("CPU Time: %f \n", timeTakenCPU);
    printf("GPU Time: %f \n", timeTakenGPU);
    printf("Speed Up: %f \n", timeTakenCPU/timeTakenGPU);
    printf("Verification: %s \n", success ? "true":"false");
}
