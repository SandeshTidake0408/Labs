// vector_add.cu
#include <iostream>
#include <cuda.h>
#define N 1000000

__global__ void vectorAdd(float *a, float *b, float *c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N)
        c[id] = a[id] + b[id];
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    // Allocate host memory
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c);

    // Copy result back
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Display a few results
    std::cout << "Sample results: \n";
    for (int i = 0; i < 5; i++)
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c);
    return 0;
}
