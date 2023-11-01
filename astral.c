#include <stdio.h>
#include <cuda.h>

__global__ void dotProductKernel(float* a, float* b, float* c, int size) {
    int index = threadIdx.x;
    if (index < size) {
        c[index] = a[index] * b[index];
    }
}

void dotProduct(float* a, float* b, float* c, int size) {
    float *d_a, *d_b, *d_c;
    int sizeInBytes = size * sizeof(float);

    cudaMalloc((void**)&d_a, sizeInBytes);
    cudaMalloc((void**)&d_b, sizeInBytes);
    cudaMalloc((void**)&d_c, sizeInBytes);

    cudaMemcpy(d_a, a, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeInBytes, cudaMemcpyHostToDevice);

    dotProductKernel<<<1, size>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c, d_c, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    int size = 1000;
    float a[size], b[size], c[size];

    // Initialize a and b here...

    dotProduct(a, b, c, size);

    // Print c here...

    return 0;
}