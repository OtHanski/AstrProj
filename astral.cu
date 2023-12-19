#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


__global__ void dotProductKernel(float (*a)[3], float (*b)[3], float (*c)[3], int size) {
    int index = threadIdx.x;
    if (index < size) {
        c[index] = a[index][0] * b[index][0] + a[index][1] * b[index][1] + a[index][2] * b[index][2];
    }
}

void dotProduct(float (*a)[3], float (*b)[3], float* c, int size) {
    float (*d_a)[3], (*d_b)[3], *d_c;
    int sizeInBytes = size * sizeof(float[3]);

    cudaMalloc((void**)&d_a, sizeInBytes);
    cudaMalloc((void**)&d_b, sizeInBytes);
    cudaMalloc((void**)&d_c, size * sizeof(float));

    cudaMemcpy(d_a, a, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeInBytes, cudaMemcpyHostToDevice);

    dotProductKernel<<<1, size>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c, d_c, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    const int size = 1000;
    float a[size][3], b[size][3], c[size][3];

    srand(time(0)); // Initialize random number generator

    for(int i = 0; i < size; i++) {
        float theta = 2 * M_PI * ((float)rand() / (float)RAND_MAX); // Random angle between 0 and 2pi
        float phi = acos(2 * ((float)rand() / (float)RAND_MAX) - 1); // Random angle between 0 and pi

        // Generate random 3D unit vector for a
        a[i][0] = sin(phi) * cos(theta);
        a[i][1] = sin(phi) * sin(theta);
        a[i][2] = cos(phi);

        theta = 2 * M_PI * ((float)rand() / (float)RAND_MAX); // Random angle between 0 and 2pi
        phi = acos(2 * ((float)rand() / (float)RAND_MAX) - 1); // Random angle between 0 and pi

        // Generate random 3D unit vector for b
        b[i][0] = sin(phi) * cos(theta);
        b[i][1] = sin(phi) * sin(theta);
        b[i][2] = cos(phi);
    }

    dotProduct(a, b, c, size);
    printf("%f\n", c[0]);
    // Print c here...

    return 0;
}