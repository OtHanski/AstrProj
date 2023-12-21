#define _USE_MATH_DEFINES
#define samplenumber 35000
#define debug 0
#define rawdata 0
#define binsize 0.05f
#define binnumber 720

#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>


__global__ void dotProductKernel(float *a, float *b, float* result, int* histogram, int size) {
    // Compute dot product of two arrays of vectors w/ GPU, kernel function
    // Then take acos of dot product to get angle
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    // Avoid calculating the same dot product twice (or dot product with itself)
    if (i < size+1 && j > i) {
        float res =  fabsf(acosf(a[3*i] * b[3*j] + a[3*i+1] * b[3*j+1] + a[3*i+2] * b[3*j+2]))/(M_PI/180.0f);
        result[i * size + j] = res;
        if (res/binsize < binnumber) {
            //atomicAdd(&histogram[(int)(res/binsize)], 1);
        }
    }
    // Assign -360 to diagonal and duplicate values for filtering
    if (i < size+1 && j <= i) {
        result[i * size + j] = -360.0f;
    }
}

void calculateDotProducts(float *D, float *R, float *result, int *histogram, int samples) {
    float *d_D, *d_R, *d_result;
    int *d_histogram;

    
    // Allocate memory on the GPU
    cudaMalloc(&d_D, samples * 3 * sizeof(float));
    cudaMalloc(&d_R, samples * 3 * sizeof(float));
    cudaMalloc(&d_result, samples * samples * sizeof(float));
    cudaMalloc(&d_histogram, binnumber * sizeof(int));
    

    // Copy the input data to the GPU
    cudaMemcpy(d_D, D, samples * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, samples * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threadsPerBlock(16,16);
    dim3 blocksPerGrid((samples + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (samples + threadsPerBlock.y - 1) / threadsPerBlock.y);
    dotProductKernel<<<blocksPerGrid, threadsPerBlock>>>(d_D, d_R, d_result, d_histogram, samples);

    // Copy the results back to the CPU
    cudaMemcpy(result, d_result, samples * samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory on the GPU
    cudaFree(d_D);
    cudaFree(d_R);
    cudaFree(d_result);
    
}

void read_tsv(const char *file_path, float *arr1, float *arr2, int size) {
    // Read data from tab-separated file, store in arrays
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        printf("Cannot open file\n");
        exit(0);
    }

    for(int i = 0; i < size; i++) {
        if(fscanf(file, "%f\t%f\n", &arr1[i], &arr2[i]) != 2) {
            printf("Error reading line %d\n", i+1);
            exit(0);
        }
    }

    fclose(file);
}

void write_tsv(const char *file_path, float *array, int samples) {
    FILE *file = fopen(file_path, "w");
    if (file == NULL) {
        printf("Cannot open file\n");
        exit(0);
    }

    // Create a buffer to hold the data
    char *buffer = (char*) malloc(samples * samples * sizeof(float) * 4 + samples * 2);
    char *p = buffer;

    // Write the data to the buffer
    for(int i = 0; i < samples * samples; i++) {
        p += sprintf(p, "%f\n", array[i]);
    }

    // Write the buffer to the file
    fputs(buffer, file);

    // Free the buffer
    free(buffer);

    fclose(file);
}

void write_tsv(const char *file_path, int *array, int samples) {
    FILE *file = fopen(file_path, "w");
    if (file == NULL) {
        printf("Cannot open file\n");
        exit(0);
    }

    // Create a buffer to hold the data
    char *buffer = (char*) malloc(samples *  sizeof(int) * 4 + samples * 2);
    char *p = buffer;

    p += sprintf(p, "binsize = %f\n", binsize);
    // Write the data to the buffer
    for(int i = 0; i < samples; i++) {
        p += sprintf(p, "%i\n", array[i]);
    }

    // Write the buffer to the file
    fputs(buffer, file);

    // Free the buffer
    free(buffer);

    fclose(file);
}

void write_debug(const char *file_path, float (*D)[3], float (*R)[3], int samples) {
    FILE *file = fopen(file_path, "w");
    if (file == NULL) {
        printf("Cannot open file\n");
        exit(0);
    }

    fprintf(file, "D array:\n");
    for(int i = 0; i < samples; i++) {
        fprintf(file, "%f\t%f\t%f\n", D[i][0], D[i][1], D[i][2]);
    }

    fprintf(file, "\nR array:\n");
    for(int i = 0; i < samples; i++) {
        fprintf(file, "%f\t%f\t%f\n", R[i][0], R[i][1], R[i][2]);
    }

    fclose(file);
}


int main() {
    printf("Running...\n");
    // Read data from file, store in arrays
    // samples is the number of samples, change this to 1000 for testing, 100k for final run
    const int samples = samplenumber;
    // threads is the number of CUDA threads
    //const int threads = 1000;
    // Initialize arrays, thetaD and phiD are the declination and right ascension of the data points
    float thetaD[samples], phiD[samples], thetaR[samples], phiR[samples];

    // Read real data from file
    read_tsv("data_100k_arcmin.txt", thetaD, phiD, samples);
    // Read synthetic data from file
    read_tsv("flat_100k_arcmin.txt", thetaR, phiR, samples);
    // Vector arrays for D and R
    float D[samples][3], R[samples][3];
    float* Dflat = (float*)malloc(samples * 3 * sizeof(float));
    float* Rflat = (float*)malloc(samples * 3 * sizeof(float));
    // Result matrices (dot products => angles)
    float* DD = (float*)malloc((unsigned long long)samples * samples * sizeof(float));
    float* DR = (float*)malloc((unsigned long long)samples * samples * sizeof(float));
    float* RR = (float*)malloc((unsigned long long)samples * samples * sizeof(float));


    //Initialize histogram arrays
    int histogramDD[binnumber];
    int histogramDR[binnumber];
    int histogramRR[binnumber];
    //initialize distribution array
    float w_i[binnumber];
    
    // Convert spherical coordinates to cartesian coordinates for D and R
    for (int i = 0; i < samples; i++) {
        // Convert arcminutes to radians
        float thetaD_rad = thetaD[i] / (180.0f * 60.0f / M_PI);
        float phiD_rad = phiD[i] / (180.0f * 60.0f / M_PI);
        float thetaR_rad = thetaR[i] / (180.0f * 60.0f / M_PI);
        float phiR_rad = phiR[i] / (180.0f * 60.0f / M_PI);

        D[i][0] = cos(thetaD[i]) * cos(phiD[i]);
        D[i][1] = cos(thetaD[i]) * sin(phiD[i]);
        D[i][2] = sin(thetaD[i]);
        
        R[i][0] = cos(thetaR[i]) * cos(phiR[i]);
        R[i][1] = cos(thetaR[i]) * sin(phiR[i]);
        R[i][2] = sin(thetaR[i]);
    }
    // Flatten D and R arrays
    for (int i = 0; i < samples; i++) {
        Dflat[3*i] = D[i][0];
        Dflat[3*i+1] = D[i][1];
        Dflat[3*i+2] = D[i][2];
        Rflat[3*i] = R[i][0];
        Rflat[3*i+1] = R[i][1];
        Rflat[3*i+2] = R[i][2];
    }

    if (debug){write_debug("debug.dat", D, R, samples);}

    //calculate dot products DD, DR, RR
    printf("Calculating DD...");
    double start = clock();
    calculateDotProducts(Dflat, Dflat, DD, histogramDD, samples);
    double end = clock();
    printf("Done, Time taken: %f, ", (double)(end - start) / CLOCKS_PER_SEC);
    if (rawdata){write_tsv("DD.dat", DD, samples);}
    for (int i = 0; i < (unsigned long long)samples * samples; i++) {
        if (DD[i]/binsize < binnumber) {
            histogramDD[(int)(DD[i]/binsize)]++;
        }
    }
    free(DD);
    start = clock();
    printf(" File write time: %f\n", (double)(start - end) / CLOCKS_PER_SEC);

    printf("Calculating DR...");
    calculateDotProducts(Dflat, Rflat, DR, histogramDR, samples);
    end = clock();
    printf("Done, Time taken: %f, ", (double)(end - start) / CLOCKS_PER_SEC);
    if (rawdata){write_tsv("DR.dat", DR, samples);}
    for (int i = 0; i < (unsigned long long)samples * samples; i++) {
        if (DR[i]/binsize < binnumber) {
            histogramDR[(int)(DD[i]/binsize)]++;
        }
    }
    free(DR);
    start = clock();
    printf(" File write time: %f\n", (double)(start - end) / CLOCKS_PER_SEC);

    printf("Calculating RR...");
    calculateDotProducts(Rflat, Rflat, RR,histogramRR, samples);
    end = clock();
    printf("Done, Time taken: %f, ", (double)(end - start) / CLOCKS_PER_SEC);
    if (rawdata){write_tsv("RR.dat", RR, samples);}
    start = clock();
    for (int i = 0; i < (unsigned long long)samples * samples; i++) {
        if (RR[i]/binsize < binnumber) {
            histogramRR[(int)(DD[i]/binsize)]++;
        }
    }
    free(RR);
    printf(" File write time: %f\n", (double)(start - end) / CLOCKS_PER_SEC);

    // Increment histogram values for DD, DR, RR
    /*
    printf("Calculating histogram...");
    for (int i = 0; i < (unsigned long long)samples * samples; i++) {
        if (DD[i]/binsize < binnumber) {
            histogramDD[(int)(DD[i]/binsize)]++;
        }
        if (DR[i]/binsize < binnumber) {
            histogramDR[(int)(DR[i]/binsize)]++;
        }
        if (RR[i]/binsize < binnumber) {
            histogramRR[(int)(RR[i]/binsize)]++;
        }
    }*/

    // Calculate w_i
    for (int i = 0; i < binnumber; i++) {
        w_i[i] = (histogramDD[i] - 2*histogramDR[i] + histogramRR[i])/((float)histogramRR[i]+1E-20);
    }
    end = clock();
    printf(" Done, Time taken: %f, ", (double)(end - start) / CLOCKS_PER_SEC);

    // Write histogram to file
    write_tsv("histogramDD.dat", histogramDD, binnumber);
    write_tsv("histogramDR.dat", histogramDR, binnumber);
    write_tsv("histogramRR.dat", histogramRR, binnumber);
    // Write w_i to file manually
    FILE *file = fopen("w_i.dat", "w");
    if (file == NULL) {
        printf("Cannot open file\n");
        exit(0);
    }
    fprintf(file, "binsize = %f\n", binsize);
    for (int i = 0; i < binnumber; i++) {
        fprintf(file, "%f\n", w_i[i]);
    }
    fclose(file);
    printf("All done!\n");


    // Free memory
    free(DR);
    free(RR);
    free(Dflat);
    free(Rflat);

    return 0;
}