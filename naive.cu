#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 10000
#define BLOCKSIZE 256

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// CUDA kernel for naive scan
__global__ void naive_scan(int* input, int* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < SIZE) {
    int value = 0;
        for (int j = 0; j <= i; j++) {
            value += input[j];
        }
        output[i] = value;
    }
}

int main() {
    // Allocate host memory
    int* h_input = (int*)malloc(sizeof(int) * SIZE);
    int* h_output = (int*)malloc(sizeof(int) * SIZE);

    // Initialize input on host
    for (int i = 0; i < SIZE; i++) {
        h_input[i] = 1;
    }

    // Allocate device memory
    int* d_input;
    int* d_output;
    cudaMalloc((void**)&d_input, sizeof(int) * SIZE);
    cudaMalloc((void**)&d_output, sizeof(int) * SIZE);

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    double t0,t1;
    t0=get_clock();
    int numSize = (SIZE + BLOCKSIZE-1)/BLOCKSIZE;
    naive_scan<<<numSize, BLOCKSIZE>>>(d_input, d_output);
    t1=get_clock();
    
    // Copy output data back to host
    cudaMemcpy(h_output, d_output, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    printf("time per call: %f ns\n", (1000000000.0*(t1-t0)));
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
