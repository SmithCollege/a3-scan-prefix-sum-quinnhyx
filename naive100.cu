#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define SIZE 100

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

    // Launch the scan kernel with threads (1 block)
    naive_scan<<<1, SIZE>>>(d_input, d_output);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < SIZE; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
