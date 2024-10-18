#include <stdio.h>
#include <stdlib.h>

#define SIZE 100

__global__ void recursive_doubling(int *input, int *output) {
	   __shared__ int temp[SIZE];
	   int i = blockIdx.x*blockDim.x + threadIdx.x;
	   if (i < SIZE) {
	      temp[threadIdx.x] = input[i];
	   }

	   for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
   	        __syncthreads();
   		temp[threadIdx.x] += temp[threadIdx.x-stride];
		}
		output[i] = temp[threadIdx.x];
}
int main(void) {
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
    recursive_doubling<<<1, SIZE>>>(d_input, d_output);

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