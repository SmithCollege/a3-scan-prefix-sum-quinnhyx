#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 100
#define BLOCKSIZE 256

double get_clock() {
 struct timeval tv; int ok;
 ok = gettimeofday(&tv, (void *) 0);
 if (ok<0) { printf("gettimeofday error"); }
 return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void recursive_doubling(int *input, int *output) {
   int i = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index

        if (i >= SIZE) return;

     int *source = input; 
    int *destination = output;

        for (int stride = 1; stride < SIZE; stride *= 2) {
            __syncthreads();

        if (i >= stride) {
                destination[i] = source[i] + source[i - stride]; 
        } else {
            destination[i] = source[i]; 
        }

        __syncthreads();

        int *temp = destination; 
        destination = source;
        source = temp;
    }

    if (i < SIZE) {
        output[i] = source[i]; 
    }
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
    int numSize = (SIZE + BLOCKSIZE -1)/BLOCKSIZE;
    recursive_doubling<<<numSize, BLOCKSIZE>>>(d_input, d_output);

    // Copy output data back to host
    cudaMemcpy(h_output, d_output, sizeof(int) * SIZE, cudaMemcpyDeviceToHost);

    double t0, t1;
    t0=get_clock();
    // Print the results
    for (int i = 0; i < SIZE; i++) {
      printf("%d ", h_output[i]);
    }
    printf("\n");
    t1=get_clock();
    printf("time per call: %f ns\n", (1000000000.0*(t1-t0)/SIZE) );

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}