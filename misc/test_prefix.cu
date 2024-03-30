#include <stdio.h>
#include <iostream>

const int BLOCK_SIZE = 16;
#define ARRAY_SIZE (BLOCK_SIZE * 1) // Total number of elements

struct prescan_result {
    int prefix_sum;
    int block_sum;
};

int div_up(int dividend, int divisor)
{
	return (dividend % divisor == 0)?(dividend/divisor):(dividend/divisor+1);
}

__device__ prescan_result block_prefix_sum(const int val) {
    __shared__ int block_data[BLOCK_SIZE]; // Assuming maximum block size of 1024 threads
    prescan_result result;

    int thid = threadIdx.x;
    block_data[thid] = val; // Assign value to block_data
    

    __syncthreads();

    int offset = 1;

    // Compute prefix sum
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid+1) - 1;
            int bi = offset * (2 * thid+2) - 1;
            block_data[bi] += block_data[ai];
        }

        offset *= 2;
    }

    if (thid == 0) { 
        // result.block_sum = block_data[blockDim.x - 1];
        block_data[blockDim.x - 1] = 0; // Clear the last element
    }

    for (int d = 1; d < blockDim.x; d *= 2) {

        offset /= 2;

        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid+1) - 1;
            int bi = offset * (2 * thid+2) - 1;
            int t = block_data[ai];

            block_data[ai] = block_data[bi];
            block_data[bi] += t;
        }
    }

    __syncthreads();
    
    result.prefix_sum = block_data[thid];
    // if (thid == blockDim.x - 1) {
    result.block_sum = block_data[blockDim.x - 1];
    // }

    // if (thid == 0) 
    // {
    //     printf("%d \n", block_data[thid]);
    // }

    return result;
}

__global__ void test_block_prefix_sum(int *input, int *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ARRAY_SIZE) {
        int val = input[idx];
        prescan_result result = block_prefix_sum(val);
        printf("%d \n", result.block_sum);
        output[idx] = result.prefix_sum;
    }
}

int main() {
    int *d_input, *d_output;
    int input[ARRAY_SIZE];

    // Initialize input array
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        input[i] = i; // Each thread will contribute 1
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_input, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void **)&d_output, ARRAY_SIZE * sizeof(int));

    // Copy input array to device
    cudaMemcpy(d_input, input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions

    // Launch kernel
    test_block_prefix_sum<<<1, BLOCK_SIZE>>>(d_input, d_output);

    // Copy result back to host
    int output[ARRAY_SIZE];
    // for (int i = 0; i < ARRAY_SIZE; ++i) {
    //     output[i] = 0;
    // }
    cudaMemcpy(output, d_output, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Prefix sum result:\n");
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        printf("%d ", output[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
