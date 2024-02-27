#include "gpuBFS.cuh"

__global__
void init_distance_kernel(int *device_distance, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        device_distance[i] = -1;
    }
}

__global__
void expand_contract_kernel(csr graph, int *device_queue, int *device_cur_queue_size)
{
    // // printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    // printf("hi %d\n", *device_cur_queue_size);

    *device_cur_queue_size = *device_cur_queue_size - 1;
}

__host__
gpuBFS::gpuBFS(csr graph, int source)
{
    // initialize member variable
    num_nodes = (graph.row_offset.size()-1);

    // initialize queue and sizes
    host_queue = (int *)malloc(num_nodes * sizeof(int));
    cudaMalloc(&device_queue, num_nodes * sizeof(int));

    host_cur_queue_size = (int *)malloc(sizeof(int));
    cudaMalloc(&device_cur_queue_size, sizeof(int));

    *host_cur_queue_size = 0;

    // initialize distance with -1 on host
    host_distance = (int *)malloc(num_nodes * sizeof(int));
    init_distance(graph);

    // start with source (update distance and queue)
    host_distance[source] = 0;
    *host_cur_queue_size = *host_cur_queue_size + 1;

    std::cout << "hi" << std::endl;

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        // copy host to device
        cudaMemcpy(device_queue, host_queue, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // probably need to just copy new nodes, look into this
        cudaMemcpy(device_cur_queue_size, host_cur_queue_size, sizeof(int), cudaMemcpyHostToDevice);

        // std::cout << "hi" << std::endl;

        // neighbor adding kernel
        dim3 block(1024, 1);
        dim3 grid((*host_cur_queue_size+block.x-1)/block.x, 1);
        expand_contract_kernel<<<block, grid>>>(graph, device_queue, device_cur_queue_size);
        cudaGetLastError();
        cudaDeviceSynchronize();

        // copy device queue to host
        cudaMemcpy(host_queue, device_queue, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cur_queue_size, device_cur_queue_size, sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << *host_cur_queue_size << std::endl;

        // temp
        // *host_cur_queue_size--;
    }
}


__host__
void gpuBFS::init_distance(csr graph)
{
    // allocate device memory
    cudaMalloc(&device_distance, num_nodes * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(device_distance, host_distance, num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel to inialize kernel
    dim3 block(1024, 1);
    dim3 grid((num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(device_distance, num_nodes);

    // copy back
    cudaMemcpy(host_distance, device_distance, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

__host__
gpuBFS::~gpuBFS()
{
    free(host_distance);
    free(host_queue);
    free(host_cur_queue_size);

    cudaFree(device_distance);
    cudaFree(device_queue);
    cudaFree(device_cur_queue_size);
}