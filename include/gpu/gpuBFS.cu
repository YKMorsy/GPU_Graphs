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
void expand_contract_kernel(csr graph, int num_nodes, int *device_queue, int *device_cur_queue_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < *device_cur_queue_size)
    {
        // get node from queue
        int cur_node = device_queue[i];

        //warp culling and history culling

        // load row range (neighbor idx) - check if cur_node is part of queue       
        int row_offset_start = graph.row_offset[cur_node];
        int row_offset_end =  graph.row_offset[cur_node+1];

        if (device_queue[cur_node] == 1)
        {
            *device_cur_queue_size = *device_cur_queue_size - 1;
        }

        // *device_cur_queue_size = *device_cur_queue_size - 1;
    }
    
}

__host__
gpuBFS::gpuBFS(csr graph, int source) : num_nodes(graph.row_offset.size()-1)
{
    // initialize queue and sizes
    init_queue();

    // initialize distance with -1 on host
    init_distance();

    // start with source (update distance and queue)
    host_distance[source] = 0;
    *host_cur_queue_size = *host_cur_queue_size + 1;

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        // copy host to device
        cudaMemcpy(device_queue, host_queue, num_nodes * sizeof(int), cudaMemcpyHostToDevice); // probably need to just copy new nodes, look into this
        cudaMemcpy(device_cur_queue_size, host_cur_queue_size, sizeof(int), cudaMemcpyHostToDevice);

        // neighbor adding kernel
        dim3 block(1024, 1);
        dim3 grid((*host_cur_queue_size+block.x-1)/block.x, 1);
        expand_contract_kernel<<<block, grid>>>(graph, num_nodes, device_queue, device_cur_queue_size);

        // copy device queue to host
        cudaMemcpy(host_queue, device_queue, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cur_queue_size, device_cur_queue_size, sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << grid.x << std::endl;
    }
}

__host__
void gpuBFS::init_queue()
{
    // allocate host memory
    host_queue = (int *)malloc(num_nodes * sizeof(int));
    host_cur_queue_size = (int *)malloc(sizeof(int));
    *host_cur_queue_size = 0;

    // allocate device memory
    cudaMalloc(&device_queue, num_nodes * sizeof(int));
    cudaMalloc(&device_cur_queue_size, sizeof(int));
}


__host__
void gpuBFS::init_distance()
{
    // allocate host memory
    host_distance = (int *)malloc(num_nodes * sizeof(int));

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