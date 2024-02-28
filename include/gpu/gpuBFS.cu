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
void expand_contract_kernel(int *device_col_idx, int *device_row_offset, int num_nodes, int *device_in_queue, int *device_cur_queue_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < *device_cur_queue_size)
    {
        // get node from queue
        int cur_node = device_in_queue[i];

        //warp culling and history culling

        printf("%d\n", cur_node);
        printf("%d\n", *device_cur_queue_size);


        // load row range (neighbor idx) - check if cur_node is part of queue       
        int row_offset_start = device_row_offset[cur_node];
        int row_offset_end =  device_row_offset[cur_node+1];

        // only create outqueue if current node is not isolated
        if (row_offset_start != -1)
        {
            // do stuff
            printf("%d and %d\n", row_offset_start, row_offset_end);
        }

        // temp            
        *device_cur_queue_size = *device_cur_queue_size - 1;
        printf("%d\n", *device_cur_queue_size);
    }
    
}

__host__
gpuBFS::gpuBFS(csr &graph, int source)
{
    // initialize queue and sizes
    init_queue(graph);

    // initialize distance with -1 on host
    init_distance(graph);

    // initialize device graph variables
    init_graph_for_device(graph);

    // start with source (update distance and queue)
    host_distance[source] = 0;
    *host_cur_queue_size = *host_cur_queue_size + 1;

    host_queue[0] = source;

    // printf("%d\n", host_queue[0]);

    // int row_offset_start = graph.row_offset[source];
    // int row_offset_end =  graph.row_offset[source+1];

    // printf("%d and %d\n", row_offset_start, row_offset_end);

    // printf("%d\n", host_distance[0]);
    // printf("%d\n", host_queue[0]);

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        // copy host to device
        cudaMemcpy(device_in_queue, host_queue, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice); // probably need to just copy new nodes, look into this
        cudaMemcpy(device_cur_queue_size, host_cur_queue_size, sizeof(int), cudaMemcpyHostToDevice);

        // int row_offset_start = graph.row_offset[source];
        // int row_offset_end =  graph.row_offset[source+1];

        // printf("%d and %d\n", row_offset_start, row_offset_end);

        // neighbor adding kernel
        dim3 block(1024, 1);
        dim3 grid((*host_cur_queue_size+block.x-1)/block.x, 1);
        expand_contract_kernel<<<block, grid>>>(device_col_idx, device_row_offset, graph.num_nodes, device_in_queue, device_cur_queue_size);
        cudaDeviceSynchronize();

        // copy device queue to host
        cudaMemcpy(host_queue, device_in_queue, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cur_queue_size, device_cur_queue_size, sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << grid.x << std::endl;
        // *host_cur_queue_size = *host_cur_queue_size - 1;
    }
}

__host__
void gpuBFS::init_graph_for_device(csr &graph)
{
    cudaMalloc(&device_col_idx, graph.num_edges * sizeof(int));
    cudaMalloc(&device_row_offset, (graph.num_nodes+1) * sizeof(int));

    cudaMemcpy(device_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
}

__host__
void gpuBFS::init_queue(csr &graph)
{
    // allocate host memory
    host_queue = (int *)malloc(graph.num_nodes * sizeof(int));
    host_cur_queue_size = (int *)malloc(sizeof(int));
    *host_cur_queue_size = 0;

    // allocate device memory
    cudaMalloc(&device_in_queue, graph.num_nodes * sizeof(int));
    cudaMalloc(&device_cur_queue_size, sizeof(int));
}


__host__
void gpuBFS::init_distance(csr &graph)
{
    // allocate host memory
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    // allocate device memory
    cudaMalloc(&device_distance, graph.num_nodes * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(device_distance, host_distance, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel to inialize kernel
    dim3 block(1024, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(device_distance, graph.num_nodes);

    // copy back
    cudaMemcpy(host_distance, device_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

__host__
gpuBFS::~gpuBFS()
{
    free(host_distance);
    free(host_queue);
    free(host_cur_queue_size);

    cudaFree(device_distance);
    cudaFree(device_in_queue);
    cudaFree(device_cur_queue_size);
}