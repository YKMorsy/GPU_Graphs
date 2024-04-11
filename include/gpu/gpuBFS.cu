#include "gpuBFS.cuh"

__global__
void nextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_distance, int *d_parent,
               int queueSize, int *d_currentQueue) 
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) 
    {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = cur_node < 0 ? 0 : d_edgesOffset[cur_node];
        int row_offset_end = cur_node < 0 ? 0 : d_edgesOffset[cur_node+1];
        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == -1) 
            {
                // printf("neighbor %d\n", v);
                d_distance[v] = level + 1;
                d_parent[v] = i;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                  int queueSize, int *d_currentQueue, int *d_degrees)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = cur_node < 0 ? 0 : d_edgesOffset[cur_node];
        int row_offset_end = cur_node < 0 ? 0 : d_edgesOffset[cur_node+1];

        int degree = 0;

        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != cur_node) 
            {
                degree++;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void block_prefix_sum(int size, int *d_degrees, int *incrDegrees) 
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < size) {
        //write initial values to shared memory
        __shared__ int prefixSum[1024];
        int modulo = threadIdx.x;
        prefixSum[modulo] = d_degrees[thid];
        __syncthreads();

        //calculate scan on this block
        //go up
        for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int nextPosition = modulo + (nodeSize >> 1);
                    prefixSum[modulo] += prefixSum[nextPosition];
                }
            }
            __syncthreads();
        }

        //write information for increment prefix sums
        if (modulo == 0) {
            int block = thid >> 10;
            incrDegrees[block + 1] = prefixSum[modulo];
            // printf("total %d %d\n", block + 1, incrDegrees[block + 1]);
        }

        //go down
        for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
            if ((modulo & (nodeSize - 1)) == 0) {
                if (thid + (nodeSize >> 1) < size) {
                    int next_position = modulo + (nodeSize >> 1);
                    int tmp = prefixSum[modulo];
                    prefixSum[modulo] -= prefixSum[next_position];
                    prefixSum[next_position] = tmp;

                }
            }
            __syncthreads();
        }
        d_degrees[thid] = prefixSum[modulo];

        // printf("individiual %d %d\n", thid, d_degrees[thid]);
    }

}
__global__
void gather(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
                             int *d_currentQueue, int *d_nextQueue, int *d_degrees, int *incrDegrees)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        __shared__ int sharedIncrement;
        if (!threadIdx.x) {
            sharedIncrement = incrDegrees[thid >> 10];
        }
        __syncthreads();

        int sum = 0;
        if (threadIdx.x) {
            sum = d_degrees[thid - 1];
        }

        int cur_node = d_currentQueue[thid];
        int row_offset_start = cur_node < 0 ? 0 : d_edgesOffset[cur_node];
        int row_offset_end = cur_node < 0 ? 0 : d_edgesOffset[cur_node+1];
        int counter = 0;
        for (int i = row_offset_start; i < row_offset_end; i++)
        {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != cur_node) {
                int nextQueuePlace = sharedIncrement + sum + counter;
                // printf("individiual %d %d\n", thid, nextQueuePlace);
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
        }
    }
}

__global__
void init_distance_kernel(int *device_distance, int *device_parent, int size, int source)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        if (i == source)
        {
            device_distance[i] = 0;
            device_parent[i] = 0;
        }
        else
        {
            device_distance[i] = -1;
            device_parent[i] = -1;
        }

    }
}

__host__
gpuBFS::gpuBFS(csr &graph, int source)
{
    cudaMalloc(&d_col_idx, graph.num_edges * sizeof(int));
    cudaMalloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int));
    // cudaMalloc(&d_edges_size, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_distance, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_parent, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_in_q, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_out_q, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_degrees, graph.num_nodes * sizeof(int));

    cudaMallocHost((void **) &d_degrees_total, graph.num_nodes * sizeof(int));

    init_distance(graph, source);

    int firstElementQueue = source;
    cudaMemcpy(d_in_q, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
    
    // d_degrees_total = (int *)malloc(graph.num_nodes * sizeof(int));

    cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    int *queueSize;
    queueSize = (int *)malloc(sizeof(int));
    *queueSize = 1;
    int *nextQueueSize;
    nextQueueSize = (int *)malloc(sizeof(int));
    *nextQueueSize = 0;
    iteration = 0;
    while (*queueSize)
    {

        // std::cout << "iter and size: " << iteration << " " << *queueSize << std::endl;
        
        // next layer phase
        int block_size = 1024;
        int num_blocks = *queueSize / block_size + 1;

        // std::cout << num_blocks << std::endl;
        
        nextLayer<<<num_blocks, block_size>>>(iteration, d_col_idx, d_row_offset, d_distance, d_parent, *queueSize, d_in_q);
        cudaDeviceSynchronize();

        // cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << host_distance[1] << " " << host_distance[2] << std::endl;

        countDegrees<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_degrees);
        cudaDeviceSynchronize();
        
        block_prefix_sum<<<num_blocks, block_size>>>(*queueSize, d_degrees, d_degrees_total) ;
        cudaDeviceSynchronize();

        *nextQueueSize = d_degrees_total[(*queueSize - 1) / 1024 + 1];

        // std::cout << *nextQueueSize << std::endl;

        gather<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_out_q, d_degrees, d_degrees_total);
        cudaDeviceSynchronize();

        iteration++;
        *queueSize = *nextQueueSize;
        std::swap(d_in_q, d_out_q);

        cudaDeviceSynchronize();
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << host_distance[1] << " " << host_distance[2] << std::endl;
    // host_distance[source] = 0;

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);

    free(queueSize);
    free(nextQueueSize);

}

__host__
void gpuBFS::init_distance(csr &graph, int source)
{
    // allocate host memory
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    // allocate device memory
    cudaMalloc(&d_distance, graph.num_nodes * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_distance, host_distance, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel to inialize distance
    dim3 block(1024, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(d_distance, d_parent, graph.num_nodes, source);

    // copy back
    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}


__host__
gpuBFS::~gpuBFS()
{
    free(host_distance);
    // free(host_queue);
    

    cudaFree(d_distance);
    cudaFree(d_in_q);
    cudaFree(d_out_q);
    cudaFree(d_parent);
    cudaFree(d_degrees);
    cudaFree(d_col_idx);
    cudaFree(d_row_offset);
    cudaFree(d_degrees_total);
}
