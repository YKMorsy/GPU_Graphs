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
            if (level + 1 < d_distance[v]) 
            {
                d_distance[v] = level + 1;
                d_parent[v] = i;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                  int *queueSize, int *d_currentQueue, int *d_degrees)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < *queueSize) {
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
void block_prefix_sum(int *size, int *d_degrees, int *d_degrees_total) 
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < *size) 
    {
        __shared__ int block_data[1024]; // Assuming maximum block size of 1024 threads

        int thid_block = threadIdx.x;
        block_data[thid_block] = d_degrees[thid]; // Assign value to block_data

        __syncthreads();

        int os = 1;

        // Compute prefix sum
        for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (thid_block < d) {
                int ai = os * (2 * thid_block+1) - 1;
                int bi = os * (2 * thid_block+2) - 1;
                block_data[bi] += block_data[ai];
            }

            os *= 2;
        }

        if (thid_block == 0) { 
            // result.block_sum = block_data[blockDim.x - 1];
            block_data[blockDim.x - 1] = 0; // Clear the last element
        }

        for (int d = 1; d < blockDim.x; d *= 2) {

            os /= 2;

            __syncthreads();
            if (thid_block < d) {
                int ai = os * (2 * thid_block+1) - 1;
                int bi = os * (2 * thid_block+2) - 1;
                int t = block_data[ai];

                block_data[ai] = block_data[bi];
                block_data[bi] += t;
            }
        }

        __syncthreads();

        d_degrees[thid] = block_data[thid_block];
        d_degrees_total[thid] = block_data[blockDim.x - 1];

    }
}

__global__
void gather(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int *queueSize,
                             int *d_currentQueue, int *d_nextQueue, int *d_degrees, int *incrDegrees)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < *queueSize) {
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
                d_nextQueue[nextQueuePlace] = v;
                counter++;
            }
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
    cudaMalloc(&d_degrees_total, graph.num_nodes * sizeof(int));
    
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
    iteration = 0;
    while (*queueSize)
    {

        std::cout << iteration << " " << *queueSize << std::endl;
        
        // next layer phase
        int block_size = 1024;
        int num_blocks = *queueSize / block_size + 1;
        
        nextLayer<<<num_blocks, block_size>>>(iteration, d_col_idx, d_row_offset, d_distance, d_parent, *queueSize, d_in_q);

        countDegrees<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, queueSize, d_in_q, d_degrees);
        
        block_prefix_sum<<<num_blocks, block_size>>>(queueSize, d_degrees, d_degrees_total) ;

        // nextQueueSize = d_degrees_total[(queueSize - 1) / 1024 + 1];

        gather<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, queueSize, d_in_q, d_out_q, d_degrees, d_degrees_total);

        int index = (*queueSize - 1) / 1024 + 1; // Calculate the index in the array
        int *device_ptr = &d_degrees_total[index]; // Calculate the pointer to the correct element
        cudaMemcpy(queueSize, device_ptr, sizeof(int), cudaMemcpyDeviceToHost);

        iteration++;
        std::swap(d_in_q, d_out_q);
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    host_distance[source] = 0;

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);

    free(queueSize);

}

__host__
gpuBFS::~gpuBFS()
{
    // free(host_distance);
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
