#include "gpuBFS.cuh"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "'" << std::endl;
        cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
        exit(99);
    }
}

__global__
void nextLayer(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                int queueSize, int *d_currentQueue, int *d_distance, int iteration) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];

        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == -1) 
            {
                d_parent[v] = i;
                d_distance[v] = iteration + 1;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                  int queueSize, int *d_currentQueue, int *d_degrees, int *d_distance, int iteration)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];

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
void gather(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
            int *d_currentQueue, int *d_nextQueue, int *incrDegrees, int *d_distance, int iteration)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int nextQueuePlace = incrDegrees[thid];

        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];
        for (int i = row_offset_start; i < row_offset_end; i++)
        {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != cur_node) 
            {
                d_nextQueue[nextQueuePlace] = v;
                nextQueuePlace++;
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
            device_parent[i] = -1;
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
    init_device(graph, source);

    PrefixSum ps;

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

    int block = 1024;
    while (*queueSize)
    {
        std::cout << "iter and size " << iteration << " " << *queueSize << std::endl;

        int grid = (*queueSize+block-1)/block;

        nextLayer<<<grid, block>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_distance, iteration);
        // cudaDeviceSynchronize();

        countDegrees<<<grid, block>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_degrees, d_distance, iteration);
        // cudaDeviceSynchronize();

        ps.sum_scan_blelloch(d_degrees_total, d_degrees, *queueSize+1);
        // cudaDeviceSynchronize();

        cudaMemcpy(nextQueueSize, &d_degrees_total[*queueSize], sizeof(int), cudaMemcpyDeviceToHost);

        gather<<<grid, block>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_out_q, d_degrees_total, d_distance, iteration);
        // cudaDeviceSynchronize();

        iteration++;
        *queueSize = *nextQueueSize;
        int *temp = d_in_q;
        d_in_q = d_out_q;
        d_out_q = temp;
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);

    free(queueSize);
    free(nextQueueSize);
}

__host__
void gpuBFS::init_device(csr &graph, int source)
{
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    checkCudaErrors(cudaMalloc(&d_distance, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_col_idx, graph.num_edges * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_distance, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_parent, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_in_q, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_out_q, graph.num_nodes * sizeof(int)));

    checkCudaErrors(cudaMalloc(&d_degrees, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_degrees_total, graph.num_nodes * sizeof(int)));

    dim3 block(1024, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(d_distance, d_parent, graph.num_nodes, source);
    cudaDeviceSynchronize();

    int firstElementQueue = source;
    cudaMemcpy(d_in_q, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
}

__host__
gpuBFS::~gpuBFS()
{
    free(host_distance);    

    cudaFree(d_distance);
    cudaFree(d_in_q);
    cudaFree(d_out_q);
    cudaFree(d_parent);
    cudaFree(d_degrees);
    cudaFree(d_col_idx);
    cudaFree(d_row_offset);
    cudaFree(d_degrees_total);
}
