#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda_runtime.h>
#include <cstdio>
#include "gpuBFS.cuh"

// Error checking for CUDA calls
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "'" << std::endl;
        cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
        exit(99);
    }
}

// BFS kernels
__global__
void nextLayer(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
                int *d_currentQueue, int *d_distance, int iteration, int *d_edges_traversed,
                int node_start, int node_end) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        if (cur_node < node_start || cur_node >= node_end) return;

        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];
        int local_edges_traversed = 0;
        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == -1) {
                d_parent[v] = i;
                d_distance[v] = iteration + 1;
            }
            local_edges_traversed++;
        }
        atomicAdd(d_edges_traversed, local_edges_traversed);
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
                  int *d_currentQueue, int *d_degrees, int *d_distance, int iteration,
                  int node_start, int node_end) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        if (cur_node < node_start || cur_node >= node_end) return;

        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];
        int degree = 0;
        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != cur_node) {
                degree++;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void gather(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
            int *d_currentQueue, int *d_nextQueue, int *incrDegrees, int *d_distance, int iteration,
            int node_start, int node_end) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;
    if (thid < queueSize) {
        int nextQueuePlace = incrDegrees[thid];
        int cur_node = d_currentQueue[thid];
        if (cur_node < node_start || cur_node >= node_end) return;

        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];
        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != cur_node) {
                d_nextQueue[nextQueuePlace] = v;
                nextQueuePlace++;
            }
        }
    }
}

__global__
void init_distance_kernel(int *device_distance, int *device_parent, int size, int source) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (i == source) {
            device_distance[i] = 0;
            device_parent[i] = -1;
        } else {
            device_distance[i] = -1;
            device_parent[i] = -1;
        }
    }
}


// Constructor to initialize BFS on GPU
gpuBFS::gpuBFS(csr &graph, int source) {
    int mype_node;
    int mype;
    int npes;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    cudaSetDevice(mype_node);

    // Partition the graph among GPUs
    int nodes_per_pe = (graph.num_nodes + npes - 1) / npes;
    int node_start = mype * nodes_per_pe;
    int node_end = min(node_start + nodes_per_pe, graph.num_nodes);

    init_device(graph, source, mype, npes, node_start, node_end);

    PrefixSum ps;

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    // Size depends on device
    if (source >= node_start && source < node_end)
    {
        int *queueSize;
        queueSize = (int *)malloc(sizeof(int));
        *queueSize = 1;
    }

    int *nextQueueSize;
    nextQueueSize = (int *)malloc(sizeof(int));
    *nextQueueSize = 0;
    
    iteration = 0;

    int block = 1024;
    total_edges_traversed = 0; // Initialize total edges traversed
    int *d_edges_traversed;
    nvshmem_malloc(&d_edges_traversed, sizeof(int));
    cudaMemset(d_edges_traversed, 0, sizeof(int));

    while (*queueSize) {
        int grid = (*queueSize+block-1)/block;

        nextLayer<<<grid, block>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_distance, iteration, d_edges_traversed, node_start, node_end);
        nvshmemx_barrier_all();

        countDegrees<<<grid, block>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_degrees, d_distance, iteration, node_start, node_end);
        nvshmemx_barrier_all();

        ps.sum_scan_blelloch(d_degrees_total, d_degrees, *queueSize+1);
        nvshmemx_barrier_all();

        cudaMemcpy(nextQueueSize, &d_degrees_total[*queueSize], sizeof(int), cudaMemcpyDeviceToHost);

        gather<<<grid, block>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_out_q, d_degrees_total, d_distance, iteration, node_start, node_end);
        nvshmemx_barrier_all();

        iteration++;

        for(int j = 0; j < npes; j++)
        {
            nvshmem_int_put(d_in_q, d_out_q, *nextQueueSize, j)
        }
        

        // TODO 
        /* SOMETHING SIMILAR TO THIS BUT UTILIZING NVSHMEM PUT
        for(int i = 0; i < DeviceNum; ++i)
        {
            int offset = 0;
            for(int j = 0; j < DeviceNum; ++j)
            {
                // cudaMemcpy(d_currentQueue2[i] + offset, &(d_nextQueue2[j][i * G.numVertices]), 
                //     nextQueueSize[i][j] * sizeof(int), 
                //     cudaMemcpyDefault);
                // printf("copying %d\n", nextQueueSize[i][j]);
                cudaMemcpyPeer(d_currentQueue2[i] + offset, i, &(d_nextQueue2[j][i * G.numVertices]), j, 
                    nextQueueSize[j][i] * sizeof(int));
                offset += nextQueueSize[j][i];
                // offset += nextQueueSize[i][j];
            }
        }
        */



        *queueSize = *nextQueueSize;
        int *temp = d_in_q;
        d_in_q = d_out_q;
        d_out_q = temp;

        nvshmemx_barrier_all();
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_edges_traversed, d_edges_traversed, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);

    nvshmem_free(d_edges_traversed);
    free(queueSize);
    free(nextQueueSize);

    nvshmem_finalize();
}

// Initialize device and allocate memory
void gpuBFS::init_device(csr &graph, int source, int mype, int npes, int node_start, int node_end) {
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    checkCudaErrors(nvshmem_malloc(&d_distance, graph.num_nodes * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_col_idx, graph.num_edges * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_distance, graph.num_nodes * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_parent, graph.num_nodes * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_in_q, graph.num_nodes * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_out_q, graph.num_nodes * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_degrees, graph.num_nodes * sizeof(int)));
    checkCudaErrors(nvshmem_malloc(&d_degrees_total, graph.num_nodes * sizeof(int)));

    dim3 block(1024, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(d_distance, d_parent, graph.num_nodes, source);
    cudaDeviceSynchronize();

    // Source placed in in queue depending on device
    if (source >= node_start && source < node_end)
    {
        int firstElementQueue = source;
        cudaMemcpy(d_in_q, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);
    }

    checkCudaErrors(cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
}

// Destructor to free memory
gpuBFS::~gpuBFS() {
    free(host_distance);    

    nvshmem_free(d_distance);
    nvshmem_free(d_in_q);
    nvshmem_free(d_out_q);
    nvshmem_free(d_parent);
    nvshmem_free(d_degrees);
    nvshmem_free(d_col_idx);
    nvshmem_free(d_row_offset);
    nvshmem_free(d_degrees_total);
}
