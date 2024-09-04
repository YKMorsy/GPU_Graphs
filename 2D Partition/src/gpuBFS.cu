#include "../include/gpuBFS.cuh"
#include "../include/prefixSum.cuh"
#include <stdio.h>

__global__ 
void cumulative_degree(const int frontier_count, const int* d_col_offset, const int* d_row_index, const int* d_frontier, int* d_degrees, int* d_total_degrees)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < frontier_count)
    {
        int cur_node = d_frontier[global_tid];
        int num_edges = d_col_offset[cur_node + 1] - d_col_offset[cur_node];

        d_degrees[global_tid] = num_edges;
        atomicAdd(d_total_degrees, num_edges);
    }
}

__device__ 
int binsearch_maxle(const int *d_cumulative_degrees, const int frontier_count, const int global_tid) {
    int left = 0;
    int right = frontier_count - 1;
    int mid;

    while (left <= right) {
        mid = (left + right) / 2;

        if (d_cumulative_degrees[mid] <= global_tid) {
            left = mid + 1;  // Move right to find a larger valid index
        } else {
            right = mid - 1; // Move left to find a smaller valid index
        }
    }

    // The desired index is right because it will be the largest index such that cumul[right] <= gid
    return right;
}

__global__
void expand_frontier(const int* d_frontier, const int frontier_count, const int iteration, const int* d_cumulative_degrees, const int* d_col_offset, 
                     const int* d_row_index, int* d_bmap, int* d_distance, int* d_pred, const int total_degrees, int* d_dest_verts, int* d_frontier_count)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < total_degrees)
    {
        // find greatest index k such that cumul[k] is less than or equal to tid
        int k = binsearch_maxle(d_cumulative_degrees, frontier_count, global_tid);

        // get frontier node u using k
        int u = d_frontier[k];

        // get neighbor v using row[col[u] + tid - cumul[k]]
        int v = d_row_index[d_col_offset[u] + global_tid - d_cumulative_degrees[k]];

        if (d_bmap[v] == 1) return;

        d_bmap[v] = 1;
        d_pred[v] = u;
        d_distance[v] = iteration;
        uint32_t off = atomicAdd(d_frontier_count, 1);
        d_dest_verts[off] = v;
        
    }
}

__global__
void update_frontier(int* d_frontier, const int* d_dest_verts, int* d_frontier_count)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < *d_frontier_count)
    {
        d_frontier[global_tid] = d_dest_verts[global_tid];
    }
}

gpuBFS::gpuBFS(csc &graph, int source, int R, int C)
{
    /* NVSHMEM INIT */
    nvshmem_init();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    CUDA_CHECK(cudaSetDevice(mype));

    total_nodes = graph.num_nodes;
    pe_nodes = graph.csc_vect[mype].first.size()-1;

    pe_rows = R;
    pe_cols = C;

    init_device(graph, mype);
    init_source(mype, source);

    /* CUDA EVENTS FOR TIMING */
    cudaEvent_t gpu_start, gpu_end;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_end));
    CUDA_CHECK(cudaEventRecord(gpu_start));

    int frontier_count = 1;
    int iteration = 1;
    int total_degrees = 0;

    PrefixSum ps;

    while (true)
    {
        
        CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_total_degrees, 0, sizeof(int)));

        
        /* expand */
        // communicate frontier (share frontier with processors in the same processor column (processors Pij with the same j))
        // get sizes

        // // create all frontier according to new size
        // int *d_all_frontiers = (int *)nvshmem_malloc(col_offset.size() * sizeof(int));

        // calculate degrees then expand frontier
        cumulative_degree<<< (frontier_count + 1024 - 1) / 1024, 1024 >>>(frontier_count, d_col_offset, d_row_index, d_frontier, d_degrees, d_total_degrees);
        ps.sum_scan_blelloch(d_cumulative_degrees, d_degrees, frontier_count);
        cudaMemcpy(&total_degrees, d_total_degrees, sizeof(int), cudaMemcpyDeviceToHost);
        expand_frontier<<< (total_degrees + 1024 - 1) / 1024, 1024 >>>(d_frontier, frontier_count, iteration, d_cumulative_degrees, d_col_offset, d_row_index, d_bmap, d_distance, d_pred, total_degrees, d_dest_verts, d_frontier_count);

        cudaMemcpy(&frontier_count, d_frontier_count, sizeof(int), cudaMemcpyDeviceToHost);

        /* fold */
        // update frontier
        update_frontier<<< (frontier_count + 1024 - 1) / 1024, 1024 >>>(d_frontier, d_dest_verts, d_frontier_count);
        
        if (frontier_count == 0) break;

        // CUDA_CHECK(cudaDeviceSynchronize());
        iteration++;
    }

    /* CUDA EVENTS FOR TIMING */
    CUDA_CHECK(cudaEventRecord(gpu_end));
    CUDA_CHECK(cudaEventSynchronize(gpu_end));
    CUDA_CHECK(cudaEventElapsedTime(&exec_time, gpu_start, gpu_end));

    // transfer distance to host
    CUDA_CHECK(cudaMemcpy(host_distance, d_distance, pe_nodes * sizeof(int), cudaMemcpyDeviceToHost));


}

void gpuBFS::init_source(int mype, int source)
{
    // check which pe to put source in
    int num_pes = nvshmem_n_pes();
    int p_i = mype / pe_cols;
    int p_j = mype % pe_cols;
    int p_block = p_j * pe_rows + p_i;
    int source_block;

    if (source < (total_nodes%num_pes)*(total_nodes/num_pes + 1))
    {
        source_block = source/(total_nodes/num_pes + 1);
    }
    else
    {
        source_block = ((source - (total_nodes%num_pes)*(total_nodes/num_pes + 1)) / total_nodes/num_pes) + (total_nodes%num_pes);
    }

    if (source_block == p_block)
    {
        int value1 = 1;
        int value2 = 0;

        CUDA_CHECK(cudaMemcpy(d_frontier + 0, &source, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_frontier + 1, &value1, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bmap + source, &value1, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_distance + source, &value2, sizeof(int), cudaMemcpyHostToDevice));
    }
}


void gpuBFS::init_device(csc &graph, int mype)
{
    // get csc arrays
    std::vector<int> col_offset = graph.csc_vect[mype].first;
    std::vector<int> row_index = graph.csc_vect[mype].second;

    host_distance = (int *)malloc(pe_nodes * sizeof(int));

    // initialize device memory
    d_col_offset = (int *)nvshmem_malloc(col_offset.size() * sizeof(int));
    d_row_index = (int *)nvshmem_malloc(row_index.size() * sizeof(int));
    d_frontier = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_pred = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_bmap = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_distance = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_degrees = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_total_degrees = (int *)nvshmem_malloc(sizeof(int));
    d_cumulative_degrees = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_frontier_count = (int *)nvshmem_malloc(sizeof(int));
    d_dest_verts = (int *)nvshmem_malloc(pe_nodes * sizeof(int));

    CUDA_CHECK(cudaMemcpy(d_col_offset, col_offset.data(), col_offset.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_index, row_index.data(), row_index.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_frontier, 0, pe_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pred, -1, pe_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bmap, 0, pe_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_distance, -1, pe_nodes * sizeof(int)));
    
}


void gpuBFS::print_distance(csc &graph)
{
    std::cout << "\n------GPU DISTANCE VECTOR------" << std::endl;

    for (int i = 0; i < pe_nodes; i++) 
    {
        std::cout << host_distance[i] << " | ";
    }
    std::cout << std::endl;
}

gpuBFS::~gpuBFS() 
{
    nvshmem_free(d_col_offset);
    nvshmem_free(d_row_index);
    nvshmem_free(d_frontier);
    nvshmem_free(d_pred);
    nvshmem_free(d_bmap);
    nvshmem_free(d_distance);
    nvshmem_free(d_degrees);
    nvshmem_free(d_total_degrees);
    nvshmem_free(d_cumulative_degrees);
    nvshmem_free(d_frontier_count);
    nvshmem_free(d_dest_verts);
    nvshmem_finalize();
}