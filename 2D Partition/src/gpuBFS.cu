#include "../include/gpuBFS.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__
void update_frontier(int* d_frontier, const int* d_dest_verts, int* d_frontier_count, const int* d_dest_verts_count, int* d_bmap, const int start_node, const int iteration, int* d_distance, int* d_te)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < *d_dest_verts_count)
    {
        // check if node is visisted and update frontier, distance, bmap, and pred if not visited

        // if (d_dest_verts[global_tid] == 2)
        // {
        //     printf("%d foreign node %d\n", nvshmem_my_pe(), d_dest_verts[global_tid]);
        // }

        if (d_distance[d_dest_verts[global_tid]-start_node] == -1)
        {
            d_distance[d_dest_verts[global_tid]-start_node] = iteration;
            // atomicAdd(d_te, 1);
        } 
        d_bmap[d_dest_verts[global_tid]] = 1;
        d_frontier[global_tid] = d_dest_verts[global_tid];

        // if (d_dest_verts[global_tid] == 2)
        // {
        //     printf("%d foreign node %d\n", nvshmem_my_pe(), d_frontier[global_tid]);
        // }

        atomicAdd(d_frontier_count, 1);
    }
}

__device__ 
int binsearch_maxle(const int *d_cumulative_degrees, const int frontier_count, const int global_tid) 
{
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
void print_d_dest_verts_count(int* d_dest_verts_count)
{
    // printf("pe: %d target pe: %d d_dest_verts_count check: %d\n", nvshmem_my_pe(), nvshmem_my_pe(), *d_dest_verts_count);
}

__global__
void expand_frontier(const int* d_frontier, const int frontier_count, const int iteration, const int* d_cumulative_degrees, const int* d_col_offset, 
                     const int* d_row_index, int* d_bmap, int* d_distance, int* d_pred, const int total_degrees, int* d_dest_verts, int* d_dest_verts_count, int* d_frontier_count,
                     const int* d_start_col_node, const int total_nodes, const int pe_rows, const int pe_cols, const int mype, const int start_node, int* d_te)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < total_degrees)
    {
        // find greatest index k such that cumul[k] is less than or equal to tid
        int k = binsearch_maxle(d_cumulative_degrees, frontier_count, global_tid);

        // get frontier node u using k
        int u = d_frontier[k] - *d_start_col_node;

        // printf("%d d_row_index_index %d\n", nvshmem_my_pe(), d_col_offset[u] + global_tid - d_cumulative_degrees[k]);

        // get neighbor v using row[col[u] + tid - cumul[k]]
        int v = d_row_index[d_col_offset[u] + global_tid - d_cumulative_degrees[k]];

        // printf("%d HELLO %d %d\n", nvshmem_my_pe(), u, v);

        // if (d_frontier[k] == 513)
        // {
        //     printf("%d HELLO %d %d\n", nvshmem_my_pe(), u, v);
        // }

        if (d_bmap[v] == 1) return; // bmap holds all nodes no matter the pe

        d_bmap[v] = 1;

        // check which pe owns v and put in correct pe
        int p_i = mype / pe_cols;
        int p_j = mype % pe_cols;
        // int pe_block = p_j*pe_rows + p_i;

        int target_j = v/(total_nodes/(pe_cols));

        if (p_j == target_j)
        {
            // if (d_frontier[k] == 513)
            // {
            //     printf("%d HELLO %d %d\n", nvshmem_my_pe(), u, v);
            // }
            // get local value of v
            d_distance[v-start_node] = iteration;
            uint32_t off = atomicAdd(d_dest_verts_count, 1);
            d_dest_verts[off] = v;
        }
        else
        {
            int target_pe = p_i*pe_cols + target_j;
            // printf("%d foreign node %d %d\n", nvshmem_my_pe(), target_pe, v);
            int off = nvshmem_int_atomic_fetch_add(d_dest_verts_count, 1, target_pe);
            nvshmem_int_put(d_dest_verts + off, &v, 1, target_pe);

            // if (d_distance[d_dest_verts[global_tid]-start_node] == -1)
            // {
            //     d_distance[d_dest_verts[global_tid]-start_node] = iteration;
            // } 
            // d_bmap[d_dest_verts[global_tid]] = 1;
            // d_frontier[global_tid] = d_dest_verts[global_tid];
        }



        // if (d_frontier[k] == 513)
        // {
        //     printf("%d HELLO %d %d\n", nvshmem_my_pe(), u, v);
        // }
    }
}


__global__ 
void cumulative_degree(const int all_frontier_count, const int* d_col_offset, const int* d_row_index, const int* d_all_frontier, int* d_degrees, int* d_total_degrees, int* d_start_col_node)
{
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_tid < all_frontier_count)
    {
        // convert cur_node to correct local node
        int cur_node = d_all_frontier[global_tid] - *d_start_col_node;
        int num_edges = d_col_offset[cur_node + 1] - d_col_offset[cur_node];

        // if (d_all_frontier[global_tid] == 513)
        // {
        //     printf("%d cur_node %d\n", nvshmem_my_pe(), cur_node);
        //     printf("%d num_edges %d\n", nvshmem_my_pe(), num_edges);
        // }

        // printf("%d cur_node %d\n", nvshmem_my_pe(), cur_node);
        // printf("%d num_edges %d\n", nvshmem_my_pe(), num_edges);

        d_degrees[global_tid] = num_edges;
        atomicAdd(d_total_degrees, num_edges);

        // printf("%d d_total_degrees %d\n", nvshmem_my_pe(), *d_total_degrees);
    }
}

__global__
void comm_frontier(const int* d_frontier_count, const int* d_frontier, int* d_all_frontier_count, int* d_all_frontier, int pe_rows, int pe_cols, const int mype)
{
    // pe gets frontiers of all processors in the same column

    // printf("%d d_frontier_count %d\n", mype, *d_frontier_count);
    
    for (int i = 0; i < pe_rows; i++)
    {
        int cur_pe_j = mype % pe_cols;
        int target_pe = i*pe_cols+cur_pe_j;
        int target_frontier_count = nvshmem_int_g(d_frontier_count, target_pe);
        nvshmem_int_get(d_all_frontier + *d_all_frontier_count, d_frontier, target_frontier_count, target_pe);
        *d_all_frontier_count += target_frontier_count;
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

    pe_rows = R;
    pe_cols = C;

    pe_nodes = total_nodes/(R*C);

    init_device(graph, mype);
    init_source(mype, source);

    /* CUDA EVENTS FOR TIMING */
    cudaEvent_t gpu_start, gpu_end;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_end));
    CUDA_CHECK(cudaEventRecord(gpu_start));

    int global_frontier_count = 1;
    // int frontier_count = 1;
    int dest_vert_count = 1;
    int iteration = 1;
    int total_degrees = 0;
    int all_frontier_count = 1;

    PrefixSum ps;

    nvshmem_barrier_all();

    int p_i = mype / pe_cols;
    int p_j = mype % pe_cols;
    int vertex_block = p_j*pe_rows + p_i;
    start_node = vertex_block*(total_nodes/(pe_rows*pe_cols));
    // start_node = (total_nodes/pe_cols) * (mype % pe_cols);
    // std::cout << mype << " start node " << start_node << std::endl;

    while (true)
    {
        nvshmem_barrier_all();
        dest_vert_count = 0;
        CUDA_CHECK(cudaMemset(d_total_degrees, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_all_frontier_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_dest_verts_count, 0, sizeof(int)));
        nvshmem_barrier_all();
        
        /* expand */

        // communicate frontier (share frontier with processors in the same processor column (processors Pij with the same j))
        // in: d_frontier, d_frontier_count, pe_cols
        // out: d_all_frontier, d_all_frontier_count
        // nvshmem_barrier_all();
        comm_frontier<<< 1, 1 >>>(d_frontier_count, d_frontier, d_all_frontier_count, d_all_frontier, pe_rows, pe_cols, mype);
        nvshmem_barrier_all();

        cudaMemcpy(&all_frontier_count, d_all_frontier_count, sizeof(int), cudaMemcpyDeviceToHost);
        nvshmem_barrier_all();

        // calculate degrees then expand frontier
        // get number of neighbors for each node in the all frontier and the total number of neighbor nodes a frontier will check
        // in: d_all_frontier_count, d_all_frontier, d_col_offset, d_row_index, d_start_col_node (to adjust frontier node to correct index in d_col_offset)
        // out: d_degrees, d_total_degrees
        
        cumulative_degree<<< (all_frontier_count + 1024 - 1) / 1024, 1024 >>>(all_frontier_count, d_col_offset, d_row_index, d_all_frontier, d_degrees, d_total_degrees, d_start_col_node);
        nvshmem_barrier_all();

        // get prefix sum of d_degrees for
        // in: d_degrees, d_all_frontier_count
        // out: d_cumulative_degrees
        ps.sum_scan_blelloch(d_cumulative_degrees, d_degrees, all_frontier_count);

        // copy total_degrees from device to host to specify kernel size for expansion
        cudaMemcpy(&total_degrees, d_total_degrees, sizeof(int), cudaMemcpyDeviceToHost);
        nvshmem_barrier_all();

        // std::cout << "HI\n";


        /*
        - performs binary search on d_cumulative_degrees to find index of frontier array that a thread is responsible for (index k)
        - each thread is responsible for one neighbor, so it the node in the frontier array, finds it's neighbor with respect to thread id cumulative degrees 
            - d_row_index[d_col_offset[u] + global_tid - d_cumulative_degrees[k]]
        - it then checks and updates local bmap
        - finds target pe to put v in it's destination vertices
        - in:
        - out: d_dest_verts
        */
        // std::cout << mype << " total degrees " << total_degrees << std::endl;
        // nvshmem_barrier_all();
        // CUDA_CHECK(cudaMemset(d_dest_verts_count, 0, sizeof(int)));
        nvshmem_barrier_all();
        int kernel_threads = (total_degrees == 0) ? 100 : total_degrees;
        expand_frontier<<< (kernel_threads + 1024 - 1) / 1024, 1024 >>>(d_all_frontier, all_frontier_count, iteration, d_cumulative_degrees, d_col_offset, d_row_index, d_bmap, d_distance, 
                                                                        d_pred, total_degrees, d_dest_verts, d_dest_verts_count, d_frontier_count, d_start_col_node, total_nodes, pe_rows, 
                                                                        pe_cols, mype, start_node, d_te);
        cudaDeviceSynchronize();
        nvshmem_barrier_all();
        
        CUDA_CHECK(cudaMemcpy(&dest_vert_count, d_dest_verts_count, sizeof(int), cudaMemcpyDeviceToHost));
        nvshmem_barrier_all();

        CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(int)));

        /* fold */
        // update local frontier
        // in: d_dest_verts, d_dest_verts_count
        // out: d_frontier, d_frontier_count
        nvshmem_barrier_all();
        update_frontier<<< (dest_vert_count + 1024 - 1) / 1024, 1024 >>>(d_frontier, d_dest_verts, d_frontier_count, d_dest_verts_count, d_bmap, start_node, iteration, d_distance, d_te);
        // cudaDeviceSynchronize();
        nvshmem_barrier_all();

        int frontier_count;
        cudaMemcpy(&frontier_count, d_frontier_count, sizeof(int), cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        nvshmem_barrier_all();

        // copy frontier count to host to check whether to continue for all processor frontier count
        nvshmem_int_sum_reduce(NVSHMEM_TEAM_WORLD, d_global_frontier_count, d_frontier_count, 1);
        cudaMemcpy(&global_frontier_count, d_global_frontier_count, sizeof(int), cudaMemcpyDeviceToHost);
        nvshmem_barrier_all();
        
        if (global_frontier_count == 0) break;

        iteration++;
    }

    /* CUDA EVENTS FOR TIMING */
    CUDA_CHECK(cudaEventRecord(gpu_end));
    CUDA_CHECK(cudaEventSynchronize(gpu_end));
    CUDA_CHECK(cudaEventElapsedTime(&exec_time, gpu_start, gpu_end));

    
    float* d_exec_time = (float *)nvshmem_malloc(sizeof(float));
    float* d_global_exec_time = (float *)nvshmem_malloc(sizeof(float));
    cudaMemcpy(d_exec_time, &exec_time, sizeof(float), cudaMemcpyHostToDevice);
    nvshmem_barrier_all();
    nvshmem_float_max_reduce(NVSHMEM_TEAM_WORLD, d_global_exec_time, d_exec_time, 1);
    nvshmem_barrier_all();
    cudaMemcpy(&exec_time, d_global_exec_time, sizeof(float), cudaMemcpyDeviceToHost);
    nvshmem_free(d_exec_time);
    nvshmem_free(d_global_exec_time);

    nvshmem_barrier_all();
    gpuBFS::get_device_distance(graph);
    nvshmem_barrier_all();
    

    CUDA_CHECK(cudaMemcpy(&te, d_te, sizeof(int), cudaMemcpyDeviceToHost));

}

void gpuBFS::get_device_distance(csc &graph) 
{
    if (nvshmem_my_pe() == 0) 
    {
        CUDA_CHECK(cudaMemcpy(host_distance, d_distance, pe_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        for (int target_pe = 1; target_pe < nvshmem_n_pes(); target_pe++) 
        {
            int *temp_array = (int *)nvshmem_malloc(pe_nodes * sizeof(int));

            nvshmem_int_get(temp_array, d_distance, pe_nodes, target_pe);

            int p_i = target_pe / pe_cols;
            int p_j = target_pe % pe_cols;
            int p_block = p_j * pe_rows + p_i;
            int offset = p_block * pe_nodes;

            // std::cout << offset << std::endl;

            CUDA_CHECK(cudaMemcpy(host_distance + offset, temp_array, pe_nodes * sizeof(int), cudaMemcpyDeviceToHost));
            nvshmem_free(temp_array);
        }
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(host_distance, d_distance, pe_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

void gpuBFS::init_source(int mype, int source)
{
    // check which pe to put source in
    int num_pes = nvshmem_n_pes();
    int p_i = mype / pe_cols;
    int p_j = mype % pe_cols;
    int p_block = p_j * pe_rows + p_i;

    int num_blocks = pe_cols*pe_rows;
    int nodes_per_block = total_nodes/num_blocks;
    int source_block = source/nodes_per_block;
    int blocks_first_node = source_block*nodes_per_block;

    if (source_block == p_block)
    {
        int value1 = 1;
        int value2 = 0;
        
        int source_position = source - blocks_first_node;

        // CUDA_CHECK(cudaMemset(d_frontier_count, 1, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_frontier_count, &value1, sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemset(d_frontier[0], &source, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_frontier + 0, &source, sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemset(d_bmap[source], 1, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bmap + source, &value1, sizeof(int), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemset(d_distance[source_position], 0, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_distance + source_position, &value2, sizeof(int), cudaMemcpyHostToDevice));
    }
    else
    {
        int value1 = 0;
        // CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_frontier_count, &value1, sizeof(int), cudaMemcpyHostToDevice));
    }
}


void gpuBFS::init_device(csc &graph, int mype)
{
    // get csc arrays
    std::vector<int> col_offset = graph.csc_vect[mype].first;
    std::vector<int> row_index = graph.csc_vect[mype].second;
    
    if (nvshmem_my_pe() == 0)
    {
        host_distance = (int *)malloc(total_nodes * sizeof(int));
    }
    else
    {
        host_distance = (int *)malloc(pe_nodes * sizeof(int));
    }
    

    // initialize device memory
    d_dest_verts_count = (int *)nvshmem_malloc(sizeof(int));
    d_dest_verts = (int *)nvshmem_malloc(10 * total_nodes * sizeof(int));

    d_frontier = (int *)nvshmem_malloc(10 * total_nodes * sizeof(int));
    d_bmap = (int *)nvshmem_malloc(total_nodes * sizeof(int));
    d_distance = (int *)nvshmem_malloc(pe_nodes * sizeof(int));
    d_degrees = (int *)nvshmem_malloc(10 * total_nodes * sizeof(int));
    d_total_degrees = (int *)nvshmem_malloc(sizeof(int));
    d_cumulative_degrees = (int *)nvshmem_malloc(10 * total_nodes * sizeof(int));
    d_frontier_count = (int *)nvshmem_malloc(sizeof(int));
    d_start_col_node = (int *) nvshmem_malloc(sizeof(int));
    d_global_frontier_count = (int *)nvshmem_malloc(sizeof(int));
    d_all_frontier_count = (int *)nvshmem_malloc(sizeof(int));
    d_all_frontier = (int *)nvshmem_malloc(10 * total_nodes * sizeof(int));

    // d_col_offset = (int *)nvshmem_malloc(total_nodes * sizeof(int));
    // d_row_index = (int *)nvshmem_malloc(total_nodes * sizeof(int));
    d_col_offset = (int *)nvshmem_malloc(col_offset.size() * sizeof(int));
    d_row_index = (int *)nvshmem_malloc(row_index.size() * sizeof(int));

    d_te = (int *)nvshmem_malloc(sizeof(int));


    CUDA_CHECK(cudaMemcpy(d_col_offset, col_offset.data(), col_offset.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_index, row_index.data(), row_index.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_frontier, 0, 10 * total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_bmap, 0, total_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_distance, -1, pe_nodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_dest_verts_count, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_te, 0, sizeof(int)));

    int p_i = mype / pe_cols;
    int p_j = mype % pe_cols;
    // int vertex_block = p_j*pe_rows + p_i;
    // int nodes_per_col = total_nodes/pe_cols;
    int start_col_node = (total_nodes/pe_cols) * p_j;

    // std::cout << mype << " start_col_node " << start_col_node << std::endl;

    CUDA_CHECK(cudaMemcpy(d_start_col_node, &start_col_node, sizeof(int), cudaMemcpyHostToDevice));
    
}


void gpuBFS::print_distance(csc &graph)
{
    if (nvshmem_my_pe() == 0) 
    {
        std::cout << "\n------GPU DISTANCE VECTOR------" << std::endl;

        std::cout << nvshmem_my_pe() << std::endl;

        for (int i = 0; i < total_nodes; i++) 
        {
            std::cout << host_distance[i] << " | ";
        }
        std::cout << std::endl;
    }
}

void gpuBFS::check_distance(std::vector<int>& correct_distance)
{
    if (nvshmem_my_pe() == 0)
    {
        int num_mismatch = 0;

        for (int i = 0; i < total_nodes; i++) 
        {

            if (host_distance[i] != correct_distance[i])
            {
                num_mismatch++;
            }
            
        }
        std::cout << "Number of mismatches: " << num_mismatch << std::endl;
    }
}

void gpuBFS::print_stats(int total_edges_traversed)
{
    if (nvshmem_my_pe() == 0)
    {
        std::cout << exec_time << " ms" << std::endl;
        std::cout << total_edges_traversed/(exec_time/1000) << " TEPS" << std::endl;
    }
}

gpuBFS::~gpuBFS() 
{
    nvshmem_free(d_col_offset);
    nvshmem_free(d_row_index);
    nvshmem_free(d_frontier);
    nvshmem_free(d_bmap);
    nvshmem_free(d_distance);
    nvshmem_free(d_degrees);
    nvshmem_free(d_total_degrees);
    nvshmem_free(d_cumulative_degrees);
    nvshmem_free(d_frontier_count);
    nvshmem_free(d_dest_verts);
    nvshmem_free(d_start_col_node);
    nvshmem_free(d_dest_verts_count);
    nvshmem_free(d_global_frontier_count);
    nvshmem_free(d_all_frontier_count);
    nvshmem_free(d_all_frontier);
    nvshmem_finalize();
}