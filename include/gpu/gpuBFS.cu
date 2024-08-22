#include "gpuBFS.cuh"

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

gpuBFS::gpuBFS(csr &graph, int source) 
{
    

    /* NVSHMEM INIT */
    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaSetDevice(mype_node));

    /* SPLIT NODES ACROSS GPUS */ 
    graph_nodes_gpu = graph.num_nodes / npes;
    int remainder = graph.num_nodes % npes;

    if (mype < remainder) {
        start_node = mype * (graph_nodes_gpu + 1);
        graph_nodes_gpu += 1;
    } else {
        start_node = mype * graph_nodes_gpu + remainder;
    }
    end_node = start_node + graph_nodes_gpu - 1;

    int start_idx = graph.row_offset[start_node];
    int end_idx = (mype + 1 < npes) ? graph.row_offset[end_node + 1] : graph.row_offset[graph.num_nodes];
    graph_edges_gpu = end_idx - start_idx;

    if (source >= start_node && source <= end_node)
    {
        h_q_count = 1;
    }
    else
    {
        h_q_count = 0;
    }

    /* INIT DEVICE DATA */
    gpuBFS::init_device(graph, source);

    /* CUDA EVENTS FOR TIMING */
    cudaEvent_t gpu_start, gpu_end;
    CUDA_CHECK(cudaEventCreate(&gpu_start));
    CUDA_CHECK(cudaEventCreate(&gpu_end));
    CUDA_CHECK(cudaEventRecord(gpu_start));
    
    total_edges_traversed = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    int iteration = 0;
    int global_q_count = 1; // Start with a positive value to enter the loop

    while(global_q_count > 0) 
    {
        // std::cout << "pe: " << mype << " h_q_count: " << h_q_count << std::endl;

        CUDA_CHECK(cudaMemset(d_q_count, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_edges_traversed, 0, sizeof(int)));
        
        // Launch the BFS kernel
        // (h_q_count + 1024 - 1) / 1024, 1024 
        linear_bfs<<< (h_q_count + 1024 - 1) / 1024, 1024 >>>(
            graph.num_nodes, starting_col_idx_pre_pe, d_start_node, d_end_node, 
            d_row_offset, d_col_idx, d_distance_local, iteration, d_in_q, 
            h_q_count, d_out_q, d_q_count, d_edges_traversed
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        
        // Copy d_q_count from device to host for local PE
        CUDA_CHECK(cudaMemcpy(&h_q_count, d_q_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        nvshmem_barrier_all();

        // Perform global sum reduction for d_q_count across all PEs
        nvshmem_uint32_or_reduce(NVSHMEM_TEAM_WORLD, d_global_q_count, d_q_count, 1);
        nvshmem_barrier_all();

        // Copy the global sum from device to host to check the result
        CUDA_CHECK(cudaMemcpy(&global_q_count, d_global_q_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Swap queues for the next iteration
        std::swap(d_in_q, d_out_q);
        iteration++;

        CUDA_CHECK(cudaMemcpy(&total_edges_traversed, d_edges_traversed, sizeof(int), cudaMemcpyDeviceToHost));
        nvshmem_barrier_all();
    }

    /* CUDA EVENTS FOR TIMING */
    CUDA_CHECK(cudaEventRecord(gpu_end));
    CUDA_CHECK(cudaEventSynchronize(gpu_end));
    CUDA_CHECK(cudaEventElapsedTime(&exec_time, gpu_start, gpu_end));

    /* COPY BACK TO HOST */
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_CHECK(cudaMemcpy(&total_edges_traversed, d_edges_traversed, sizeof(int), cudaMemcpyDeviceToHost));

    gpuBFS::get_device_distance(graph);

    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
}

void gpuBFS::get_device_distance(csr &graph) 
{
    if (mype == 0) {
        int base_nodes_per_pe = graph.num_nodes / npes;
        int remainder_nodes = graph.num_nodes % npes;

        CUDA_CHECK(cudaMemcpy(host_distance, d_distance_local, (base_nodes_per_pe + (mype < remainder_nodes ? 1 : 0)) * sizeof(int), cudaMemcpyDeviceToHost));

        for (int target_pe = 1; target_pe < npes; target_pe++) {
            int array_size = base_nodes_per_pe + (target_pe < remainder_nodes ? 1 : 0);
            int *temp_array = (int *)nvshmem_malloc(array_size * sizeof(int));

            nvshmem_int_get(temp_array, d_distance_local, array_size, target_pe);
            int offset = target_pe * base_nodes_per_pe + min(target_pe, remainder_nodes);

            CUDA_CHECK(cudaMemcpy(host_distance + offset, temp_array, array_size * sizeof(int), cudaMemcpyDeviceToHost));
            nvshmem_free(temp_array);
        }
    }
}

void gpuBFS::init_device(csr &graph, int source) 
{
    d_global_q_count = (uint32_t *)nvshmem_malloc(sizeof(uint32_t));  // Allocate memory on device

    d_start_node = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemcpy(d_start_node, &start_node, sizeof(int), cudaMemcpyHostToDevice));
    d_end_node = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemcpy(d_end_node, &end_node, sizeof(int), cudaMemcpyHostToDevice));
    d_graph_edges_gpu = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemcpy(d_graph_edges_gpu, &graph_edges_gpu, sizeof(int), cudaMemcpyHostToDevice));

    d_edges_traversed = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemset(d_edges_traversed, 0, sizeof(int)));

    d_in_q = (int *)nvshmem_malloc(graph.num_edges * sizeof(int));
    d_out_q = (int *)nvshmem_malloc(graph.num_edges * sizeof(int));
    // d_in_q = (int *)nvshmem_malloc(graph_nodes_gpu * sizeof(int));
    // d_out_q = (int *)nvshmem_malloc(graph_nodes_gpu * sizeof(int));
    d_q_count = (uint32_t *)nvshmem_malloc(sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(d_in_q, &source, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q_count, &h_q_count, sizeof(uint32_t), cudaMemcpyHostToDevice));

    if (mype == 0) {
        host_distance = (int *)malloc(graph.num_nodes * sizeof(int));
    }

    d_distance_local = (int *)nvshmem_malloc(graph_nodes_gpu * sizeof(int));
    init_distance_kernel<<< (graph_nodes_gpu + 1024 - 1) / 1024, 1024 >>>(d_start_node, d_end_node, graph_nodes_gpu, d_distance_local, source);
    nvshmem_barrier_all();

    d_row_offset = (int *)nvshmem_malloc((graph_nodes_gpu + 1) * sizeof(int));
    d_col_idx = (int *)nvshmem_malloc(graph_edges_gpu * sizeof(int));
    CUDA_CHECK(cudaMemcpy(d_row_offset, graph.row_offset + start_node, (graph_nodes_gpu + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    starting_col_idx_pre_pe = 0;
    if (mype != 0) {
        for (int i = 0; i < mype; i++) {
            starting_col_idx_pre_pe += nvshmem_int_g(d_graph_edges_gpu, i);
        }
    }

    std::cout << "pe: " << mype << " graph_nodes_gpu: " << graph_nodes_gpu << " graph_edges_gpu: " << graph_edges_gpu << " start_node: " << start_node << " end_node: " << end_node << " starting_col_idx_pre_pe: " << starting_col_idx_pre_pe << std::endl;

    CUDA_CHECK(cudaMemcpy(d_col_idx, graph.col_idx + starting_col_idx_pre_pe, graph_edges_gpu * sizeof(int), cudaMemcpyHostToDevice));
}

gpuBFS::~gpuBFS() 
{
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    if (mype == 0) {
        free(host_distance);
    }

    /* NVSHMEM FINALIZE AND MEM CLEANUP */
    nvshmem_free(d_row_offset);
    nvshmem_free(d_col_idx);
    nvshmem_free(d_distance_local);
    nvshmem_free(d_in_q);
    nvshmem_free(d_out_q);
    nvshmem_free(d_q_count);
    nvshmem_free(d_start_node);
    nvshmem_free(d_end_node);
    nvshmem_free(d_edges_traversed);
    nvshmem_free(d_graph_edges_gpu);
    nvshmem_free(d_global_q_count);
    nvshmem_finalize();
}

void gpuBFS::print_distance(csr &graph) 
{
    if (mype == 0) {
        std::cout << "\n------GPU DISTANCE VECTOR------" << std::endl;
        for (long long int i = 0; i < graph.num_nodes; i++) {
            std::cout << host_distance[i] << " | ";
        }
        std::cout << std::endl;
    }
}