#include "gpuBFS.cuh"

gpuBFS::gpuBFS(csr &graph, int source) 
{

    int mype_node;
    int mype;
    int npes;
    int num_nodes_per_pe;

    h_q_count = 1;

    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    // init queue
    nvshmem_malloc(&d_in_q, graph.num_nodes * sizeof(int));
    nvshmem_malloc(&d_out_q, graph.num_nodes * sizeof(int));
    nvshmem_malloc(&d_q_count, sizeof(int));
    cudaMemcpy(d_in_q, &source, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_count, &h_q_count, sizeof(int), cudaMemcpyHostToDevice);

    // init distance
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));
    nvshmem_malloc(&d_distance, graph.num_nodes * sizeof(int));
    init_distance_kernel<<< (graph.num_nodes+1024-1)/1024, 1024 >>>(graph.num_nodes, d_distance, source);

    // init graph
    nvshmem_malloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int));
    nvshmem_malloc(&d_col_idx, graph.num_edges * sizeof(int));
    cudaMemcpy(d_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);
    
    total_edges_traversed = 0;
    int *d_edges_traversed;
    cudaMalloc(&d_edges_traversed, sizeof(int));
    cudaMemset(d_edges_traversed, 0, sizeof(int));

    cudaDeviceSynchronize();

    int iteration = 0;
    while(h_q_count > 0)
    {
        cudaMemset(d_q_count,0,sizeof(int));
        // linear_bfs<<< (h_q_count+1024-1)/1024, 1024 >>>(graph.num_nodes, d_row_offset, d_col_idx, d_distance, iteration, d_in_q, h_q_count, d_out_q, d_q_count, d_edges_traversed);
        expand_contract_bfs<<< (h_q_count+1024-1)/1024, 1024 >>>(graph.num_nodes, d_row_offset, d_col_idx, d_distance, iteration, d_in_q, h_q_count, d_out_q, d_q_count);
        cudaMemcpy(&h_q_count, d_q_count, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(d_in_q,d_out_q);
        iteration++;
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_edges_traversed, d_edges_traversed, sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);
}

gpuBFS::~gpuBFS() 
{
    free(host_distance);

    nvshmem_free(d_row_offset);
    nvshmem_free(d_col_idx);

    nvshmem_free(d_distance);
    nvshmem_free(d_in_q);
    nvshmem_free(d_out_q);
    nvshmem_free(d_q_count);
}

void gpuBFS::print_distance(csr &graph)
{
    std::cout << "\n------GPU DISTANCE VECTOR------" << std::endl;

    for (long long int i = 0; i < graph.num_nodes; i++) 
    {
        std::cout << host_distance[i] << " | ";
    }
    std::cout << std::endl;
}

