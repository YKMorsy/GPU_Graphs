#include "gpuBFS.cuh"

gpuBFS::gpuBFS(csr &graph, int source) 
{
    // init queue
    cudaMalloc(&d_in_q, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_out_q, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_q_count, sizeof(int));

    h_q_count = 1;
    cudaMemcpy(d_in_q, &source, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_count, &h_q_count, sizeof(int), cudaMemcpyHostToDevice);

    // init distance
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));
    cudaMalloc(&d_distance, graph.num_nodes * sizeof(int));
    init_distance_kernel<<< (graph.num_nodes+1024-1)/1024, 1024 >>>(graph.num_nodes, d_distance, source);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // init graph
    cudaMalloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int));
    cudaMalloc(&d_col_idx, graph.num_edges * sizeof(int));
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

    cudaFree(d_row_offset);
    cudaFree(d_col_idx);

    cudaFree(d_distance);
    cudaFree(d_in_q);
    cudaFree(d_out_q);
    cudaFree(d_q_count);
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

