#include "syclBFS.hpp"

syclBFS::syclBFS(csr &graph, int source)
{
    graph_num_nodes = graph.num_nodes;
    graph_num_edges = graph.num_edges;

    // initialize queue and sizes
    init_queue(graph);

    // initialize distance with -1 on host
    init_distance(graph);

    // initialize device graph variables
    init_graph_for_device(graph);

    // start with source (update distance and queue)
    host_distance[source] = 0;
    host_queue[0] = source;
    host_cur_queue_size = host_cur_queue_size + 1;

    // copy host to device queue
    gpuQueue.memcpy(device_in_queue, host_queue, graph_num_nodes * sizeof(int)).wait();
    device_out_queue_size = host_cur_queue_size;

    int iteration = 0;
    
    // loop until frontier is empty

    // copy device distance to host

}

void syclBFS::init_queue(csr &graph)
{
    // allocate host memory
    host_queue = (int *)malloc(graph_num_nodes * sizeof(int));
    host_cur_queue_size = 0;
    host_cur_queue_size = 0;

    // allocate device memory
    device_in_queue = cl::sycl::malloc_device<int>(graph_num_nodes, gpuQueue);
    device_out_queue = cl::sycl::malloc_device<int>(graph_num_nodes, gpuQueue);
    device_out_queue_size = 0;
}

void syclBFS::init_distance(csr &graph)
{
    host_distance = (int *)(malloc(graph_num_nodes * sizeof(int)));
    device_distance = cl::sycl::malloc_device<int>(graph_num_nodes, gpuQueue);

    int max_group_size = gpuQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int num_blocks = (graph_num_nodes + max_group_size - 1) / max_group_size;

    gpuQueue.submit([&](cl::sycl::handler &cgh) 
    {
        int *distance_ptr = device_distance;
        int graph_num_nodes_cp = graph_num_nodes;

        cgh.parallel_for
        (
            cl::sycl::nd_range<1>(num_blocks*max_group_size, max_group_size),
            [=] (cl::sycl::nd_item<1> item) 
            {
                int i = item.get_global_id(0);
                if (i < graph_num_nodes_cp)
                {
                    distance_ptr[i] = -1;
                }
            }
        );
    }).wait();

    // Copy back to host
    gpuQueue.memcpy(host_distance, device_distance, graph_num_nodes * sizeof(int)).wait();
}

void syclBFS::init_graph_for_device(csr &graph)
{
    device_col_idx = cl::sycl::malloc_device<int>(graph_num_edges, gpuQueue);
    device_row_offset = cl::sycl::malloc_device<int>((graph_num_nodes+1), gpuQueue);

    gpuQueue.memcpy(device_col_idx, graph.col_idx, graph_num_edges * sizeof(int)).wait();
    gpuQueue.memcpy(device_row_offset, graph.row_offset, (graph_num_nodes+1) * sizeof(int)).wait();
}

syclBFS::~syclBFS()
{
    free(host_distance);
    free(host_queue);

    cl::sycl::free(device_distance, gpuQueue);
    cl::sycl::free(device_in_queue, gpuQueue);
    cl::sycl::free(device_out_queue, gpuQueue);
    cl::sycl::free(device_col_idx, gpuQueue);
    cl::sycl::free(device_row_offset, gpuQueue);
}