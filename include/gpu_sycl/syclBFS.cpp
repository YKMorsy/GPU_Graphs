#include "syclBFS.h"

const int HASH_RANGE = 128;
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 64;
const int WARPS = BLOCK_SIZE/WARP_SIZE;

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
    *device_out_queue_size = host_cur_queue_size;

    int iteration = 0;

    int num_blocks;
    
    // loop until frontier is empty
    while (host_cur_queue_size > 0)
    {
        *device_out_queue_size = 0;

        num_blocks = (host_cur_queue_size % BLOCK_SIZE == 0)?(host_cur_queue_size/BLOCK_SIZE):(host_cur_queue_size/BLOCK_SIZE+1);

        gpuQueue.submit([&](cl::sycl::handler &cgh) 
        {
            int *device_col_idx_c = device_col_idx;
            int *device_row_offset_c =  device_row_offset;
            int num_nodes_c =  graph_num_nodes;
            int *device_in_queue_c =  device_in_queue;
            int device_in_queue_size_c =  host_cur_queue_size;
            int *device_out_queue_size_c =  device_out_queue_size;
            int *device_distance_c = device_distance;
            int iteration_c = iteration;
            int *device_out_queue_c = device_out_queue;

            sycl::local_accessor<int, 1> comm(sycl::range<1>(3), cgh);
            sycl::local_accessor<int, 1> base_offset(sycl::range<1>(1), cgh);
            sycl::local_accessor<int, 1> sums(sycl::range<1>(BLOCK_SIZE), cgh);

            cgh.parallel_for
            (
                cl::sycl::nd_range<1>(num_blocks*max_group_size, max_group_size),
                [=] (cl::sycl::nd_item<1> item) 
                {
                    expand_contract_kernel(
                        device_col_idx_c, device_row_offset_c,
                        num_nodes_c, device_in_queue_c,
                        device_in_queue_size_c, device_out_queue_size_c,
                        device_distance_c, iteration_c,
                        device_out_queue_c, item, comm.get_multi_ptr<sycl::access::decorated::no>(),
                        base_offset.get_multi_ptr<sycl::access::decorated::no>(), sums.get_multi_ptr<sycl::access::decorated::no>());
                }
            );
        }).wait();

        host_cur_queue_size = *device_out_queue_size;
        std::swap(device_in_queue, device_out_queue);

        iteration++;
    }

    // copy device distance to host
    gpuQueue.memcpy(host_distance, device_distance, graph_num_nodes * sizeof(int)).wait();

    host_distance[source] = 0;
}

void syclBFS::expand_contract_kernel(int *device_col_idx, int *device_row_offset, 
                            int num_nodes, int *device_in_queue, 
                            int device_in_queue_size, int *device_out_queue_size, 
                            int *device_distance, int iteration, int *device_out_queue,
                            cl::sycl::nd_item<1> &item, int *comm, int *base_offset, int *sums)
{
    int th_id = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0); // block_id*num_threads_in_block + thread_id
    // loop to process all threads and synchronize threads within a block
    // synchronize only if at least one of the th_id is less than the queue size
    do
    {
        int cur_node = th_id < device_in_queue_size ? device_in_queue[th_id] : -1;

        int row_offset_start = cur_node < 0 ? 0 : device_row_offset[cur_node];
        int row_offset_end = cur_node < 0 ? 0 : device_row_offset[cur_node+1];

        bool big_list = (row_offset_end - row_offset_start) >= BLOCK_SIZE;

        block_gather(device_col_idx, device_distance, iteration, device_out_queue, device_out_queue_size, row_offset_start, big_list ? row_offset_end : row_offset_start, item, comm, base_offset, sums);
        fine_gather(device_col_idx, row_offset_start,  big_list ? row_offset_start : row_offset_end, device_distance, iteration, device_out_queue, device_out_queue_size, cur_node, item, comm, base_offset, sums);

        th_id += item.get_group_range(0) * item.get_local_range(0); // num_blocks_in_grid*num_threads_in_block

        item.barrier();
    }
    while((sycl::any_of_group(item.get_group(), th_id < device_in_queue_size)));
}

void syclBFS::block_gather(int* column_index, int* distance, 
                           int iteration, int * out_queue, 
                           int* out_queue_count, int r, int r_end, 
                           cl::sycl::nd_item<1> &item, int *comm,
                           int *base_offset, int *sums)
{
    int orig_row_start = r;
    item.barrier();
	while((sycl::any_of_group(item.get_group(), r < r_end)))
	{
		// Vie for control of block.
		if(r < r_end)
			comm[0] = item.get_local_id(0);
		item.barrier();
		if(comm[0] == item.get_local_id(0))
		{
			// If won, share your range to the entire block.
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
		}
		item.barrier();
		int r_gather = comm[1] + item.get_local_id(0);
		const int r_gather_end = comm[2];
		const int total = comm[2] - comm[1];
		int block_progress = 0;
		while((total - block_progress) > 0)
		{
			int neighbor = -1;
			int valid = 0;
			if (r_gather < r_gather_end)
			{
				neighbor = column_index[r_gather];
				// Look up status of current neighbor.
				if (distance[neighbor] == -1)
                {
                    valid = 1;
					// Update label.
					distance[neighbor] = iteration + 1;
				}
			}
			// Obtain offset in queue by computing prefix sum
			const prescan_result prescan = block_prefix_sum(valid, item, sums);

			// Obtain base enqueue offset and share it to whole block.
			if(item.get_local_id(0) == 0)
            {
                int old_value = *out_queue_count;
                *out_queue_count += prescan.total;
                base_offset[0] = old_value;
            }
                // cl::sycl::atomic_fetch_add<cl::sycl::access::address_space::generic_space>(
                //                      out_queue_count, prescan.total);
                                    
			item.barrier();

			// Write vertex to the out queue.
			if (valid == 1)
				out_queue[base_offset[0]+prescan.offset] = neighbor;

			r_gather += BLOCK_SIZE;
			block_progress+= BLOCK_SIZE;
			item.barrier();
		}
	}
}

void syclBFS::fine_gather(int *device_col_idx, int row_offset_start, 
                        int row_offset_end, int *device_distance, 
                        int iteration, int *device_out_queue, 
                        int *device_out_queue_size, const int node,
                        cl::sycl::nd_item<1> &item, int *comm,
                        int *base_offset, int *sums)
{
    prescan_result rank = block_prefix_sum(row_offset_end-row_offset_start, item, sums);

    int cta_progress = 0;

    while((rank.total - cta_progress) > 0)
    {
        int orig_row_start = row_offset_start;
        while ((rank.offset < cta_progress + BLOCK_SIZE) && (row_offset_start < row_offset_end))
        {
            // add index to shared memory
            comm[rank.offset - cta_progress] = row_offset_start;
            rank.offset++;
            row_offset_start++;
        }

        item.barrier();

        int neighbor;
        int valid = 0;

        if (item.get_local_id(0) < (rank.total - cta_progress))
        {
            neighbor = device_col_idx[comm[item.get_local_id(0)]];
            if (device_distance[neighbor] == -1)
            {
                valid = 1;
                device_distance[neighbor] = iteration + 1;
            }
        } 

        item.barrier();
        const prescan_result prescan = block_prefix_sum(valid, item, sums);
        
        if (item.get_local_id(0) == 0)
        {
            // cl::sycl::atomic_ref<(*int), cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> *base_offset(*device_out_queue_size);

            // *base_offset.fetch_add(prescan.total);

            int old_value = *device_out_queue_size;
            *device_out_queue_size += prescan.total;
            base_offset[0] = old_value;

            // base_offset[0] = at_sum;

            // base_offset[0] = atomic::atomic_fetch_add<cl::sycl::access::address_space::generic_space>(
            //                         device_out_queue_size, prescan.total);
        }

        item.barrier();

		const int queue_index = base_offset[0] + prescan.offset;

        if (valid == 1)
        {
            device_out_queue[queue_index] = neighbor;
        }

        cta_progress += BLOCK_SIZE;

        item.barrier();
    }

}

prescan_result syclBFS::block_prefix_sum(int val, cl::sycl::nd_item<1> &item, int *sums)
{

    prescan_result result;

    int thid = item.get_local_id(0);
    sums[thid] = val;

    item.barrier();

    int os = 1;

    for (int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        item.barrier();
        if (thid < d) {
            int ai = os * (2 * thid+1) - 1;
            int bi = os * (2 * thid+2) - 1;
            sums[bi] += sums[ai];
        }

        os *= 2;
    }

    if (thid == 0) { 
        // result.block_sum = block_data[blockDim.x - 1];
        sums[BLOCK_SIZE - 1] = 0; // Clear the last element
    }

    for (int d = 1; d < BLOCK_SIZE; d *= 2) {

        os /= 2;

        item.barrier();
        if (thid < d) {
            int ai = os * (2 * thid+1) - 1;
            int bi = os * (2 * thid+2) - 1;
            int t = sums[ai];

            sums[ai] = sums[bi];
            sums[bi] += t;
        }
    }

    item.barrier();

    result.offset = sums[thid];
    result.total = sums[BLOCK_SIZE - 1];

    return result;
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

    int num_blocks = (graph_num_nodes + max_group_size - 1) / max_group_size;

    gpuQueue.submit([&](cl::sycl::handler &cgh) 
    {
        int *distance_c = device_distance;
        int graph_num_nodes_c = graph_num_nodes;

        cgh.parallel_for
        (
            cl::sycl::nd_range<1>(num_blocks*max_group_size, max_group_size),
            [=] (cl::sycl::nd_item<1> item) 
            {
                int i = item.get_global_id(0);
                if (i < graph_num_nodes_c)
                {
                    distance_c[i] = -1;
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