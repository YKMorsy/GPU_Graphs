#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gpuBFS.dp.hpp"
#include <chrono>

const int HASH_RANGE = 128;
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 64;
const int WARPS = BLOCK_SIZE/WARP_SIZE;

int div_up(int dividend, int divisor)
{
	return (dividend % divisor == 0)?(dividend/divisor):(dividend/divisor+1);
}


struct prescan_result
{
    int offset;
    int total;
};

void init_distance_kernel(int *device_distance, int size,
                          const sycl::nd_item<3> &item_ct1)
{
    int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
            item_ct1.get_local_id(2);

    if (i < size)
    {
        device_distance[i] = -1;
    }
}

 int warp_cull(volatile int scratch[2][HASH_RANGE], const int v,
               const sycl::nd_item<3> &item_ct1)
{
	//unsigned int active = __ballot_sync(FULL_MASK, v >= 0);
	//if( v == -1) return v;
	const int hash = v & (HASH_RANGE-1);
        const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        if(v >= 0)
		scratch[warp_id][hash]= v;
        sycl::group_barrier(item_ct1.get_sub_group());
        const int retrieved = v >= 0 ? scratch[warp_id][hash] : v;
        sycl::group_barrier(item_ct1.get_sub_group());
        unsigned int active = sycl::reduce_over_group(
            item_ct1.get_sub_group(),
            (0xffffffff &
             (0x1 << item_ct1.get_sub_group().get_local_linear_id())) &&
                    retrieved == v
                ? (0x1 << item_ct1.get_sub_group().get_local_linear_id())
                : 0,
            sycl::ext::oneapi::plus<>());
        if (retrieved == v)
	{
		// Vie to be the only thread in warp inspecting vertex v.
                scratch[warp_id][hash] = item_ct1.get_local_id(2);
                sycl::group_barrier(item_ct1.get_sub_group());
                // Some other thread has this vertex
                if (scratch[warp_id][hash] != item_ct1.get_local_id(2))
                        return -1;
	}
	return v;
}

 prescan_result block_prefix_sum(const int val,
                                 const sycl::nd_item<3> &item_ct1,
                                 volatile int *sums)
{

        int value = val;

        const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;

        // Warp-wide prefix sums.
#pragma unroll
	for(int i = 1; i <= WARP_SIZE; i <<= 1)
	{
                /*
                DPCT1023:0: The SYCL sub-group does not support mask options for
                dpct::shift_sub_group_right. You can specify
                "--use-experimental-features=masked-sub-group-operation" to use
                the experimental helper function to migrate __shfl_up_sync.
                */
                const int n = dpct::shift_sub_group_right(
                    item_ct1.get_sub_group(), value, i);
                if (lane_id >= i)
			value += n;
	}

	// Write warp total to shared array.
	if (lane_id == WARP_SIZE - 1)
	{
		sums[warp_id] = value;
	}

        /*
        DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Prefix sum of warp sums.
	if (warp_id == 0 && lane_id < WARPS)
	{
		int warp_sum = sums[lane_id];
		const unsigned int mask = (1 << (WARPS)) - 1;
#pragma unroll
		for (int i = 1; i <= WARPS; i <<= 1)
		{
                        /*
                        DPCT1023:1: The SYCL sub-group does not support mask
                        options for dpct::shift_sub_group_right. You can specify
                        "--use-experimental-features=masked-sub-group-operation"
                        to use the experimental helper function to migrate
                        __shfl_up_sync.
                        */
                        const int n = dpct::shift_sub_group_right(
                            item_ct1.get_sub_group(), warp_sum, i, WARPS);
                        if (lane_id >= i)
				warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

        /*
        DPCT1065:21: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Add total sum of previous WARPS to current element.
	if (warp_id > 0)
	{
		const int block_sum = sums[warp_id-1];
		value += block_sum;
	}

	prescan_result result;
	result.offset = value - val;
	result.total = sums[WARPS-1];
	return result; 
}

 void block_gather(const int* const column_index, int* const distance, 
                                const int iteration, int * const out_queue, 
                                int* const out_queue_count,int r, int r_end,
                                const sycl::nd_item<3> &item_ct1,
                                volatile int *sums, volatile int *comm,
                                volatile int *base_offset)
{

    int orig_row_start = r;
        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        while ((item_ct1.barrier(),
                sycl::any_of_group(item_ct1.get_group(), r < r_end)))
        {
		// Vie for control of block.
		if(r < r_end)
                        comm[0] = item_ct1.get_local_id(2);
                /*
                DPCT1118:3: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:22: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                if (comm[0] == item_ct1.get_local_id(2))
                {
			// If won, share your range to the entire block.
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
		}
                /*
                DPCT1118:4: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065:23: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                int r_gather = comm[1] + item_ct1.get_local_id(2);
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
				if ((distance[neighbor] == -1))
                {
                    valid = 1;
					// Update label.
					distance[neighbor] = iteration + 1;
				}
			}
			// Obtain offset in queue by computing prefix sum
                        const prescan_result prescan =
                            block_prefix_sum(valid, item_ct1, sums);

                        // Obtain base enqueue offset and share it to whole block.
                        if (item_ct1.get_local_id(2) == 0)
                                base_offset[0] = dpct::atomic_fetch_add<
                                    sycl::access::address_space::generic_space>(
                                    out_queue_count, prescan.total);
                        /*
                        DPCT1118:5: SYCL group functions and algorithms must be
                        encountered in converged control flow. You may need to
                        adjust the code.
                        */
                        /*
                        DPCT1065:24: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();
                        // Write vertex to the out queue.
			if (valid == 1)
				out_queue[base_offset[0]+prescan.offset] = neighbor;

			r_gather += BLOCK_SIZE;
			block_progress+= BLOCK_SIZE;
                        /*
                        DPCT1118:6: SYCL group functions and algorithms must be
                        encountered in converged control flow. You may need to
                        adjust the code.
                        */
                        /*
                        DPCT1065:25: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();
                }
	}
}


void fine_gather(int *device_col_idx, int row_offset_start, 
                            int row_offset_end, int *device_distance, 
                            int iteration, int *device_out_queue, int *device_out_queue_size, const int node,
                            const sycl::nd_item<3> &item_ct1, volatile int *sums,
                            volatile int *comm, volatile int *base_offset)
{
    // get scatter offset and total with prefix sum
    prescan_result rank =
        block_prefix_sum(row_offset_end - row_offset_start, item_ct1, sums);
    // printf("real total %d\n", row_offset_end-row_offset_start);
    // printf("offset %d\n", prescan_offset);
    // printf("total %d\n", prescan_total);

    int cta_progress = 0;

    while((rank.total - cta_progress) > 0)
    {
        // printf("start %d\n", row_offset_start);

        // All threads pack shared memory
        int orig_row_start = row_offset_start;
        while ((rank.offset < cta_progress + BLOCK_SIZE) && (row_offset_start < row_offset_end))
        {
            // add index to shared memory
            comm[rank.offset - cta_progress] = row_offset_start;
            rank.offset++;
            row_offset_start++;
        }

        // if (row_offset_start > 0)
        // {
        //     printf("start 2 %d\n", row_offset_start);
        // }

        /*
        DPCT1118:11: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:30: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // each thread gets a neighbor to add to queue
        int neighbor;
        int valid = 0;

        if (item_ct1.get_local_id(2) < (rank.total - cta_progress))
        {
            // make sure only add neighbor if it points to something
            neighbor = device_col_idx[comm[item_ct1.get_local_id(2)]];
            // printf("node %d neighbor %d and end %d\n", node, neighbor, row_offset_end);

            if ((device_distance[neighbor] == -1))
            {
                // printf("node %d neighbor %d and start %d and end %d\n", node, neighbor, orig_row_start, row_offset_end);
                valid = 1;
                // printf("neighbor %d\n", neighbor);
                device_distance[neighbor] = iteration + 1;
            }
        }

        /*
        DPCT1118:12: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:31: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // each thread now adds neighbor to queue with index determined by new prescan offset depending on if there is a neighbor
                const prescan_result prescan =
                    block_prefix_sum(valid, item_ct1, sums);

        if (item_ct1.get_local_id(2) == 0)
        {
            base_offset[0] = dpct::atomic_fetch_add<
                sycl::access::address_space::generic_space>(
                device_out_queue_size, prescan.total);
        }

        /*
        DPCT1118:13: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:32: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

                const int queue_index = base_offset[0] + prescan.offset;

        if (valid == 1)
        {
            device_out_queue[queue_index] = neighbor;
        }

        cta_progress += BLOCK_SIZE;

        /*
        DPCT1118:14: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:33: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

void expand_contract_kernel(int *device_col_idx, int *device_row_offset, 
                            int num_nodes, int *device_in_queue, 
                            const int device_in_queue_size, int *device_out_queue_size, 
                            int *device_distance, int iteration, int *device_out_queue,
                            const sycl::nd_item<3> &item_ct1, volatile int *sums,
                            volatile int *comm, volatile int *base_offset,
                            sycl::local_accessor<volatile int, 2> scratch)
{
    int th_id = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                item_ct1.get_local_id(2);

    do
    {
        // get node from queue
        int cur_node = th_id < device_in_queue_size ? device_in_queue[th_id] : -1;

        //warp culling and history culling

                cur_node = warp_cull(scratch, cur_node, item_ct1);

        int row_offset_start = cur_node < 0 ? 0 : device_row_offset[cur_node];
        int row_offset_end = cur_node < 0 ? 0 : device_row_offset[cur_node+1];

        const bool big_list = (row_offset_end - row_offset_start) >= BLOCK_SIZE;

        // if (cur_node == 31)
        // {
        //     printf("%d\n", (row_offset_end - row_offset_start) >= BLOCK_SIZE);
        // }

        /*
        DPCT1118:15: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        block_gather(device_col_idx, device_distance, iteration,
                     device_out_queue, device_out_queue_size, row_offset_start,
                     big_list ? row_offset_end : row_offset_start, item_ct1,
                     sums, comm, base_offset);
        /*
        DPCT1118:16: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        fine_gather(device_col_idx, row_offset_start,
                    big_list ? row_offset_start : row_offset_end,
                    device_distance, iteration, device_out_queue,
                    device_out_queue_size, cur_node, item_ct1, sums, comm,
                    base_offset);
        // fine_gather(device_col_idx, device_distance, iteration, device_out_queue, device_out_queue_size, row_offset_start, big_list ? row_offset_start : row_offset_end);

        th_id += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);

    }
    // sync threads in block then perform
    // returns 1 if any thread meets condition
    /*
    DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    while (
        (item_ct1.barrier(), sycl::any_of_group(item_ct1.get_group(),
                                                th_id < device_in_queue_size)));
}

gpuBFS::gpuBFS(csr &graph, int source)
{
    // initialize queue and sizes
    init_queue(graph);

    // initialize distance with -1 on host
    init_distance(graph);

    // initialize device graph variables
    init_graph_for_device(graph);

    // start with source (update distance and queue)
    host_distance[source] = 0;
    *host_cur_queue_size = *host_cur_queue_size + 1;

    host_queue[0] = source;

    // copy host to device queue
    dpct::get_in_order_queue().memcpy(device_in_queue, host_queue,
                                      graph.num_nodes * sizeof(int));
    dpct::get_in_order_queue()
        .memcpy(device_out_queue_size, host_cur_queue_size, sizeof(int))
        .wait();

    int iteration = 0;

    dpct::event_ptr gpu_start, gpu_end;
    std::chrono::time_point<std::chrono::steady_clock> gpu_start_ct1;
    std::chrono::time_point<std::chrono::steady_clock> gpu_end_ct1;
    gpu_start = new sycl::event();
    gpu_end = new sycl::event();

    /*
    DPCT1012:34: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    gpu_start_ct1 = std::chrono::steady_clock::now();
    *gpu_start = dpct::get_in_order_queue().ext_oneapi_submit_barrier();

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        dpct::get_in_order_queue()
            .memset(device_out_queue_size, 0, sizeof(int))
            .wait();

        const int num_of_blocks = div_up(*host_cur_queue_size, BLOCK_SIZE);

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            /*
            DPCT1101:39: 'WARPS' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            sycl::local_accessor<volatile int, 1> sums_acc_ct1(
                sycl::range<1>(2 /*WARPS*/), cgh);
            sycl::local_accessor<volatile int, 1> comm_acc_ct1(
                sycl::range<1>(3), cgh);
            sycl::local_accessor<volatile int, 1> base_offset_acc_ct1(
                sycl::range<1>(1), cgh);
            /*
            DPCT1101:40: 'WARPS' expression was replaced with a value. Modify
            the code to use the original expression, provided in comments, if it
            is correct.
            */
            /*
            DPCT1101:41: 'HASH_RANGE' expression was replaced with a value.
            Modify the code to use the original expression, provided in
            comments, if it is correct.
            */
            sycl::local_accessor<volatile int, 2> scratch_acc_ct1(
                sycl::range<2>(2 /*WARPS*/, 128 /*HASH_RANGE*/), cgh);

            int *device_col_idx_ct0 = device_col_idx;
            int *device_row_offset_ct1 = device_row_offset;
            int graph_num_nodes_ct2 = graph.num_nodes;
            int *device_in_queue_ct3 = device_in_queue;
            const int host_cur_queue_size_ct4 = *host_cur_queue_size;
            int *device_out_queue_size_ct5 = device_out_queue_size;
            int *device_distance_ct6 = device_distance;
            int *device_out_queue_ct8 = device_out_queue;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_of_blocks) *
                                      sycl::range<3>(1, 1, BLOCK_SIZE),
                                  sycl::range<3>(1, 1, BLOCK_SIZE)),
                [=](sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                        expand_contract_kernel(
                            device_col_idx_ct0, device_row_offset_ct1,
                            graph_num_nodes_ct2, device_in_queue_ct3,
                            host_cur_queue_size_ct4, device_out_queue_size_ct5,
                            device_distance_ct6, iteration,
                            device_out_queue_ct8, item_ct1,
                            sums_acc_ct1.get_pointer(),
                            comm_acc_ct1.get_pointer(),
                            base_offset_acc_ct1.get_pointer(), scratch_acc_ct1);
                    });
        });
        dpct::get_current_device().queues_wait_and_throw();

        // copy device queue to host
        dpct::get_in_order_queue()
            .memcpy(host_cur_queue_size, device_out_queue_size, sizeof(int))
            .wait();
        std::swap(device_in_queue, device_out_queue);

        iteration++;
    }

    // copy device distance to host
    dpct::get_in_order_queue()
        .memcpy(host_distance, device_distance, graph.num_nodes * sizeof(int))
        .wait();

    host_distance[source] = 0;

    /*
    DPCT1012:35: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    dpct::get_current_device().queues_wait_and_throw();
    gpu_end_ct1 = std::chrono::steady_clock::now();
    *gpu_end = dpct::get_in_order_queue().ext_oneapi_submit_barrier();
    exec_time =
        std::chrono::duration<float, std::milli>(gpu_end_ct1 - gpu_start_ct1)
            .count();
}

void gpuBFS::init_graph_for_device(csr &graph)
{
    device_col_idx =
        sycl::malloc_device<int>(graph.num_edges, dpct::get_in_order_queue());
    device_row_offset = sycl::malloc_device<int>((graph.num_nodes + 1),
                                                 dpct::get_in_order_queue());

    dpct::get_in_order_queue()
        .memcpy(device_col_idx, graph.col_idx, graph.num_edges * sizeof(int))
        .wait();
    dpct::get_in_order_queue()
        .memcpy(device_row_offset, graph.row_offset,
                (graph.num_nodes + 1) * sizeof(int))
        .wait();
}

void gpuBFS::init_queue(csr &graph)
{
    // allocate host memory
    host_queue = (int *)malloc(graph.num_nodes * sizeof(int));
    host_cur_queue_size = (int *)malloc(sizeof(int));
    *host_cur_queue_size = 0;

    // allocate device memory
    device_in_queue =
        sycl::malloc_device<int>(graph.num_nodes, dpct::get_in_order_queue());
    device_out_queue =
        sycl::malloc_device<int>(graph.num_nodes, dpct::get_in_order_queue());
    device_out_queue_size =
        sycl::malloc_device<int>(1, dpct::get_in_order_queue());
}


void gpuBFS::init_distance(csr &graph)
{
    // allocate host memory
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    // allocate device memory
    device_distance =
        sycl::malloc_device<int>(graph.num_nodes, dpct::get_in_order_queue());

    // copy memory from host to device
    dpct::get_in_order_queue()
        .memcpy(device_distance, host_distance, graph.num_nodes * sizeof(int))
        .wait();

    // run kernel to inialize distance
    sycl::range<3> block(1, 1, BLOCK_SIZE);
    sycl::range<3> grid(1, 1, (graph.num_nodes + block[2] - 1) / block[2]);
    /*
    DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
        int *device_distance_ct0 = device_distance;
        int graph_num_nodes_ct1 = graph.num_nodes;

        cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             init_distance_kernel(device_distance_ct0,
                                                  graph_num_nodes_ct1,
                                                  item_ct1);
                         });
    });

    dpct::get_in_order_queue().wait();

    // copy back
    dpct::get_in_order_queue()
        .memcpy(host_distance, device_distance, graph.num_nodes * sizeof(int))
        .wait();

    dpct::get_current_device().queues_wait_and_throw();
}

gpuBFS::~gpuBFS()
{
    free(host_distance);
    free(host_queue);
    free(host_cur_queue_size);

    sycl::free(device_distance, dpct::get_in_order_queue());
    sycl::free(device_in_queue, dpct::get_in_order_queue());
    sycl::free(device_out_queue_size, dpct::get_in_order_queue());
}