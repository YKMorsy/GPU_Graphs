#include "gpuBFS.cuh"

const long long int HASH_RANGE = 128;
const long long int WARP_SIZE = 32;
const long long int BLOCK_SIZE = 1024;
const long long int WARPS = BLOCK_SIZE/WARP_SIZE;

long long int div_up(long long int dividend, long long int divisor)
{
	return (dividend % divisor == 0)?(dividend/divisor):((dividend/divisor)+1);
}


struct prescan_result
{
    long long int offset;
    long long int total;
};

__global__
void init_distance_kernel(long long int *device_distance, long long int size)
{
    long long int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        device_distance[i] = -1;
    }
}

 __device__ long long int warp_cull(volatile long long int scratch[2][HASH_RANGE], const long long int v)
{
	//unsigned long long int active = __ballot_sync(FULL_MASK, v >= 0);
	//if( v == -1) return v;
	const long long int hash = v & (HASH_RANGE-1);
	const long long int warp_id = threadIdx.x / WARP_SIZE;
	if(v >= 0)
		scratch[warp_id][hash]= v;
	__syncwarp();
	const long long int retrieved = v >= 0 ? scratch[warp_id][hash] : v;
	__syncwarp();
	unsigned long long int active = __ballot_sync(0xffffffff, retrieved == v);
	if (retrieved == v)
	{
		// Vie to be the only thread in warp inspecting vertex v.
		scratch[warp_id][hash] = threadIdx.x;
		__syncwarp(active);
		// Some other thread has this vertex
		if(scratch[warp_id][hash] != threadIdx.x)
			return -1;
	}
	return v;
}


__device__ prescan_result block_prefix_sum(const long long int val) {
    __shared__ long long int block_data[BLOCK_SIZE]; // Assuming maximum block size of 1024 threads
    prescan_result result;

    long long int thid = threadIdx.x;
    block_data[thid] = val; // Assign value to block_data
    

    __syncthreads();

    long long int os = 1;

    // Compute prefix sum
    for (long long int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            long long int ai = os * (2 * thid+1) - 1;
            long long int bi = os * (2 * thid+2) - 1;
            block_data[bi] += block_data[ai];
        }

        os *= 2;
    }

    if (thid == 0) { 
        // result.block_sum = block_data[blockDim.x - 1];
        block_data[blockDim.x - 1] = 0; // Clear the last element
    }

    for (long long int d = 1; d < blockDim.x; d *= 2) {

        os /= 2;

        __syncthreads();
        if (thid < d) {
            long long int ai = os * (2 * thid+1) - 1;
            long long int bi = os * (2 * thid+2) - 1;
            long long int t = block_data[ai];

            block_data[ai] = block_data[bi];
            block_data[bi] += t;
        }
    }

    __syncthreads();
    
    result.offset = block_data[thid];
    result.total = block_data[blockDim.x - 1];

    return result;
}

 __device__ void block_gather(const long long int* const column_index, long long int* const distance, 
                                const long long int iteration, long long int * const out_queue, 
                                long long int* const out_queue_count,long long int r, long long int r_end)
{
	volatile __shared__ long long int comm[3];
    // long long int orig_row_start = r;
	while(__syncthreads_or(r < r_end))
	{
		// Vie for control of block.
		if(r < r_end)
			comm[0] = threadIdx.x;
		__syncthreads();
		if(comm[0] == threadIdx.x)
		{
			// If won, share your range to the entire block.
			comm[1] = r;
			comm[2] = r_end;
			r = r_end;
		}
		__syncthreads();
		long long int r_gather = comm[1] + threadIdx.x;
		long long int r_gather_end = comm[2];
		const long long int total = comm[2] - comm[1];

        long long int block_progress = 0;
        while((total - block_progress) > 0)
        {
            long long int valid = 0;
            long long int neighbor;
            if (r_gather < r_gather_end) {
                neighbor = column_index[r_gather];
                if (distance[neighbor] == -1) {
                    valid = 1;
                    distance[neighbor] = iteration + 1;
                }
            }

            __syncthreads();

            const prescan_result prescan = block_prefix_sum(valid);
            volatile __shared__ long long int base_offset[1];
            if(threadIdx.x == 0)
            {
                // base_offset[0] = atomicAdd(out_queue_count,prescan.total);
                long long int old_value = *out_queue_count;
                *out_queue_count += prescan.total;
                base_offset[0] = old_value;
            }

            __syncthreads();

            if (valid == 1)
            {
                out_queue[base_offset[0] + prescan.offset] = neighbor;
            }

            r_gather += BLOCK_SIZE;
            block_progress+= BLOCK_SIZE;

            __syncthreads();
        }



		// long long int block_progress = 0;
		// while((total - block_progress) > 0)
		// {
		// 	long long int neighbor = -1;
		// 	long long int valid = 0;
		// 	if (r_gather < r_gather_end)
		// 	{
		// 		neighbor = column_index[r_gather];
		// 		// Look up status of current neighbor.
		// 		if ((distance[neighbor] == -1))
        //         {
        //             valid = 1;
		// 			// Update label.
		// 			distance[neighbor] = iteration + 1;
		// 		}
		// 	}
        //     __syncthreads();
		// 	// Obtain offset in queue by computing prefix sum
		// 	const prescan_result prescan = block_prefix_sum(valid);
		// 	volatile __shared__ long long int base_offset[1];

		// 	// Obtain base enqueue offset and share it to whole block.
		// 	if(threadIdx.x == 0)
        //     {
		// 		base_offset[0] = atomicAdd(out_queue_count,prescan.total);
        //         // long long int old_value = *out_queue_count;
        //         // *out_queue_count += prescan.total;
        //         // base_offset[0] = old_value;
        //         // // base_offset[0] = *out_queue_count+prescan.total;
        //     }
		// 	__syncthreads();
		// 	// Write vertex to the out queue.
		// 	if (valid == 1)
		// 		out_queue[base_offset[0]+prescan.offset] = neighbor;

		// 	r_gather += BLOCK_SIZE;
		// 	block_progress+= BLOCK_SIZE;
		// 	__syncthreads();
		// }
	}
}

__device__ void fine_gather(long long int *device_col_idx, long long int row_offset_start, 
                            long long int row_offset_end, long long int *device_distance, 
                            long long int iteration, long long int *device_out_queue, long long int *device_out_queue_size, const long long int node)
{
    // get scatter offset and total with prefix sum
    
    prescan_result rank = block_prefix_sum(row_offset_end-row_offset_start);
    // prlong long intf("hi");
    // prlong long intf("real total %d\n", row_offset_end-row_offset_start);
    // prlong long intf("offset %d\n", prescan_offset);
    // prlong long intf("total %d\n", prescan_total);

    volatile __shared__ long long int comm[BLOCK_SIZE];

    long long int cta_progress = 0;

    while((rank.total - cta_progress) > 0)
    {
        // prlong long intf("start %d\n", row_offset_start);

        // All threads pack shared memory
        // long long int orig_row_start = row_offset_start;
        while ((rank.offset < cta_progress + BLOCK_SIZE) && (row_offset_start < row_offset_end))
        {
            // add index to shared memory
            comm[rank.offset - cta_progress] = row_offset_start;
            rank.offset++;
            row_offset_start++;
        }

        // if (row_offset_start > 0)
        // {
        //     prlong long intf("start 2 %d\n", row_offset_start);
        // }

        __syncthreads();

        // each thread gets a neighbor to add to queue
        long long int neighbor;
        long long int valid = 0;

        if (threadIdx.x < (rank.total - cta_progress))
        {
            // make sure only add neighbor if it polong long ints to something
            neighbor = device_col_idx[comm[threadIdx.x]];
            // prlong long intf("node %d neighbor %d and end %d\n", node, neighbor, row_offset_end);

            if ((device_distance[neighbor] == -1))
            {
                // prlong long intf("node %d neighbor %d and start %d and end %d\n", node, neighbor, orig_row_start, row_offset_end);
                valid = 1;
                // prlong long intf("neighbor %d\n", neighbor);
                device_distance[neighbor] = iteration + 1;
            }
        }

        __syncthreads();

        // each thread now adds neighbor to queue with index determined by new prescan offset depending on if there is a neighbor
		const prescan_result prescan = block_prefix_sum(valid);
		volatile __shared__ long long int base_offset[1];

			if(threadIdx.x == 0)
            {
				// base_offset[0] = atomicAdd(device_out_queue_size,prescan.total);
                long long int old_value = *device_out_queue_size;
                *device_out_queue_size += prescan.total;
                base_offset[0] = old_value;
                // base_offset[0] = *device_out_queue_size+prescan.total;
            }

        __syncthreads();

		const long long int queue_index = base_offset[0] + prescan.offset;

        if (valid == 1)
        {
            device_out_queue[queue_index] = neighbor;
        }

        cta_progress += BLOCK_SIZE;

        __syncthreads();
    }
}

__global__
void expand_contract_kernel(long long int *device_col_idx, long long int *device_row_offset, 
                            long long int num_nodes, long long int *device_in_queue, 
                            const long long int device_in_queue_size, long long int *device_out_queue_size, 
                            long long int *device_distance, long long int iteration, long long int *device_out_queue)
{
    long long int th_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    do
    {
        // get node from queue
        long long int cur_node = th_id < device_in_queue_size ? device_in_queue[th_id] : -1;

        //warp culling and history culling
        volatile __shared__ long long int scratch[WARPS][HASH_RANGE];
		// cur_node = warp_cull(scratch, cur_node);

        long long int row_offset_start = cur_node < 0 ? 0 : device_row_offset[cur_node];
        long long int row_offset_end = cur_node < 0 ? 0 : device_row_offset[cur_node+1];

        const bool big_list = (row_offset_end - row_offset_start) >= BLOCK_SIZE;

        // if (big_list)
        // {
        //     prlong long intf("size %d\n", (row_offset_end - row_offset_start));
        // }

        block_gather(device_col_idx, device_distance, iteration, device_out_queue, device_out_queue_size, row_offset_start, big_list ? row_offset_end : row_offset_start);
        fine_gather(device_col_idx, row_offset_start,  big_list ? row_offset_start : row_offset_end, device_distance, iteration, device_out_queue, device_out_queue_size, cur_node);

        // iterate th_id to make sure entire queue is processed and while loop is exited
        th_id += gridDim.x*blockDim.x;
        
    }
    // sync threads in block then perform
    // returns 1 if any thread meets condition
    while(__syncthreads_or(th_id < device_in_queue_size)); 
}

__host__
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
    cudaMemcpy(device_distance, host_distance, graph.num_nodes * sizeof(long long int), cudaMemcpyHostToDevice);
    *host_cur_queue_size = *host_cur_queue_size + 1;

    host_queue[0] = source;

    // copy host to device queue
    cudaMemcpy(device_in_queue, host_queue, graph.num_nodes * sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_out_queue_size, host_cur_queue_size, sizeof(long long int), cudaMemcpyHostToDevice);

    iteration = 0;

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        cudaMemset(device_out_queue_size,0,sizeof(long long int));
        
        const long long int num_of_blocks = div_up(*host_cur_queue_size, BLOCK_SIZE);

        // std::cout << iteration << " " << *host_cur_queue_size << std::endl;
        // std::cout << num_of_blocks << std::endl;

        expand_contract_kernel<<<num_of_blocks, BLOCK_SIZE>>>(device_col_idx, device_row_offset, 
                                                graph.num_nodes, device_in_queue, 
                                                *host_cur_queue_size, device_out_queue_size, 
                                                device_distance, iteration, device_out_queue);
        cudaDeviceSynchronize();

        // copy device queue to host
        cudaMemcpy(host_cur_queue_size, device_out_queue_size, sizeof(long long int), cudaMemcpyDeviceToHost);
        std::swap(device_in_queue, device_out_queue);

        iteration++;
    }

    // copy device distance to host
    cudaMemcpy(host_distance, device_distance, graph.num_nodes * sizeof(long long int), cudaMemcpyDeviceToHost);

    // host_distance[source] = 0;

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);
}

__host__
void gpuBFS::init_graph_for_device(csr &graph)
{
    cudaMalloc(&device_col_idx, graph.num_edges * sizeof(long long int));
    cudaMalloc(&device_row_offset, (graph.num_nodes+1) * sizeof(long long int));

    cudaMemcpy(device_col_idx, graph.col_idx, graph.num_edges * sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(long long int), cudaMemcpyHostToDevice);
}

__host__
void gpuBFS::init_queue(csr &graph)
{
    // allocate host memory
    host_queue = (long long int *)malloc(graph.num_nodes * sizeof(long long int));
    host_cur_queue_size = (long long int *)malloc(sizeof(long long int));
    *host_cur_queue_size = 0;

    // allocate device memory
    cudaMalloc(&device_in_queue, graph.num_nodes * sizeof(long long int));
    cudaMalloc(&device_out_queue, graph.num_nodes * sizeof(long long int));
    cudaMalloc(&device_out_queue_size, sizeof(long long int));
}


__host__
void gpuBFS::init_distance(csr &graph)
{
    // allocate host memory
    host_distance = (long long int *)malloc(graph.num_nodes * sizeof(long long int));

    // allocate device memory
    cudaMalloc(&device_distance, graph.num_nodes * sizeof(long long int));

    // copy memory from host to device
    cudaMemcpy(device_distance, host_distance, graph.num_nodes * sizeof(long long int), cudaMemcpyHostToDevice);

    // run kernel to inialize distance
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(device_distance, graph.num_nodes);

    // copy back
    cudaMemcpy(host_distance, device_distance, graph.num_nodes * sizeof(long long int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}

__host__
gpuBFS::~gpuBFS()
{
    free(host_distance);
    free(host_queue);
    free(host_cur_queue_size);

    cudaFree(device_distance);
    cudaFree(device_in_queue);
    cudaFree(device_out_queue_size);
}