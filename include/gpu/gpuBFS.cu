#include "gpuBFS.cuh"

const int HASH_RANGE = 128;
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 32;
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

__global__
void init_distance_kernel(int *device_distance, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        device_distance[i] = -1;
    }
}

 __device__ int warp_cull(volatile int scratch[2][HASH_RANGE], const int v)
{
	//unsigned int active = __ballot_sync(FULL_MASK, v >= 0);
	//if( v == -1) return v;
	const int hash = v & (HASH_RANGE-1);
	const int warp_id = threadIdx.x / WARP_SIZE;
	if(v >= 0)
		scratch[warp_id][hash]= v;
	__syncwarp();
	const int retrieved = v >= 0 ? scratch[warp_id][hash] : v;
	__syncwarp();
	unsigned int active = __ballot_sync(0xffffffff, retrieved == v);
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

//  __device__ prescan_result block_prefix_sum(const int val)
// {    
// 	volatile __shared__ int sums[WARPS];
// 	int value = val;

// 	const int lane_id = threadIdx.x % WARP_SIZE;
// 	const int warp_id = threadIdx.x / WARP_SIZE;

// 	// Warp-wide prefix sums.
// #pragma unroll
// 	for(int i = 1; i <= WARP_SIZE; i <<= 1)
// 	{
// 		const int n = __shfl_up_sync(0xffffffff, value, i, WARP_SIZE);
// 		if (lane_id >= i)
// 			value += n;
// 	}

// 	// Write warp total to shared array.
// 	if (lane_id == WARP_SIZE - 1)
// 	{
// 		sums[warp_id] = value;
// 	}

// 	__syncthreads();

// 	// Prefix sum of warp sums.
// 	if (warp_id == 0 && lane_id < WARPS)
// 	{
// 		int warp_sum = sums[lane_id];
// 		const unsigned int mask = (1 << (WARPS)) - 1;
// #pragma unroll
// 		for (int i = 1; i <= WARPS; i <<= 1)
// 		{
// 			const int n = __shfl_up_sync(mask, warp_sum, i, WARPS);
// 			if (lane_id >= i)
// 				warp_sum += n;
// 		}

// 		sums[lane_id] = warp_sum;
// 	}

// 	__syncthreads();

// 	// Add total sum of previous WARPS to current element.
// 	if (warp_id > 0)
// 	{
// 		const int block_sum = sums[warp_id-1];
// 		value += block_sum;
// 	}

// 	prescan_result result;
// 	result.offset = value - val;
// 	result.total = sums[WARPS-1];
// 	return result; 
// }

__device__ prescan_result block_prefix_sum(const int val) {
    __shared__ int block_data[BLOCK_SIZE]; // Assuming maximum block size of 1024 threads
    prescan_result result;

    int thid = threadIdx.x;
    block_data[thid] = val; // Assign value to block_data
    

    __syncthreads();

    int os = 1;

    // Compute prefix sum
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = os * (2 * thid+1) - 1;
            int bi = os * (2 * thid+2) - 1;
            block_data[bi] += block_data[ai];
        }

        os *= 2;
    }

    if (thid == 0) { 
        // result.block_sum = block_data[blockDim.x - 1];
        block_data[blockDim.x - 1] = 0; // Clear the last element
    }

    for (int d = 1; d < blockDim.x; d *= 2) {

        os /= 2;

        __syncthreads();
        if (thid < d) {
            int ai = os * (2 * thid+1) - 1;
            int bi = os * (2 * thid+2) - 1;
            int t = block_data[ai];

            block_data[ai] = block_data[bi];
            block_data[bi] += t;
        }
    }

    __syncthreads();
    
    result.offset = block_data[thid];
    // if (thid == blockDim.x - 1) {
    result.total = block_data[blockDim.x - 1];
    // }

    // if (thid == 0) 
    // {
    //     printf("%d \n", block_data[thid]);
    // }

    return result;
}

 __device__ void block_gather(const int* const column_index, int* const distance, 
                                const int iteration, int * const out_queue, 
                                int* const out_queue_count,int r, int r_end)
{
	volatile __shared__ int comm[3];
    // int orig_row_start = r;
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
		int r_gather = comm[1] + threadIdx.x;
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
			const prescan_result prescan = block_prefix_sum(valid);
			volatile __shared__ int base_offset[1];

			// Obtain base enqueue offset and share it to whole block.
			if(threadIdx.x == 0)
				base_offset[0] = atomicAdd(out_queue_count,prescan.total);
			__syncthreads();
			// Write vertex to the out queue.
			if (valid == 1)
				out_queue[base_offset[0]+prescan.offset] = neighbor;

			r_gather += BLOCK_SIZE;
			block_progress+= BLOCK_SIZE;
			__syncthreads();
		}
	}
}

__device__ void fine_gather(int *device_col_idx, int row_offset_start, 
                            int row_offset_end, int *device_distance, 
                            int iteration, int *device_out_queue, int *device_out_queue_size, const int node)
{
    // get scatter offset and total with prefix sum
    
    prescan_result rank = block_prefix_sum(row_offset_end-row_offset_start);
    // printf("hi");
    // printf("real total %d\n", row_offset_end-row_offset_start);
    // printf("offset %d\n", prescan_offset);
    // printf("total %d\n", prescan_total);

    volatile __shared__ int comm[BLOCK_SIZE];

    int cta_progress = 0;

    while((rank.total - cta_progress) > 0)
    {
        // printf("start %d\n", row_offset_start);

        // All threads pack shared memory
        // int orig_row_start = row_offset_start;
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

        __syncthreads();

        // each thread gets a neighbor to add to queue
        int neighbor;
        int valid = 0;

        if (threadIdx.x < (rank.total - cta_progress))
        {
            // make sure only add neighbor if it points to something
            neighbor = device_col_idx[comm[threadIdx.x]];
            // printf("node %d neighbor %d and end %d\n", node, neighbor, row_offset_end);

            if ((device_distance[neighbor] == -1))
            {
                // printf("node %d neighbor %d and start %d and end %d\n", node, neighbor, orig_row_start, row_offset_end);
                valid = 1;
                // printf("neighbor %d\n", neighbor);
                device_distance[neighbor] = iteration + 1;
            }
        }

        __syncthreads();

        // each thread now adds neighbor to queue with index determined by new prescan offset depending on if there is a neighbor
		const prescan_result prescan = block_prefix_sum(valid);
		volatile __shared__ int base_offset[1];

        if (threadIdx.x == 0)
        {
            base_offset[0] = atomicAdd(device_out_queue_size, prescan.total);
        }

        __syncthreads();

		const int queue_index = base_offset[0] + prescan.offset;

        if (valid == 1)
        {
            device_out_queue[queue_index] = neighbor;
        }

        cta_progress += BLOCK_SIZE;

        __syncthreads();
    }
}

__global__
void expand_contract_kernel(int *device_col_idx, int *device_row_offset, 
                            int num_nodes, int *device_in_queue, 
                            const int device_in_queue_size, int *device_out_queue_size, 
                            int *device_distance, int iteration, int *device_out_queue)
{
    int th_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    do
    {
        // get node from queue
        int cur_node = th_id < device_in_queue_size ? device_in_queue[th_id] : -1;

        //warp culling and history culling
        volatile __shared__ int scratch[WARPS][HASH_RANGE];
		cur_node = warp_cull(scratch, cur_node);

        int row_offset_start = cur_node < 0 ? 0 : device_row_offset[cur_node];
        int row_offset_end = cur_node < 0 ? 0 : device_row_offset[cur_node+1];

        const bool big_list = (row_offset_end - row_offset_start) >= BLOCK_SIZE;

        // if (cur_node == 31)
        // {
        //     printf("%d\n", (row_offset_end - row_offset_start) >= BLOCK_SIZE);
        // }
        
        block_gather(device_col_idx, device_distance, iteration, device_out_queue, device_out_queue_size, row_offset_start, big_list ? row_offset_end : row_offset_start);
        fine_gather(device_col_idx, row_offset_start,  big_list ? row_offset_start : row_offset_end, device_distance, iteration, device_out_queue, device_out_queue_size, cur_node);
        // fine_gather(device_col_idx, device_distance, iteration, device_out_queue, device_out_queue_size, row_offset_start, big_list ? row_offset_start : row_offset_end);

        // iterate th_id to make sure entire queue is processed
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
    *host_cur_queue_size = *host_cur_queue_size + 1;

    host_queue[0] = source;

    // copy host to device queue
    cudaMemcpy(device_in_queue, host_queue, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_out_queue_size, host_cur_queue_size, sizeof(int), cudaMemcpyHostToDevice);

    int iteration = 0;

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaEventRecord(gpu_start);

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        cudaMemset(device_out_queue_size,0,sizeof(int));
        
        const int num_of_blocks = div_up(*host_cur_queue_size, BLOCK_SIZE);

        expand_contract_kernel<<<num_of_blocks, BLOCK_SIZE>>>(device_col_idx, device_row_offset, 
                                                graph.num_nodes, device_in_queue, 
                                                *host_cur_queue_size, device_out_queue_size, 
                                                device_distance, iteration, device_out_queue);
        cudaDeviceSynchronize();

        // copy device queue to host
        cudaMemcpy(host_cur_queue_size, device_out_queue_size, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(device_in_queue, device_out_queue);

        iteration++;
    }

    // copy device distance to host
    cudaMemcpy(host_distance, device_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    host_distance[source] = 0;

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);
}

__host__
void gpuBFS::init_graph_for_device(csr &graph)
{
    cudaMalloc(&device_col_idx, graph.num_edges * sizeof(int));
    cudaMalloc(&device_row_offset, (graph.num_nodes+1) * sizeof(int));

    cudaMemcpy(device_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice);
}

__host__
void gpuBFS::init_queue(csr &graph)
{
    // allocate host memory
    host_queue = (int *)malloc(graph.num_nodes * sizeof(int));
    host_cur_queue_size = (int *)malloc(sizeof(int));
    *host_cur_queue_size = 0;

    // allocate device memory
    cudaMalloc(&device_in_queue, graph.num_nodes * sizeof(int));
    cudaMalloc(&device_out_queue, graph.num_nodes * sizeof(int));
    cudaMalloc(&device_out_queue_size, sizeof(int));
}


__host__
void gpuBFS::init_distance(csr &graph)
{
    // allocate host memory
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    // allocate device memory
    cudaMalloc(&device_distance, graph.num_nodes * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(device_distance, host_distance, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel to inialize distance
    dim3 block(BLOCK_SIZE, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(device_distance, graph.num_nodes);

    // copy back
    cudaMemcpy(host_distance, device_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

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