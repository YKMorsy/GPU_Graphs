#include "gpuBFS.cuh"

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

 __device__ prescan_result block_prefix_sum(const int val)
{
    const int block_size = 128;
    const int warp_size = 32;
    const int warps = block_size/warp_size;

    
	volatile __shared__ int sums[warps];
	int value = val;

	const int lane_id = threadIdx.x % warp_size;
	const int warp_id = threadIdx.x / warp_size;

	// Warp-wide prefix sums.
#pragma unroll
	for(int i = 1; i <= warp_size; i <<= 1)
	{
		const int n = __shfl_up_sync(0xffffffff, value, i, warp_size);
		if (lane_id >= i)
			value += n;
	}

	// Write warp total to shared array.
	if (lane_id == warp_size - 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	// Prefix sum of warp sums.
	if (warp_id == 0 && lane_id < warps)
	{
		int warp_sum = sums[lane_id];
		const unsigned int mask = (1 << (warps)) - 1;
#pragma unroll
		for (int i = 1; i <= warps; i <<= 1)
		{
			const int n = __shfl_up_sync(mask, warp_sum, i, warps);
			if (lane_id >= i)
				warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// Add total sum of previous warps to current element.
	if (warp_id > 0)
	{
		const int block_sum = sums[warp_id-1];
		value += block_sum;
	}

	prescan_result result;
	result.offset = value - val;
	result.total = sums[warps-1];
	return result; 
}

__device__ void fine_gather(int *device_col_idx, int row_offset_start, 
                            int row_offset_end, int *device_distance, 
                            int iteration, int *device_out_queue, int *device_out_queue_size, const int node)
{
    // get scatter offset and total with prefix sum
    prescan_result res = block_prefix_sum(row_offset_end-row_offset_start);
    int prescan_offset = res.offset;
    int prescan_total = res.total;

    // printf("real total %d\n", row_offset_end-row_offset_start);
    // printf("offset %d\n", prescan_offset);
    // printf("total %d\n", prescan_total);

    volatile __shared__ int comm[128];

    int cta_progress = 0;

    while((prescan_total - cta_progress) > 0)
    {
        // printf("start %d\n", row_offset_start);

        // All threads pack shared memory
        int orig_row_start = row_offset_start;
        while ((prescan_offset < cta_progress + 128) && (row_offset_start < row_offset_end))
        {
            // add index to shared memory
            comm[prescan_offset - cta_progress] = row_offset_start;
            prescan_offset++;
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

        if (threadIdx.x < (prescan_total - cta_progress))
        {
            // make sure only add neighbor if it points to something
            neighbor = device_col_idx[comm[threadIdx.x]];
            // printf("node %d neighbor %d and end %d\n", node, neighbor, row_offset_end);

            if ((device_distance[neighbor] == -1) && (orig_row_start != -1))
            {
                // printf("node %d neighbor %d and start %d and end %d\n", node, neighbor, orig_row_start, row_offset_end);
                valid = 1;
                // printf("neighbor %d\n", neighbor);
                device_distance[neighbor] = iteration + 1;
            }
        }

        // printf("thread and limit %d %d\n", threadIdx.x, (prescan_total - cta_progress));

        // if (threadIdx.x < 2)
        // {
        //     printf("neighbor %d\n", neighbor);
        // }
        __syncthreads();

        // each thread now adds neighbor to queue with index determined by new prescan offset depending on if there is a neighbor
        res = block_prefix_sum(valid);
        prescan_offset = res.offset;
        prescan_total = res.total;

        // printf("offset %d\n", prescan_offset);

        volatile __shared__ int base_offset[1];

        if (threadIdx.x == 0)
        {
            base_offset[0] = atomicAdd(device_out_queue_size, prescan_total);
            // printf("atomic add %d\n", *device_out_queue_size);
            // printf("prescan total %d\n", prescan_total);
            // printf("offset %d\n", base_offset[0]);
        }

        __syncthreads();

        int queue_index = base_offset[0] + prescan_offset;

        // printf("q idx %d\n", queue_index);
        
        if (valid == 1)
        {
            // printf("%d\n", neighbor);
            device_out_queue[queue_index] = neighbor;
        }

        cta_progress += 128;

        __syncthreads();
    }
}

__global__
void expand_contract_kernel(int *device_col_idx, int *device_row_offset, 
                            int num_nodes, int *device_in_queue, 
                            const int device_in_queue_size, int *device_out_queue_size, 
                            int *device_distance, int iteration, int *device_out_queue)
{
    // printf("HI\n");
    // printf("%d\n", device_in_queue_size);
    
    int th_id = blockIdx.x * blockDim.x + threadIdx.x;
    

    // if (th_id < device_in_queue_size)
    do
    {
        // printf("HI\n");


        // get node from queue
        int cur_node;
        int row_offset_start = 0;
        int row_offset_end = 0;
        if (th_id < device_in_queue_size)
        {
            cur_node = device_in_queue[th_id];
            // printf("current node %d\n", cur_node);
            row_offset_start = device_row_offset[cur_node];
            if (row_offset_start != -1)
            {
                for (int i = cur_node+1; i < num_nodes + 1; i++)
                {
                    if (device_row_offset[i] != -1)
                    {
                        row_offset_end = device_row_offset[i];
                        break;
                    }
                }
            }

            // printf("r_start %d and r_end %d\n", row_offset_start, row_offset_end);
        }
        else
        {
            cur_node = -1;
        }

        //warp culling and history culling

        // printf("%d\n", *device_cur_queue_size);

        // load row range (neighbor idx) - check if cur_node is part of queue       
        // int row_offset_start = device_row_offset[cur_node];
        // int row_offset_end =  device_row_offset[cur_node+1];

        // printf("%d\n", row_offset_start);

        fine_gather(device_col_idx, row_offset_start, row_offset_end, device_distance, iteration, device_out_queue, device_out_queue_size, cur_node);

        // temp            
        // *device_cur_queue_size = *device_cur_queue_size - 1;
        // printf("%d\n", *device_cur_queue_size);

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

    // loop until frontier is empty
    while (*host_cur_queue_size > 0)
    {
        cudaMemset(device_out_queue_size,0,sizeof(int));

        // neighbor adding kernel
        dim3 block(128, 1);
        dim3 grid((*host_cur_queue_size+block.x-1)/block.x, 1);

        // int *temp_host_cur_queue_size;
        // temp_host_cur_queue_size = (int *)malloc(sizeof(int));
        // cudaMemcpy(temp_host_cur_queue_size, device_out_queue_size, sizeof(int), cudaMemcpyDeviceToHost);
        // std::cout << "size pre " << *temp_host_cur_queue_size << std::endl;

        // std::cout << "size " << *host_cur_queue_size << std::endl;
        
        expand_contract_kernel<<<grid, block>>>(device_col_idx, device_row_offset, 
                                                graph.num_nodes, device_in_queue, 
                                                *host_cur_queue_size, device_out_queue_size, 
                                                device_distance, iteration, device_out_queue);
        cudaDeviceSynchronize();

        // break;

        // copy device queue to host
        // cudaMemcpy(host_queue, device_in_queue, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_cur_queue_size, device_out_queue_size, sizeof(int), cudaMemcpyDeviceToHost);
        std::swap(device_in_queue, device_out_queue);

        // std::cout << grid.x << std::endl;
        // *host_cur_queue_size = *host_cur_queue_size - 1;

        iteration++;

        // std::cout << "size " << *host_cur_queue_size << std::endl;
    }

    // copy device distance to host
    cudaMemcpy(host_distance, device_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    host_distance[source] = 0;
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

    // run kernel to inialize kernel
    dim3 block(128, 1);
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