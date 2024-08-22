#include "gpuBFS_kernels.cuh"
#include <stdio.h>


__global__
void init_distance_kernel(const int* start_node, const int* end_node, int num_nodes, int *device_distance, int source)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;\
    
    if (i < num_nodes)
    {
        device_distance[i] = INF;
    }
}

__global__ 
void linear_bfs(const int total_nodes, const int starting_col_idx_pre_pe, const int* start_node, const int* end_node, 
                const int* row_offset, const int* column_index, int* distance, const int iteration, const int* in_queue, 
                const uint32_t in_queue_count, int* out_queue, uint32_t* out_queue_count, int* edges_traversed)
{
    // Compute index of corresponding vertex in the queue.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	do
    {

		if(global_tid >= in_queue_count) continue;

		int total_pe = nvshmem_n_pes();

		// Get node from the queue
		int v = in_queue[global_tid] - *start_node;

		// Only proceed if this node hasn't been visited yet (use atomicCAS to update distance)
		if (atomicCAS(&distance[v], INF, iteration) == INF) 
		{
			atomicAdd(edges_traversed, 1);

			// Get neighbors range from offset
			int r = row_offset[v] - starting_col_idx_pre_pe;
			int r_end = row_offset[v + 1] - starting_col_idx_pre_pe;

			// Each thread visits its neighbors
			for(int offset = r; offset < r_end; ++offset)
			{
				// Get neighbor
				int neighbor = column_index[offset];

				if (neighbor >= *start_node && neighbor <= *end_node)
				{
					uint32_t ind = atomicAdd(out_queue_count, 1);
					out_queue[ind] = neighbor;
				}
				else
				{
					int target_pe, start_node_target;
					int total_pe = nvshmem_n_pes();
					int base_nodes_per_pe = total_nodes / total_pe;
					int remainder_nodes = total_nodes % total_pe;

					if (neighbor < (base_nodes_per_pe + 1) * remainder_nodes)
					{
						target_pe = neighbor / (base_nodes_per_pe + 1);
						start_node_target = target_pe * (base_nodes_per_pe + 1);
					}
					else
					{
						target_pe = remainder_nodes + (neighbor - remainder_nodes * (base_nodes_per_pe + 1)) / base_nodes_per_pe;
						start_node_target = remainder_nodes * (base_nodes_per_pe + 1) + (target_pe - remainder_nodes) * base_nodes_per_pe;
					}
					uint32_t ind = nvshmem_size_atomic_fetch_add(reinterpret_cast<size_t*>(out_queue_count), 1, target_pe);
					nvshmem_int_p(&out_queue[ind], neighbor, target_pe);
				}
			}
		}

		global_tid += gridDim.x * blockDim.x;

	}while(__syncthreads_or(global_tid < in_queue_count)); 
}

// __global__ 
// void linear_bfs(const int total_nodes, const int starting_col_idx_pre_pe, const int* start_node, const int* end_node, 
//                 const int* row_offset, const int* column_index, int* distance, const int iteration, const int* in_queue, 
//                 const int in_queue_count, int* out_queue, int* out_queue_count, int* edges_traversed)
// {
//     // Compute index of corresponding vertex in the queue.
//     int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

//     do
//     {
//         // Check if thread has work to do
//         if(global_tid < in_queue_count)
//         {
//             int total_pe = nvshmem_n_pes();

//             // Compute base nodes per PE and remainder
//             int base_nodes_per_pe = total_nodes / total_pe;
//             int remainder_nodes = total_nodes % total_pe;

//             // Get node from the queue
//             int v = in_queue[global_tid] - *start_node;

//             // Only proceed if this node hasn't been visited yet (use atomicCAS to update distance)
//             if (distance[v] == INF) 
//             {
//                 atomicAdd(edges_traversed, 1);
// 				distance[v] = iteration;

//                 // Get neighbors range from offset
//                 int r = row_offset[v] - starting_col_idx_pre_pe;
//                 int r_end = row_offset[v + 1] - starting_col_idx_pre_pe;

//                 // Each thread visits its neighbors
//                 for(int offset = r; offset < r_end; ++offset)
//                 {
//                     // Get neighbor
//                     int neighbor = column_index[offset];
					
//                     // Add neighbor to the appropriate queue
//                     if (neighbor >= *start_node && neighbor <= *end_node)
//                     {
//                         int ind = atomicAdd(out_queue_count, 1);
//                         out_queue[ind] = neighbor;
//                     }
//                     else
//                     {
// 						// Determine which PE is responsible for the neighbor
// 						int target_pe, start_node_target;

// 						if (neighbor < (base_nodes_per_pe + 1) * remainder_nodes)
// 						{
// 							target_pe = neighbor / (base_nodes_per_pe + 1);
// 							start_node_target = target_pe * (base_nodes_per_pe + 1);
// 						}
// 						else
// 						{
// 							target_pe = remainder_nodes + (neighbor - remainder_nodes * (base_nodes_per_pe + 1)) / base_nodes_per_pe;
// 							start_node_target = remainder_nodes * (base_nodes_per_pe + 1) + (target_pe - remainder_nodes) * base_nodes_per_pe;
// 						}
//                         int ind = nvshmem_int_atomic_fetch_add(out_queue_count, 1, target_pe);
//                         nvshmem_int_p(&out_queue[ind], neighbor, target_pe);
//                     }
//                 }
//             }
//         }

//         // Increment global thread ID to move to the next batch of work
//         global_tid += gridDim.x * blockDim.x;

//         // Synchronize all threads to ensure that they all complete their work before checking the next iteration
//         __syncthreads();
//     } 
//     // Continue until all threads have completed their work
//     while(__syncthreads_or(global_tid < in_queue_count)); 
// }



// __global__ 
// void linear_bfs(const int total_nodes, const int starting_col_idx_pre_pe, const int* start_node, const int* end_node, 
// 				const int* row_offset, const int* column_index, int* distance, const int iteration, const int* in_queue, 
// 				const int in_queue_count, int* out_queue, int* out_queue_count, int* edges_traversed)
// {
// 	// Compute index of corresponding vertex in the queue.
// 	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

// 	do
// 	{
// 		// skip thread if nothing for it to process
// 		if(global_tid >= in_queue_count) continue;

// 		int total_pe = nvshmem_n_pes();

//         // Compute base nodes per PE and remainder
//         int base_nodes_per_pe = total_nodes / total_pe;
//         int remainder_nodes = total_nodes % total_pe;

// 		// Get node from the queue
// 		int v = in_queue[global_tid]-*start_node;

// 		// get neighbors range from offset
// 		int r = row_offset[v] - starting_col_idx_pre_pe;
// 		int r_end = row_offset[v+1] - starting_col_idx_pre_pe;


// 		// printf("cur pe: %d and cur node abs: %d and cur node: %d and start r: %d and end r: %d\n", nvshmem_my_pe(), in_queue[global_tid], v, r, r_end);

// 		// each visits neighbors
// 		for(int offset = r; offset < r_end; offset++)
// 		{
// 			// get neighbor
// 			int neighbor = column_index[offset];

// 			// if current pe is responsible for neighbor
// 			if (neighbor >= *start_node && neighbor <= *end_node)
// 			{
// 				// printf("cur pe: %d and target pe: %d for cur node: %d for neighbor: %d and start_node_target: %d with value: %d\n", nvshmem_my_pe(), nvshmem_my_pe(), v, neighbor, *start_node, distance[neighbor-*start_node]);

// 				// add neighbor if not traversed
// 				if(distance[neighbor-*start_node] == INF)
// 				{
// 					atomicAdd(edges_traversed, 1);
// 					distance[neighbor-*start_node]=iteration+1;
// 					// Enqueue vertex.
// 					int ind = atomicAdd(out_queue_count,1);
// 					out_queue[ind]=neighbor;
// 				}
// 			}
// 			// another pe is responsible for neighbor
// 			else
// 			{
// 				int target_pe, start_node_target;

// 				// Handle the first `remainder_nodes` PEs getting an extra node
// 				if (neighbor < (base_nodes_per_pe + 1) * remainder_nodes)
// 				{
// 					target_pe = neighbor / (base_nodes_per_pe + 1);
// 					start_node_target = target_pe * (base_nodes_per_pe + 1);
// 				}
// 				else
// 				{
// 					target_pe = remainder_nodes + (neighbor - remainder_nodes * (base_nodes_per_pe + 1)) / base_nodes_per_pe;
// 					start_node_target = remainder_nodes * (base_nodes_per_pe + 1) + (target_pe - remainder_nodes) * base_nodes_per_pe;
// 				}
			
// 				// printf("cur pe: %d and target pe: %d for cur node: %d for neighbor: %d and start_node_target: %d with value: %d\n", nvshmem_my_pe(), target_pe, v, neighbor, start_node_target, nvshmem_int_g(&distance[neighbor-start_node_target], target_pe));
// 				// add neighbor if not traversed
// 				if(nvshmem_int_g(&distance[neighbor-start_node_target], target_pe) == INF)
// 				{

// 					// printf("cur pe: %d and target pe: %d for cur node: %d for neighbor: %d and start_node_target: %d with value: %d\n", nvshmem_my_pe(), v, target_pe, neighbor, start_node_target, nvshmem_int_g(&distance[neighbor-start_node_target], target_pe));
// 					// update distance vector of target node
// 					nvshmem_int_atomic_add(edges_traversed, 1, target_pe);
// 					nvshmem_int_p(&distance[neighbor-start_node_target], iteration+1, target_pe);
// 					// add to queue of target node
// 					int ind = nvshmem_int_atomic_fetch_add(out_queue_count, 1, target_pe);
// 					nvshmem_int_p(&out_queue[ind], neighbor, target_pe);
// 					// printf("out_queue of target now contains: %d in idx: %d\n", nvshmem_int_g(&out_queue[ind], target_pe), ind);
// 				}
// 			}
// 		}
// 		global_tid += gridDim.x*blockDim.x;
// 	} 
// 	// ensure atleast one thread has something to process
// 	while(__syncthreads_or(global_tid < in_queue_count)); 
// }

 __device__ int warp_cull(volatile int scratch[WARPS][HASH_RANGE], const int v)
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
	unsigned int active = __ballot_sync(FULL_MASK, retrieved == v);
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

__device__ 
prescan_result block_prefix_sum(const int val)
{
	// Heavily inspired/copied from sample "shfl_scan" provided by NVIDIA.
	// Block-wide prefix sum using shfl intrinsic.
	volatile __shared__ int sums[WARPS];
	int value = val;

	const int lane_id = threadIdx.x % WARP_SIZE;
	const int warp_id = threadIdx.x / WARP_SIZE;

	// Warp-wide prefix sums.
#pragma unroll
	for(int i = 1; i <= WARP_SIZE; i <<= 1)
	{
		const int n = __shfl_up_sync(FULL_MASK, value, i, WARP_SIZE);
		if (lane_id >= i)
			value += n;
	}

	// Write warp total to shared array.
	if (lane_id == WARP_SIZE- 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	// Prefix sum of warp sums.
	if (warp_id == 0 && lane_id < WARPS)
	{
		int warp_sum = sums[lane_id];
		const unsigned int mask = (1 << (WARPS)) - 1;
#pragma unroll
		for (int i = 1; i <= WARPS; i <<= 1)
		{
			const int n = __shfl_up_sync(mask, warp_sum, i, WARPS);
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
	// Subtract value given by thread to get exclusive prefix sum.
	result.offset = value - val;
	// Get total sum.
	result.total = sums[WARPS-1];
	return result; 
}

__device__ 
void block_gather(const int* column_index, int* distance, const int iteration, int* out_queue, int* out_queue_count, int r, int r_end)
{
	volatile __shared__ int comm[3];
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

		// entire block gets the neighbors of one thread's nodes
		while((total - block_progress) > 0)
		{
			int neighbor = -1;
			bool is_valid = false;
			if (r_gather < r_gather_end)
			{
				neighbor = column_index[r_gather];
				// Look up status of current neighbor.
				if(distance[neighbor] == INF)
				{
					is_valid = true;
					// Update label.
					distance[neighbor] = iteration + 1;
				}
			}
			// Obtain offset in queue by computing prefix sum
			const prescan_result prescan = block_prefix_sum(is_valid?1:0);
			volatile __shared__ int base_offset[1];

			// Obtain base enqueue offset and share it to whole block.
			if(threadIdx.x == 0)
				base_offset[0] = atomicAdd(out_queue_count,prescan.total);
			__syncthreads();
			// Write vertex to the out queue.
			if (is_valid)
				out_queue[base_offset[0]+prescan.offset] = neighbor;

			r_gather += 1024;
			block_progress+= 1024;
			__syncthreads();
		}
	}
}


__device__ 
void fine_gather(const int* column_index, int* distance, const int iteration, int* out_queue, int* out_queue_count, int r, int r_end)
{
	prescan_result rank = block_prefix_sum(r_end-r);

	__shared__ int comm[1024];
	int cta_progress = 0;

	while ((rank.total - cta_progress) > 0)
	{
		// Pack shared array with neighbors from adjacency lists.
		while((rank.offset < cta_progress + 1024) && (r < r_end))
		{
			comm[rank.offset - cta_progress] = r;
			rank.offset++;
			r++;
		}
		__syncthreads();
		// label neighbor distance
		int neighbor;
		bool is_valid = false;
		if (threadIdx.x < (rank.total - cta_progress))
		{
			neighbor = column_index[comm[threadIdx.x]];
			if(distance[neighbor] == INF)
			{
				is_valid = true;
				// Update label
				distance[neighbor] = iteration + 1;
			}
		}
		__syncthreads();

		// add neighbor to enqueue

		// Obtain offset in queue by computing prefix sum.
		const prescan_result prescan = block_prefix_sum(is_valid?1:0);
		volatile __shared__ int base_offset[1];
		// Obtain base enqueue offset
		if(threadIdx.x == 0)
		{
			base_offset[0] = atomicAdd(out_queue_count,prescan.total);
		}
		__syncthreads();
		const int queue_index = base_offset[0] + prescan.offset;
		// Write to queue
		if (is_valid)
		{
			out_queue[queue_index] = neighbor;
		}

		cta_progress += 1024;
		__syncthreads();
	}
}



__global__ 
void expand_contract_bfs(const int num_nodes, const int* row_offset, const int* column_index, int* distance, const int iteration, const int* in_queue, const int in_queue_count, int* out_queue, int* out_queue_count)
{
	// Compute index of corresponding vertex in the queue.
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

	do
	{
		// Get node from the queue
		int v = global_tid < in_queue_count? in_queue[global_tid]:-1;

		// Do local warp-culling.
		volatile __shared__ int scratch[WARPS][HASH_RANGE];
		v = warp_cull(scratch, v);

		// get neighbors range from offset
		int r = row_offset[v];
		int r_end = row_offset[v+1];
		bool big_list = (r_end - r) >= 1024;

		block_gather(column_index, distance, iteration, out_queue, out_queue_count, r, big_list ? r_end : r);
		fine_gather(column_index, distance, iteration, out_queue, out_queue_count, r, big_list ? r : r_end);

		global_tid += gridDim.x*blockDim.x;
	} 
	// ensure atleast one thread has something to process
	while(__syncthreads_or(global_tid < in_queue_count)); 
}
