#include "gpuBFS.cuh"

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
	((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "'" << std::endl;
        cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
        exit(99);
    }
}

__global__
void nextLayer(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                int queueSize, int *d_currentQueue, int *d_distance, int iteration) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];

        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == -1) 
            {
                d_parent[v] = i;
                d_distance[v] = iteration + 1;
            }
        }
    }
}

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                  int queueSize, int *d_currentQueue, int *d_degrees, int *d_distance, int iteration)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];

        int degree = 0;

        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_parent[v] == i && v != cur_node) 
            // if (d_distance[v] == -1) 
            {
                degree++;
            }
        }

        // if (iteration == 4 && degree != 0)
        // {
        //     printf("thid and degree %d %d\n", thid, degree);
        // }
        
        d_degrees[thid] = degree;
    }
}

__global__
void gather(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
            int *d_currentQueue, int *d_nextQueue, int *incrDegrees, int *d_distance, int iteration)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int nextQueuePlace = incrDegrees[thid];

        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];
        // int counter = 0;
        for (int i = row_offset_start; i < row_offset_end; i++)
        {
            int v = d_adjacencyList[i];
            // if (d_distance[v] == -1) {
            if (d_parent[v] == i && v != cur_node) 
            {
                // int nextQueuePlace = sharedIncrement + sum + counter;
                // printf("individiual %d %d\n", thid, nextQueuePlace);
                // d_distance[v] = iteration + 1;
                d_nextQueue[nextQueuePlace] = v;
                nextQueuePlace++;
            }
        }
    }
}

__global__
void init_distance_kernel(int *device_distance, int *device_parent, int size, int source)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        if (i == source)
        {
            device_distance[i] = 0;
            device_parent[i] = -1;
        }
        else
        {
            device_distance[i] = -1;
            device_parent[i] = -1;
        }

    }
}

__global__
void gpu_add_block_sums(int* const d_out,
	const int* const d_in,
	int* const d_block_sums,
	const size_t numElems)
{
	//int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int d_block_sum_val = d_block_sums[blockIdx.x];

	//int d_in_val_0 = 0;
	//int d_in_val_1 = 0;

	// Simple implementation's performance is not significantly (if at all)
	//  better than previous verbose implementation
	int cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	if (cpy_idx < numElems)
	{
		d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
		if (cpy_idx + blockDim.x < numElems)
			d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
	}

	//if (2 * glbl_t_idx < numElems)
	//{
	//	d_out[2 * glbl_t_idx] = d_in[2 * glbl_t_idx] + d_block_sum_val;
	//	if (2 * glbl_t_idx + 1 < numElems)
	//		d_out[2 * glbl_t_idx + 1] = d_in[2 * glbl_t_idx + 1] + d_block_sum_val;
	//}

	//if (2 * glbl_t_idx < numElems)
	//{
	//	d_in_val_0 = d_in[2 * glbl_t_idx];
	//	if (2 * glbl_t_idx + 1 < numElems)
	//		d_in_val_1 = d_in[2 * glbl_t_idx + 1];
	//}
	//else
	//	return;
	//__syncthreads();

	//d_out[2 * glbl_t_idx] = d_in_val_0 + d_block_sum_val;
	//if (2 * glbl_t_idx + 1 < numElems)
	//	d_out[2 * glbl_t_idx + 1] = d_in_val_1 + d_block_sum_val;
}


__global__
void gpu_prescan(int* const d_out,
	const int* const d_in,
	int* const d_block_sums,
	const int len,
	const int shmem_sz,
	const int max_elems_per_block)
{
	// Allocated on invocation
	extern __shared__ int s_out[];

	int thid = threadIdx.x;
	int ai = thid;
	int bi = thid + blockDim.x;

	// Zero out the shared memory
	// Helpful especially when input size is not power of two
	s_out[thid] = 0;
	s_out[thid + blockDim.x] = 0;
	// If CONFLICT_FREE_OFFSET is used, shared memory
	//  must be a few more than 2 * blockDim.x
	if (thid + max_elems_per_block < shmem_sz)
		s_out[thid + max_elems_per_block] = 0;

	__syncthreads();
	
	// Copy d_in to shared memory
	// Note that d_in's elements are scattered into shared memory
	//  in light of avoiding bank conflicts
	int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
	if (cpy_idx < len)
	{
		s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
		if (cpy_idx + blockDim.x < len)
			s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
	}

	// For both upsweep and downsweep:
	// Sequential indices with conflict free padding
	//  Amount of padding = target index / num banks
	//  This "shifts" the target indices by one every multiple
	//   of the num banks
	// offset controls the stride and starting index of 
	//  target elems at every iteration
	// d just controls which threads are active
	// Sweeps are pivoted on the last element of shared memory

	// Upsweep/Reduce step
	int offset = 1;
	for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
	{
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			s_out[bi] += s_out[ai];
		}
		offset <<= 1;
	}

	// Save the total sum on the global block sums array
	// Then clear the last element on the shared memory
	if (thid == 0) 
	{ 
		d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
		s_out[max_elems_per_block - 1 
			+ CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
	}

	// Downsweep step
	for (int d = 1; d < max_elems_per_block; d <<= 1)
	{
		offset >>= 1;
		__syncthreads();

		if (thid < d)
		{
			int ai = offset * ((thid << 1) + 1) - 1;
			int bi = offset * ((thid << 1) + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int temp = s_out[ai];
			s_out[ai] = s_out[bi];
			s_out[bi] += temp;
		}
	}
	__syncthreads();

	// Copy contents of shared memory to global memory
	if (cpy_idx < len)
	{
		d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
		if (cpy_idx + blockDim.x < len)
			d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
	}
}



void sum_scan_blelloch(int* const d_out,
	const int* const d_in,
	const size_t numElems)
{
	// Zero out d_out
	(cudaMemset(d_out, 0, numElems * sizeof(int)));

	// Set up number of threads and blocks
	
	int block_sz = MAX_BLOCK_SZ / 2;
	int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
	//int grid_sz = (int) std::ceil((double) numElems / (double) max_elems_per_block);
	// UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically  
	//  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
	int grid_sz = numElems / max_elems_per_block;
	// Take advantage of the fact that integer division drops the decimals
	if (numElems % max_elems_per_block != 0) 
		grid_sz += 1;

	// Conflict free padding requires that shared memory be more than 2 * block_sz
	int shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks
	int* d_block_sums;
	(cudaMalloc(&d_block_sums, sizeof(int) * grid_sz));
	(cudaMemset(d_block_sums, 0, sizeof(int) * grid_sz));

	// Sum scan data allocated to each block
	//gpu_sum_scan_blelloch<<<grid_sz, block_sz, sizeof(int) * max_elems_per_block >>>(d_out, d_in, d_block_sums, numElems);
	gpu_prescan<<<grid_sz, block_sz, sizeof(int) * shmem_sz>>>(d_out, 
																	d_in, 
																	d_block_sums, 
																	numElems, 
																	shmem_sz,
																	max_elems_per_block);

	// Sum scan total sums produced by each block
	// Use basic implementation if number of total sums is <= 2 * block_sz
	//  (This requires only one block to do the scan)
	if (grid_sz <= max_elems_per_block)
	{
		int* d_dummy_blocks_sums;
		(cudaMalloc(&d_dummy_blocks_sums, sizeof(int)));
		(cudaMemset(d_dummy_blocks_sums, 0, sizeof(int)));
		//gpu_sum_scan_blelloch<<<1, block_sz, sizeof(int) * max_elems_per_block>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz);
		gpu_prescan<<<1, block_sz, sizeof(int) * shmem_sz>>>(d_block_sums, 
																	d_block_sums, 
																	d_dummy_blocks_sums, 
																	grid_sz, 
																	shmem_sz,
																	max_elems_per_block);
		(cudaFree(d_dummy_blocks_sums));
	}
	// Else, recurse on this same function as you'll need the full-blown scan
	//  for the block sums
	else
	{
		int* d_in_block_sums;
		(cudaMalloc(&d_in_block_sums, sizeof(int) * grid_sz));
		(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(int) * grid_sz, cudaMemcpyDeviceToDevice));
		sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
		(cudaFree(d_in_block_sums));
	}
	
	//// Uncomment to examine block sums
	//int* h_block_sums = new int[grid_sz];
	//(cudaMemcpy(h_block_sums, d_block_sums, sizeof(int) * grid_sz, cudaMemcpyDeviceToHost));
	//std::cout << "Block sums: ";
	//for (int i = 0; i < grid_sz; ++i)
	//{
	//	std::cout << h_block_sums[i] << ", ";
	//}
	//std::cout << std::endl;
	//std::cout << "Block sums length: " << grid_sz << std::endl;
	//delete[] h_block_sums;

	// Add each block's total sum to its scan output
	// in order to get the final, global scanned array
	gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

	(cudaFree(d_block_sums));
}


__host__
gpuBFS::gpuBFS(csr &graph, int source)
{
    checkCudaErrors(cudaMalloc(&d_col_idx, graph.num_edges * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_distance, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_parent, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_in_q, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_out_q, graph.num_nodes * sizeof(int)));

    checkCudaErrors(cudaMalloc(&d_degrees, graph.num_nodes * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_degrees_total, graph.num_nodes * sizeof(int)));

    init_distance(graph, source);

    int firstElementQueue = source;
    cudaMemcpy(d_in_q, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));

    // std::cout << "hi5" << std::endl;

    cudaEvent_t gpu_start, gpu_end;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_start);

    int *queueSize;
    queueSize = (int *)malloc(sizeof(int));
    *queueSize = 1;
    int *nextQueueSize;
    nextQueueSize = (int *)malloc(sizeof(int));
    *nextQueueSize = 0;
    iteration = 0;
    while (*queueSize)
    {

        // std::cout << "iter and size: " << iteration << " " << *queueSize << std::endl;
        
        // next layer phase
        int block_size = 1024;
        int num_blocks = (*queueSize / block_size) + 1;

        nextLayer<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_distance, iteration);
        cudaDeviceSynchronize();

        // Kernel launch code
        countDegrees<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent,
                                            *queueSize, d_in_q, d_degrees, 
                                            d_distance, iteration);
        cudaDeviceSynchronize();

        sum_scan_blelloch(d_degrees_total, d_degrees, *queueSize+1);
        // blockPrefixSum(*queueSize);        
        cudaDeviceSynchronize();

        // std::cout << "fin sum_scan_blelloch\n";

        // *nextQueueSize = d_degrees_total[(*queueSize - 1) / 1024 + 1];
        // std::cout << "queue size " << *queueSize << std::endl;
        // *nextQueueSize = d_degrees_total[*queueSize];
        cudaMemcpy(nextQueueSize, &d_degrees_total[*queueSize], sizeof(int), cudaMemcpyDeviceToHost);

        // std::cout << "next size " << *nextQueueSize << std::endl;

        gather<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_out_q, d_degrees_total, d_distance, iteration);
        cudaDeviceSynchronize();

        // std::cout << "fin gather\n";


        iteration++;
        *queueSize = *nextQueueSize;
        std::swap(d_in_q, d_out_q);

        // if (iteration == 10)
        // {
        //     break;
        // }

        // break;
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&exec_time, gpu_start, gpu_end);

    free(queueSize);
    free(nextQueueSize);

}

__host__
void gpuBFS::init_distance(csr &graph, int source)
{
    // allocate host memory
    host_distance = (int *)malloc(graph.num_nodes * sizeof(int));

    // allocate device memory
    cudaMalloc(&d_distance, graph.num_nodes * sizeof(int));

    // copy memory from host to device
    cudaMemcpy(d_distance, host_distance, graph.num_nodes * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel to inialize distance
    dim3 block(1024, 1);
    dim3 grid((graph.num_nodes+block.x-1)/block.x, 1);
    init_distance_kernel<<<grid, block>>>(d_distance, d_parent, graph.num_nodes, source);

    // copy back
    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
}


__host__
gpuBFS::~gpuBFS()
{
    free(host_distance);
    // free(host_queue);
    

    cudaFree(d_distance);
    cudaFree(d_in_q);
    cudaFree(d_out_q);
    cudaFree(d_parent);
    cudaFree(d_degrees);
    cudaFree(d_col_idx);
    cudaFree(d_row_offset);
    cudaFree(d_degrees_total);
}
