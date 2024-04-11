#include "gpuBFS.cuh"

__global__
void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_parent,
                  int queueSize, int *d_currentQueue, long long int *d_degrees, int *d_distance)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];

        int degree = 0;

        // printf("thid and start and end %lld %lld %lld\n", thid, row_offset_start, row_offset_end);

        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_adjacencyList[i];
            if (d_distance[v] == -1) 
            {
                degree++;
            }
        }
        d_degrees[thid] = degree;
    }
}

__global__
void gather(int *d_adjacencyList, int *d_edgesOffset, int *d_parent, int queueSize,
            int *d_currentQueue, int *d_nextQueue, long long int *d_degrees, long long int *incrDegrees, int *d_distance, int iteration)
{
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

    if (thid < queueSize) {
        // __shared__ int sharedIncrement;
        // if (!threadIdx.x) {
        //     sharedIncrement = incrDegrees[thid >> 10];
        // }
        // __syncthreads();

        // int sum = 0;
        // if (threadIdx.x) {
        //     sum = d_degrees[thid - 1];
        // }

        int nextQueuePlace = d_degrees[thid];

        int cur_node = d_currentQueue[thid];
        int row_offset_start = d_edgesOffset[cur_node];
        int row_offset_end = d_edgesOffset[cur_node+1];
        // int counter = 0;
        for (int i = row_offset_start; i < row_offset_end; i++)
        {
            int v = d_adjacencyList[i];
            if (d_distance[v] == -1) {
                // int nextQueuePlace = sharedIncrement + sum + counter;
                // printf("individiual %d %d\n", thid, nextQueuePlace);
                d_distance[v] = iteration + 1;
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
            device_parent[i] = 0;
        }
        else
        {
            device_distance[i] = -1;
            device_parent[i] = -1;
        }

    }
}

int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}


__global__ 
void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ 
void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}

__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}



void scanLargeDeviceArray(int *d_out, int *d_in, int length) {
	int remainder = length % (1024);
	if (remainder == 0) 
    {
		scanLargeEvenDeviceArray(d_out, d_in, length);
	}
	else 
    {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length) 
{
	int powerOfTwo = nextPowerOfTwo(length);
    prescan_arbitrary_unoptimized<< <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length) {
	const int blocks = length / 1024;
	const int sharedMemArraySize = 1024 * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

    prescan_large_unoptimized<<<blocks, 1024, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > 1024) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, 1024, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}


__host__
gpuBFS::gpuBFS(csr &graph, int source)
{
    // std::cout << "hi1" << std::endl;

    cudaMalloc(&d_col_idx, graph.num_edges * sizeof(int));
    cudaMalloc(&d_row_offset, (graph.num_nodes+1) * sizeof(int));
    // cudaMalloc(&d_edges_size, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_distance, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_parent, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_in_q, graph.num_nodes * sizeof(int));
    cudaMalloc(&d_out_q, graph.num_nodes * sizeof(int));
    cudaMallocHost(&d_degrees, graph.num_nodes * sizeof(long long int));

    cudaMallocHost((void **) &d_degrees_total, graph.num_nodes * sizeof(long long int));

    // std::cout << "hi2" << std::endl;

    init_distance(graph, source);

    // std::cout << "hi3" << std::endl;

    int firstElementQueue = source;
    cudaMemcpy(d_in_q, &firstElementQueue, sizeof(int), cudaMemcpyHostToDevice);

    // std::cout << "hi4" << std::endl;
    
    // d_degrees_total = (int *)malloc(graph.num_nodes * sizeof(int));

    // intToLongLong<<<1024, graph.num_edges>>>(graph.col_idx, d_col_idx, graph.num_edges);
    // intToLongLong<<<1024, (graph.num_nodes+1)>>>(graph.row_offset, d_row_offset, (graph.num_nodes+1));
    cudaMemcpy(d_col_idx, graph.col_idx, graph.num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_offset, graph.row_offset, (graph.num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice);

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

        std::cout << "iter and size: " << iteration << " " << *queueSize << std::endl;
        
        // next layer phase
        int block_size = 1024;
        int num_blocks = *queueSize / block_size + 1;

        countDegrees<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_degrees, d_distance);
        cudaDeviceSynchronize();

        if (*queueSize > 1024) 
        {
            scanLargeDeviceArray(d_degrees_total, d_degrees, *queueSize);
        }
        else {
            scanSmallDeviceArray(d_degrees_total, d_degrees, *queueSize);
        }
            
        // prescan_large_unoptimized<<<num_blocks, block_size>>>(d_degrees_total, d_degrees, 1024, d_sums);
        // block_prefix_sum<<<num_blocks, block_size>>>(*queueSize, d_degrees) ;
        // prescan<<<num_blocks, block_size>>>(d_degrees_total, d_degrees, *queueSize) ;
        cudaDeviceSynchronize();

        // *nextQueueSize = d_degrees_total[(*queueSize - 1) / 1024 + 1];
        // std::cout << *queueSize << std::endl;
        // *nextQueueSize = d_degrees_total[*queueSize-1];
        // std::cout << d_degrees[*queueSize-1] << std::endl;
        *nextQueueSize = d_degrees[*queueSize];
        std::cout << *nextQueueSize << std::endl;

        gather<<<num_blocks, block_size>>>(d_col_idx, d_row_offset, d_parent, *queueSize, d_in_q, d_out_q, d_degrees_total, d_degrees_total, d_distance, iteration);
        cudaDeviceSynchronize();

        iteration++;
        *queueSize = *nextQueueSize;
        std::swap(d_in_q, d_out_q);

        if (iteration == 4)
        {
            break;
        }

        // break;
    }

    cudaMemcpy(host_distance, d_distance, graph.num_nodes * sizeof(int), cudaMemcpyDeviceToHost);
    // std::cout << host_distance[1] << " " << host_distance[2] << std::endl;
    // host_distance[source] = 0;

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
