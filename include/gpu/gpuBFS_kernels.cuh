#ifndef GPU_BFS_KERNELS_CUH
#define GPU_BFS_KERNELS_CUH

#include <limits>
#include <nvshmem.h>
#include <nvshmemx.h>
#define INF std::numeric_limits<int>::max()
constexpr unsigned int FULL_MASK = 0xffffffff;
constexpr unsigned int WARP_SIZE = 32;
constexpr unsigned int WARPS = 1024/WARP_SIZE;
constexpr size_t HASH_RANGE = 128;

struct prescan_result
{
	int offset, total;
};

__global__ void linear_bfs(const int total_nodes, const int starting_col_idx_pre_pe, const int* start_node, const int* end_node, const int* row_offset, const int* column_index, int* distance, const int iteration, const int* in_queue, const int in_queue_count, int* out_queue, int* out_queue_count, int* edges_traversed);
__global__ void init_distance_kernel(const int* start_node, const int* end_node, int size, int *device_distance, int source);
__global__ void expand_contract_bfs(const int num_nodes, const int* row_offset, const int* column_index, int* distance, const int iteration, const int* in_queue, const int in_queue_count, int* out_queue, int* out_queue_count);

#endif