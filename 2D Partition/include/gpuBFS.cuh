#ifndef GPUBFS_H
#define GPUBFS_H

#include <nvshmem.h>
#include <nvshmemx.h>
#include <vector>

#include "csc.h"

class gpuBFS
{
    public:

        gpuBFS(csc &graph, int source, int R, int C);
        ~gpuBFS();

        void print_distance(csc &graph);

        int *host_distance;
        float exec_time;

    private:

        void init_device(csc &graph, int mype);
        void init_source(int mype, int source);

        int pe_rows, pe_cols;

        int total_nodes;
        int pe_nodes;
        
        int *d_col_offset;
        int *d_row_index;

        int *d_frontier;
        int *d_frontier_count;
        int *d_pred;
        int *d_bmap;

        int *d_dest_verts;

        int *d_distance;
        int *d_cumulative_degrees;
        int *d_degrees;
        int *d_total_degrees;
};

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// NVSHMEM error checking macro
#define NVSHMEM_CHECK(stmt)                                                                \
    do {                                                                                   \
        int result = (stmt);                                                               \
        if (NVSHMEMX_SUCCESS != result) {                                                  \
            fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, \
                    result);                                                               \
            exit(-1);                                                                      \
        }                                                                                  \
    } while (0)

#endif