#ifndef GPUBFS_H
#define GPUBFS_H

#include <nvshmem.h>
#include "gpuBFS_kernels.cuh"
#include "../csr/csr.h"

class gpuBFS
{
    public:

        gpuBFS(csr &graph, int source);
        ~gpuBFS();
        void print_distance(csr &graph);
        
        int *host_distance;
        float exec_time = 0;
        int total_edges_traversed = 0;
        int iteration = 0;
        
    private:

        void init_device(csr &graph, int source, int mype, int npes, int node_start, int node_end);
        void scanLargeDeviceArray(int *d_out, int *d_in, int length);
        void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length);
        void blockPrefixSum(int size);
        
        int *d_distance;
        
        int *d_in_q;
        int *d_out_q;
        int *d_q_count;
        int h_q_count = 0;
        
        int *d_col_idx;
        int *d_row_offset;
};

#endif