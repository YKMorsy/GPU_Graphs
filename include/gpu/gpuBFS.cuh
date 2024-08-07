#ifndef GPUBFS_H
#define GPUBFS_H

#include <stdio.h>
#include <queue>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include "device_launch_parameters.h"
#include "../csr/csr.h"
#include "../prefix_sum/prefixSum.cuh"

class gpuBFS
{
    public:

        gpuBFS(csr &graph, int source);
        ~gpuBFS();
        
        int *host_distance;
        float exec_time;
        int total_edges_traversed;
        int iteration;
        
    private:

        void init_device(csr &graph, int source, int mype, int npes, int node_start, int node_end);
        void scanLargeDeviceArray(int *d_out, int *d_in, int length);
        void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length);
        void blockPrefixSum(int size);
        
        int *d_distance;
        int *d_in_q;
        int *d_out_q;
        int *d_parent;
        int *d_degrees;
        int *d_degrees_total;
        int *d_col_idx;
        int *d_row_offset;
};

#endif