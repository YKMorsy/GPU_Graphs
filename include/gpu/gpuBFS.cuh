#ifndef GPUBFS_H
#define GPUBFS_H

// #include <dpct/dpct.hpp>
// #include <CL/sycl.hpp>
#include <stdio.h>
#include <queue>
#include <iostream>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "../csr/csr.h"

class gpuBFS
{
    public:

        gpuBFS(csr &graph, int source);
        ~gpuBFS();
        
        int *host_distance;
        float exec_time;
        int iteration;
    private:

        void init_distance(csr &graph, int source);
        void scanLargeDeviceArray(int *d_out, int *d_in, int length);
        void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length);
        void blockPrefixSum(int size);
        
        // void init_queue(csr &graph);
        // void init_graph_for_device(csr &graph);

        int *d_degrees_total;

        int *host_queue;
        
        int *d_distance;
        int *d_in_q;
        // int *d_out_q_size;
        int *d_out_q;

        // int *d_edges_size;
        int *d_parent;
        int *d_degrees;

        int *d_col_idx;
        int *d_row_offset;
};

#endif