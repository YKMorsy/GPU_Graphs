#ifndef GPUBFS_H
#define GPUBFS_H

// #include <dpct/dpct.hpp>
// #include <CL/sycl.hpp>
#include <stdio.h>
#include <queue>
#include <iostream>
#include <time.h>
#include "../csr/csr.h"

class gpuBFS
{
    public:

        gpuBFS(csr &graph, int source);
        ~gpuBFS();
        
        long long int *host_distance;
        float exec_time;
        long long int iteration;
    private:

        void init_distance(csr &graph);
        void init_queue(csr &graph);
        void init_graph_for_device(csr &graph);

        long long int *host_queue;
        long long int *host_cur_queue_size;

        long long int *device_distance;
        long long int *device_in_queue;
        long long int *device_out_queue_size;
        long long int *device_out_queue;

        long long int *device_col_idx;
        long long int *device_row_offset;
};

#endif