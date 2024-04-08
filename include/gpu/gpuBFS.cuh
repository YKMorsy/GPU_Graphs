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
        
        int *host_distance;
        float exec_time;
        int iteration;
    private:

        void init_distance(csr &graph);
        void init_queue(csr &graph);
        void init_graph_for_device(csr &graph);

        int *host_queue;
        int *host_cur_queue_size;

        int *device_distance;
        int *device_in_queue;
        int *device_out_queue_size;
        int *device_out_queue;

        int *device_col_idx;
        int *device_row_offset;
};

#endif