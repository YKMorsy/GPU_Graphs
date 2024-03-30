#ifndef SYCLBFS_H
#define SYCLBFS_H

#include <iostream>
#include <CL/sycl.hpp>
// #include <atomic.hpp>
// #include <dpct/dpct.hpp>
// #include <syclcompat/syclcompat.hpp>
#include "../csr/csr.h"



class syclBFS
{
    public:

        syclBFS(csr &graph, int source);
        ~syclBFS();
        
        int *host_distance;
        float exec_time;
    private:

        cl::sycl::queue gpuQueue{ cl::sycl::gpu_selector_v };

        int max_group_size = gpuQueue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

        int graph_num_nodes;
        int graph_num_edges;

        void init_distance(csr &graph);
        void init_queue(csr &graph);
        void init_graph_for_device(csr &graph);

        int *host_queue;
        int host_cur_queue_size;

        int *device_distance;
        int *device_in_queue;
        int *device_out_queue;

        int *device_col_idx;
        int *device_row_offset;

        int *device_out_queue_size;
};


#endif