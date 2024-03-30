#ifndef SYCLBFS_H
#define SYCLBFS_H

#include <iostream>
#include <CL/sycl.hpp>
#include <atomic.hpp>
// #include <dpct/dpct.hpp>
// #include <syclcompat/syclcompat.hpp>
#include "../csr/csr.h"

struct prescan_result
{
    int offset;
    int total;
};

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

        // void expand_contract_kernel(int *device_col_idx, int *device_row_offset, 
        //                             int num_nodes, int *device_in_queue, 
        //                             int device_in_queue_size, int *device_out_queue_size, 
        //                             int *device_distance, int iteration, int *device_out_queue,
        //                             cl::sycl::nd_item<1> &item, int *comm, int *base_offset, int *sums);

        // void block_gather(int* column_index, int* distance, 
        //                     int iteration, int * out_queue, 
        //                     int* out_queue_count, int r, int r_end, 
        //                     cl::sycl::nd_item<1> &item, int *comm,
        //                     int *base_offset, int *sums);

        // void fine_gather(int *device_col_idx, int row_offset_start, 
        //                 int row_offset_end, int *device_distance, 
        //                 int iteration, int *device_out_queue, 
        //                 int *device_out_queue_size, const int node,
        //                 cl::sycl::nd_item<1> &item, int *comm,
        //                 int *base_offset, int *sums);

        // prescan_result block_prefix_sum(int val, cl::sycl::nd_item<1> &item, int *sums);

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