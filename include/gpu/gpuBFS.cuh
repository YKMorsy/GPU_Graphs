#ifndef GPUBFS_H
#define GPUBFS_H

#include <nvshmem.h>
#include <nvshmemx.h>
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

        void init_device(csr &graph, int source);
        void get_device_distance(csr &graph);
        
        int *d_distance_local;
        
        int *d_in_q;
        int *d_out_q;
        int *d_q_count;
        
        
        int *d_col_idx;
        int *d_row_offset;

        int *d_start_node;
        int *d_end_node;
        int *d_edges_traversed;
        int *d_graph_edges_gpu;
        int *d_global_q_count;

        int mype_node;
        int mype;
        int npes;
        int start_node;
        int end_node;
        int starting_col_idx_pre_pe;

        int h_q_count;
        int graph_nodes_gpu;
        int remainder_nodes;
        int graph_edges_gpu;
};

#endif