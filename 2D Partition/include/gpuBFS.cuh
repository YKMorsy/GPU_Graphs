#ifndef GPUBFS_H
#define GPUBFS_H

#include <nvshmem.h>
#include <nvshmemx.h>
#include <vector>
#include <stdio.h>

#include "csc.h"
#include "prefixSum.cuh"

class gpuBFS
{
    public:

        gpuBFS(csc &graph, int source, int R, int C);
        ~gpuBFS();

        void print_distance(csc &graph);
        void get_device_distance(csc &graph);
        void check_distance(std::vector<int>& correct_distance);
        void print_stats(int total_edges_traversed);

        int *host_distance;
        float exec_time;
        int te;

    private:

        void init_device(csc &graph, int mype);
        void init_source(int mype, int source);

        int pe_rows, pe_cols;

        int start_node;

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

        int *d_start_col_node;

        int *d_dest_verts_count;
        int *d_global_frontier_count;
        int *d_all_frontier_count;
        int *d_all_frontier;

        int *d_te;
};


#endif