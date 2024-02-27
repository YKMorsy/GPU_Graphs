#ifndef GPUBFS_H
#define GPUBFS_H

#include <stdio.h>
#include <queue>
#include <iostream>
#include "../csr/csr.h"

class gpuBFS
{
    public:

        gpuBFS(csr graph, int source);
        ~gpuBFS();
        
        int num_nodes;
        int *host_distance;
    private:

        void init_distance();
        void init_queue();

        int *host_queue;
        int *host_cur_queue_size;

        int *device_distance;
        int *device_queue;
        int *device_cur_queue_size;
};

#endif