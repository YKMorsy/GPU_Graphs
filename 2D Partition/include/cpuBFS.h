#ifndef CPUBFS_H
#define CPUBFS_H

#include <vector>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <limits>

#include "csc.h"

class cpuBFS
{
public:
    cpuBFS(csc &graph, int source);
    ~cpuBFS();
    void print_distance(csc &graph);

    std::vector<int> distance;
    float exec_time;
    int total_edges_traversed;
};

#endif