#ifndef CPUBFS_H
#define CPUBFS_H

#include <queue>
#include <iostream>
#include <time.h>
#include <stdlib.h> // for malloc and free
#include "../csr/csr.h"
#include <limits>

#define INF std::numeric_limits<int>::max()

class cpuBFS
{
public:
    cpuBFS(csr &graph, int source);
    ~cpuBFS(); // Destructor to free allocated memory
    void print_distance(csr &graph);
    long long int *distance; // Using raw pointer for dynamic allocation
    float exec_time;
    int iteration;
    int total_edges_traversed;
};

#endif