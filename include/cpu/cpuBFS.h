#ifndef CPUBFS_H
#define CPUBFS_H

#include <queue>
#include <iostream>
#include <time.h>
#include <stdlib.h> // for malloc and free
#include "../csr/csr.h"

class cpuBFS
{
public:
    cpuBFS(csr &graph, int source);
    ~cpuBFS(); // Destructor to free allocated memory
    long long int *distance; // Using raw pointer for dynamic allocation
    float exec_time;
    int iteration;
};

#endif