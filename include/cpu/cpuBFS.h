#ifndef CPUBFS_H
#define CPUBFS_H

#include <vector>
#include <queue>
#include <iostream>
#include "../csr/csr.h"

class cpuBFS
{
    public:
        cpuBFS(csr graph, int source);
        std::vector<int> distance;

};


#endif