#ifndef CPUBFS_H
#define CPUBFS_H

#include <vector>
#include <queue>
#include <iostream>

class cpuBFS
{
    public:
        cpuBFS(std::vector<int> col_idx, std::vector<int> row_offset, int source);
        std::vector<int> distance;

};


#endif