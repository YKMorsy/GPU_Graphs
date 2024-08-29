#ifndef CSR_H
#define CSR_H

#include <vector>
#include <fstream>
#include <iostream>
#include <vector>

class csr
{
    public:
        csr(const char* filename);
        ~csr();
        int *col_idx;
        int *row_offset;
        int num_nodes;
        int num_edges;
        // std::vector<int> col_idx;
        // std::vector<int> row_offset;
};

#endif