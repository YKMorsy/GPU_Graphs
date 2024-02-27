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
        std::vector<int> col_idx;
        std::vector<int> row_offset;
};

#endif