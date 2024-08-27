#ifndef CSC_H
#define CSC_H

#include <vector>
#include <fstream>
#include <iostream>
#include <vector>

class csc
{
    public:
        csc(const char* filename);
        ~csc();
        
        void print_info();
        
        int *col_offset;
        int *row_index;
        int num_nodes;
        int num_edges;
};

#endif