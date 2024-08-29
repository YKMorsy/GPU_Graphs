#ifndef CSC_H
#define CSC_H

#include <vector>
#include <fstream>
#include <iostream>
#include <cfenv>
#include <cmath>
#include <utility>
#include <algorithm> 

class csc
{
    public:
        csc(const char* filename, int R, int C);
        ~csc();
        
        void print_info(int r, int c);

        std::vector<std::pair<std::vector<int> , std::vector<int>>> csc_vect;
        int num_nodes;
        int num_edges;
        int num_R;
        int num_C;
};

#endif