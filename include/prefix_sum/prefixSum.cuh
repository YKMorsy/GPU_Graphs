#ifndef PREFIXSUM_H
#define PREFIXSUM_H

class PrefixSum 
{
    public:
        void sum_scan_blelloch(int* const d_out,
                                const int* const d_in,
                                const size_t numElems);  // Function used by GPU
};

#endif