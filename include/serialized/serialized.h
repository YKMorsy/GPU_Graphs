#ifndef SERIALIZED_H
#define SERIALIZED_H

#include <vector>
#include <queue>
#include <iostream>
#include <time.h>
#include "../csr/csr.h"

class serialized {
public:
    serialized(csr &graph, int source);
    ~serialized();

    long long int* distance; // Distance array accessible publicly
    float exec_time; // Execution time accessible publicly
    int iteration;

private:
    void init_distance(csr &graph, int source);
    void nextLayer(long long int level);
    void countDegrees();
    void blockPrefixSum(long long int size);
    void gather();

    int *visited;
    long long int *d_col_idx;
    long long int *d_row_offset;
    long long int *d_distance;
    long long int *d_parent;
    long long int *d_in_q;
    long long int *d_out_q;
    long long int *d_degrees;
    long long int *d_degrees_scan;
    long long int *d_degrees_total;
    long long int queueSize;
};

#endif