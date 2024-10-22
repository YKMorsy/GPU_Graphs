#ifndef SERIALIZED_H
#define SERIALIZED_H

#include <vector>
#include <queue>
#include <iostream>
#include <time.h>
#include <limits>
#include "csr.h"
#include <cstring>

class serialized {
public:
    serialized(csr &graph, int source);
    ~serialized();

    int* distance; // Distance array accessible publicly
    float exec_time; // Execution time accessible publicly
    int iteration;

private:
    void init_distance(csr &graph, int source);
    void nextLayer();
    void countDegrees();
    void blockPrefixSum(int size);
    void gather();

    int *visited;
    int *d_col_idx;
    int *d_row_offset;
    int *d_distance;
    int *d_parent;
    int *d_in_q;
    int *d_out_q;
    int *d_degrees;
    int *d_degrees_scan;
    int *d_degrees_total;
    int queueSize;
};

#endif