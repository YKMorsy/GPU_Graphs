#include "serialized.h"

serialized::serialized(csr &graph, int source) : iteration(0) {

    d_col_idx = new long long int[graph.num_edges];
    d_row_offset = new long long int[graph.num_nodes + 1];
    d_distance = new long long int[graph.num_nodes];
    d_parent = new long long int[graph.num_nodes];
    d_in_q = new long long int[graph.num_nodes];
    d_out_q = new long long int[graph.num_nodes];
    d_degrees = new long long int[graph.num_nodes];
    d_degrees_scan = new long long int[graph.num_nodes];
    d_degrees_total = new long long int[graph.num_nodes];
    std::copy(graph.col_idx, graph.col_idx + graph.num_edges, d_col_idx);
    std::copy(graph.row_offset, graph.row_offset + graph.num_nodes + 1, d_row_offset);

    init_distance(graph, source);

    long long int firstElementQueue = source;
    d_in_q[0] = firstElementQueue;

    queueSize = 1;
    long long int nextQueueSize = 0;


    while (queueSize) {

        std::cout << "iter and size " << iteration << " " << queueSize << std::endl;

        countDegrees();

        blockPrefixSum(queueSize);

        nextQueueSize = d_degrees_scan[queueSize];

        gather();

        iteration++;
        queueSize = nextQueueSize;
        std::swap(d_in_q, d_out_q);
    }

    // std::cout << "iter " << iteration << std::endl;

    distance = d_distance; // Assign the distance array to public variable
    delete[] d_col_idx;
    delete[] d_row_offset;
    delete[] d_parent;
    delete[] d_in_q;
    delete[] d_out_q;
    delete[] d_degrees;
    delete[] d_degrees_scan;
    delete[] d_degrees_total;
}

serialized::~serialized() {}

void serialized::init_distance(csr &graph, int source) {
    for (long long int i = 0; i < graph.num_nodes; ++i) {
        if (i == source) {
            d_distance[i] = 0;
            d_parent[i] = -1;
        } else {
            d_distance[i] = -1;
            d_parent[i] = -1;
        }
    }
}

void serialized::countDegrees() {
    for (long long int thid = 0; thid < queueSize; thid++) {

        long long int cur_node = d_in_q[thid];
        long long int row_offset_start = d_row_offset[cur_node];
        long long int row_offset_end = d_row_offset[cur_node + 1];

        long long int degree = 0;

        for (long long int j = row_offset_start; j < row_offset_end; ++j) {
            long long int v = d_col_idx[j];
            // if (d_parent[v] == j && v != cur_node) 
            if (d_distance[v] == -1) 
            {
                degree++;
            }
        }
        d_degrees[thid] = degree;
    }
}

void serialized::blockPrefixSum(long long int size) {

    d_degrees_scan[0] = 0;

    for (long long int thid = 1; thid <= size; thid++) {
        d_degrees_scan[thid] = d_degrees[thid-1] + d_degrees_scan[thid-1];
    }
}

void serialized::gather() {

    for (long long int thid = 0; thid < queueSize; thid++) 
    {

        long long int totalPlace = d_degrees_scan[thid+1];
        
        long long int nextQueuePlace = d_degrees_scan[thid];

        long long int u = d_in_q[thid];
        long long int row_offset_start = d_row_offset[u];
        long long int row_offset_end = d_row_offset[u + 1];
        for (long long int i = row_offset_start; i < row_offset_end; i++) {
            long long int v = d_col_idx[i];
            // if (d_parent[v] == i && v != u) 
            if (d_distance[v] == -1) 
            {
                d_distance[v] = iteration + 1;
                d_out_q[nextQueuePlace] = v; // Use nextQueuePlace to index long long into nextQueue and then increment it
                nextQueuePlace++;
            }
            
        }
    }
}