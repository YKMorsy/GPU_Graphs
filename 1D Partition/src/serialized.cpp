#include "../include/serialized.h"

serialized::serialized(csr &graph, int source) : iteration(0) {


    d_col_idx = (int*)malloc(sizeof(int) * graph.num_edges);
    d_row_offset = (int*)malloc(sizeof(int) * (graph.num_nodes + 1));
    d_distance = (int*)malloc(sizeof(int) * graph.num_nodes);
    d_parent = (int*)malloc(sizeof(int) * graph.num_nodes);
    d_in_q = (int*)malloc(sizeof(int) * graph.num_nodes);
    d_out_q = (int*)malloc(sizeof(int) * graph.num_nodes);
    d_degrees = (int*)malloc(sizeof(int) * graph.num_nodes);
    d_degrees_scan = (int*)malloc(sizeof(int) * graph.num_nodes);
    d_degrees_total = (int*)malloc(sizeof(int) * graph.num_nodes);
    memcpy(d_col_idx, graph.col_idx, sizeof(int) * graph.num_edges);
    memcpy(d_row_offset, graph.row_offset, sizeof(int) * (graph.num_nodes + 1));

    init_distance(graph, source);

    int firstElementQueue = source;
    d_in_q[0] = firstElementQueue;

    queueSize = 1;
    int nextQueueSize = 0;


    while (queueSize) {

        // std::cout << "iter and size " << iteration << " " << queueSize << std::endl;

        nextLayer();

        countDegrees();
        
        blockPrefixSum(queueSize);

        nextQueueSize = d_degrees_scan[queueSize];

        gather();

        iteration++;
        queueSize = nextQueueSize;
        std::swap(d_in_q, d_out_q);
    }

    distance = d_distance; // Assign the distance array to public variable
    free(d_col_idx);
    free(d_row_offset);
    free(d_distance);
    free(d_parent);
    free(d_in_q);
    free(d_out_q);
    free(d_degrees);
    free(d_degrees_scan);
    free(d_degrees_total);
}

serialized::~serialized() {}

void serialized::init_distance(csr &graph, int source) {
    for (int i = 0; i < graph.num_nodes; ++i) {
        if (i == source) {
            d_distance[i] = 0;
            d_parent[i] = -1;
        } else {
            d_distance[i] = -1;
            d_parent[i] = -1;
        }
    }
}

void serialized::nextLayer() {
    for (int thid = 0; thid < queueSize; thid++) {

        int cur_node = d_in_q[thid];

        int row_offset_start = d_row_offset[cur_node];
        int row_offset_end = d_row_offset[cur_node + 1];

        int degree = 0;

        for (int j = row_offset_start; j < row_offset_end; j++) {
            int v = d_col_idx[j];

            if (d_distance[v] == -1) 
            {
                d_parent[v] = j;
                d_distance[v] = iteration + 1;
            }
        }

    }
}

void serialized::countDegrees() {

    int deg_all = 0;

    for (int thid = 0; thid < queueSize; thid++) {

        int cur_node = d_in_q[thid];

        int row_offset_start = d_row_offset[cur_node];
        int row_offset_end = d_row_offset[cur_node + 1];

        int degree = 0;

        for (int j = row_offset_start; j < row_offset_end; j++) {
            int v = d_col_idx[j];

            if (d_parent[v] == j && v != cur_node) 
            {
                degree++;
            }
        }        

        d_degrees[thid] = degree;

        deg_all += degree;

    }

    // if (iteration == 4)
    // {
    //     std::cout << "degree total " << deg_all << std::endl;
    //     // printf("thid and degree %d %d\n", thid, degree);
    // }
}

void serialized::blockPrefixSum(int size) {

    d_degrees_scan[0] = 0;

    for (int thid = 1; thid <= size; thid++) {
        d_degrees_scan[thid] = d_degrees[thid-1] + d_degrees_scan[thid-1];
    }
}

void serialized::gather() {

    for (int thid = 0; thid < queueSize; thid++) 
    {
        int nextQueuePlace = d_degrees_scan[thid];

        int count = 0;

        int u = d_in_q[thid];
        int row_offset_start = d_row_offset[u];
        int row_offset_end = d_row_offset[u + 1];
        for (int i = row_offset_start; i < row_offset_end; i++) {
            int v = d_col_idx[i];

            if (d_parent[v] == i && v != u) 
            {
                d_out_q[nextQueuePlace] = v; // Use nextQueuePlace to index into nextQueue and then increment it
                nextQueuePlace++;
                count++;
            }
            
        }
    }
}