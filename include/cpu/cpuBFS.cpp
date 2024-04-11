#include "cpuBFS.h"

cpuBFS::cpuBFS(csr &graph, int source) {
    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    distance = (long long int*)malloc(graph.num_nodes * sizeof(long long int));
    if (distance == nullptr) {
        std::cerr << "Memory allocation failed for distance array." << std::endl;
        return;
    }

    std::fill_n(distance, graph.num_nodes, -1);

    std::queue<long long int> frontier;
    frontier.push(source);
    distance[source] = 0;

    iteration = 0;

    while (!frontier.empty()) {
        size_t level_size = frontier.size();  // Number of nodes at the current level

        // std::cout << "iter and size " << iteration << " " << level_size << std::endl;

        for (size_t i = 0; i < level_size; ++i) {
            long long int cur_node = frontier.front();
            frontier.pop();

            long long int row_offset_start = graph.row_offset[cur_node];
            long long int row_offset_end = graph.row_offset[cur_node + 1];

            for (long long int j = row_offset_start; j < row_offset_end; ++j) {
                long long int neighbor = graph.col_idx[j];

                if (distance[neighbor] == -1) {
                    frontier.push(neighbor);
                    distance[neighbor] = iteration + 1;
                }
            }
        }

        iteration++;  // Completed processing of one level
    }

    cpu_end = clock();
    exec_time = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
}

cpuBFS::~cpuBFS() {
    free(distance);
}
