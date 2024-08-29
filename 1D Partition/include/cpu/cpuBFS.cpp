#include "cpuBFS.h"



cpuBFS::cpuBFS(csr &graph, int source) {
    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    distance = (long long int*)malloc(graph.num_nodes * sizeof(long long int));
    if (distance == nullptr) {
        std::cerr << "Memory allocation failed for distance array." << std::endl;
        return;
    }

    std::fill_n(distance, graph.num_nodes, INF);


    distance_alt = (long long int*)malloc(graph.num_nodes * sizeof(long long int));
    if (distance_alt == nullptr) {
        std::cerr << "Memory allocation failed for distance array." << std::endl;
        return;
    }

    std::fill_n(distance_alt, graph.num_nodes, INF);


    std::queue<long long int> frontier;
    frontier.push(source);
    distance[source] = 0;

    iteration = 0;
    total_edges_traversed = 0;

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

                if (distance[neighbor] == INF) {
                    total_edges_traversed++;
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

void cpuBFS::cpuBFSAlt(csr &graph, int source) {
    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    std::queue<long long int> frontier;
    frontier.push(source);
    // distance[source] = 0;

    iteration = 0;
    total_edges_traversed = 0;

    while (!frontier.empty()) 
    {
        size_t level_size = frontier.size();  // Number of nodes at the current level

        // std::cout << "CPU h_q_count: " << level_size << std::endl;
        
        for (size_t i = 0; i < level_size; ++i) {
            long long int cur_node = frontier.front();
            frontier.pop();

            if (distance_alt[cur_node] == INF)
            {
                distance_alt[cur_node] = iteration;

                long long int row_offset_start = graph.row_offset[cur_node];
                long long int row_offset_end = graph.row_offset[cur_node + 1];

                for (long long int j = row_offset_start; j < row_offset_end; ++j) {
                    long long int neighbor = graph.col_idx[j];

                    total_edges_traversed++;
                    frontier.push(neighbor);
                }
            }
        }

        iteration++;  // Completed processing of one level
    }

    cpu_end = clock();
    exec_time = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
}

void cpuBFS::testDistance(csr &graph) 
{
    int num_mismatch = 0;
    for (long long int i = 0; i < graph.num_nodes; i++) 
    {
        if (distance[i] != distance_alt[i])
        {
            num_mismatch++;
        }
    }
    std::cout << "Num Mismatch: " << num_mismatch << std::endl;
}

cpuBFS::~cpuBFS() 
{
    free(distance);
    free(distance_alt);
}

void cpuBFS::print_distance(csr &graph)
{
    std::cout << "\n------CPU DISTANCE VECTOR------" << std::endl;

    for (long long int i = 0; i < graph.num_nodes; i++) 
    {
        std::cout << distance[i] << " | ";
    }
    std::cout << std::endl;
}
