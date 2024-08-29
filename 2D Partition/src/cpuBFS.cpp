#include "../include/cpuBFS.h"

cpuBFS::cpuBFS(csc &graph, int source) 
{
    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    // get csc arrays
    std::vector<int> col_offset = graph.csc_vect[0].first;
    std::vector<int> row_index = graph.csc_vect[0].second;
    int num_nodes = col_offset.size()-1;

    distance.resize(num_nodes, -1);

    std::vector<int> frontier;
    std::vector<int> pred(num_nodes, -1);
    std::vector<int> bmap(num_nodes, 0);

    frontier.push_back(source);
    distance[source] = 0;
    bmap[source] = 1;

    int iteration = 1;
    total_edges_traversed = 0;

    while (true) 
    {
        size_t level_size = frontier.size();  // Number of nodes at the current level

        std::vector< std::pair< int, int > > frontier_edges;

        // get nodes in fronteir and get edges of each node
        for (int i = 0; i < frontier.size(); i++)
        {
            int u = frontier[i];
            int row_start = col_offset[u];
            int row_end = col_offset[u+1];

            std::vector<int> u_edge_list(row_index.begin() + row_start, row_index.begin() + row_end);

            for (int j = 0; j < u_edge_list.size(); j++)
            {
                frontier_edges.push_back({u, u_edge_list[j]});
            }
        }

        frontier.clear();

        for (int i = 0; i < frontier_edges.size(); i++)
        {
            int u = frontier_edges[i].first;
            int v = frontier_edges[i].second;
            if (bmap[v] == 0)
            {
                bmap[v] = 1;
                pred[v] = u;
                distance[v] = iteration;
                frontier.push_back(v);
            }
        }

        iteration++;

        if (frontier.size() == 0)
        {
            break;
        }
    }

    cpu_end = clock();
    exec_time = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000;
}

cpuBFS::~cpuBFS() 
{
}

void cpuBFS::print_distance(csc &graph)
{
    std::cout << "\n------CPU DISTANCE VECTOR------" << std::endl;

    std::vector<int> col_offset = graph.csc_vect[0].first;
    std::vector<int> row_index = graph.csc_vect[0].second;
    int num_nodes = col_offset.size()-1;

    for (int i = 0; i < num_nodes; i++) 
    {
        std::cout << distance[i] << " | ";
    }
    std::cout << std::endl;
}
