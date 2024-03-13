#include "cpuBFS.h"

cpuBFS::cpuBFS(csr &graph, int source)
{    
    clock_t cpu_start, cpu_end;
    cpu_start = clock();
    std::queue<int> frontier;

    distance.reserve(graph.num_nodes);
    for (int i = 0; i < graph.num_nodes; i++)
    {
        distance[i] = -1;
    }

    
    // add source to node
    frontier.push(source);
    distance[source] = 0;


    // loop until node queue is empty
    while(!frontier.empty())
    {
        // get front node from queue
        int cur_node = frontier.front();
        frontier.pop();

        // std::cout << cur_node << std::endl;
        
        // get neighbors of current node and add to queue
        int row_offset_start = graph.row_offset[cur_node];
        int row_offset_end = graph.row_offset[cur_node+1];

        for (int i = row_offset_start; i < row_offset_end; i++)
        {
            int neighbor = graph.col_idx[i];

            // std::cout << "Neighbor Node: " << neighbor << std::endl;

            // check if node already visited
            if (distance[neighbor] == -1)
            {
                frontier.push(neighbor);
                // update level of added neighbor nodes
                distance[neighbor] = distance[cur_node] + 1;
            }
        }

    }

    cpu_end = clock();

    exec_time = (((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC) * 1000;

        
}