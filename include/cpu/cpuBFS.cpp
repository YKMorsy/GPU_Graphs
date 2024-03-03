#include "cpuBFS.h"

cpuBFS::cpuBFS(csr &graph, int source) : distance(graph.num_nodes, -1)
{    
    int cur_level = 0;
    std::queue<int> frontier;
    
    // add source to node
    frontier.push(source);
    distance[source] = cur_level;

    // loop until node queue is empty
    while(!frontier.empty())
    {
        // get front node from queue
        int cur_node = frontier.front();
        frontier.pop();

        // std::cout << cur_node << std::endl;
        
        // get neighbors of current node and add to queue
        int row_offset_start = graph.row_offset[cur_node];
        
        // std::cout << "Current Node: " << cur_node << std::endl;

        if (row_offset_start != -1)
        {
            // find next row offset that isn't negative 1
            int row_offset_end = -1;
            for (int i = cur_node+1; i < graph.num_nodes + 1; i++)
            {
                // std::cout << graph.row_offset[i] << std::endl;
                if (graph.row_offset[i] != -1)
                {
                    row_offset_end = graph.row_offset[i];
                    break;
                }
            }
             

            // std::cout << "Current Node to Check: " << cur_node << std::endl;
            // std::cout << "Row Start: " << row_offset_start << std::endl;
            // std::cout << "Row End: " << row_offset_end << std::endl;

            // std::cout << row_offset_start << " and " << row_offset_end << std::endl;
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
    }

        
}