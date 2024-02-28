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
        int row_offset_end =  graph.row_offset[cur_node+1];

        if (graph.row_offset[cur_node] != -1)
        {
            // std::cout << row_offset_start << " and " << row_offset_end << std::endl;
            for (int i = row_offset_start; i < row_offset_end; i++)
            {
                // check if node already visited
                if (distance[graph.col_idx[i]] == -1)
                {
                    frontier.push(graph.col_idx[i]);
                    // update level of added neighbor nodes
                    distance[graph.col_idx[i]] = distance[cur_node] + 1;
                }
            }
        }
    }

        
}