#include "cpuBFS.h"

cpuBFS::cpuBFS(std::vector<int> col_idx, std::vector<int> row_offset, int source) : distance(row_offset.size() - 1, -1)
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
        
        // get neighbors of current node and add to queue
        int row_offset_start = row_offset[cur_node];
        int row_offset_end =  row_offset[cur_node+1];

        for (int i = row_offset_start; i < row_offset_end; i++)
        {
            // check if node already visited
            if (distance[col_idx[i]] == -1)
            {
                frontier.push(col_idx[i]);
                // update level of added neighbor nodes
                distance[col_idx[i]] = distance[cur_node] + 1;
            }
        }
    }

        
}