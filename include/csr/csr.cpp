#include "csr.h"

csr::csr(const char* filename)
{
    // read file
    std::ifstream infile(filename);

    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open file" << filename << "\n";
        return;
    }
    
    // get number of nodes and edges
    int from, to;
    infile >> num_nodes;
    infile >> num_edges;

    // Create adjacency list
    // col_idx.reserve(num_edges);
    // row_offset.reserve(num_nodes+1);
    col_idx = (int *)malloc(num_edges * sizeof(int));
    row_offset = (int *)malloc((num_nodes+1) * sizeof(int));

    std::vector<std::vector<int>> adjList(num_nodes+1);
    while (infile >> from >> to)
    {
        adjList[from-1].push_back(to-1);
    }

    // convert adjacency list to compressed sparse row format
    int offset_counter = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        std::vector<int> cur_node = adjList[i];

        if (cur_node.size() > 0)
        {
            row_offset[i] = offset_counter;
            // row_offset.push_back(offset_counter);
        }
        else
        {
            row_offset[i] = -1;
            // row_offset.push_back(-1);
        }

        for (int j = 0; j < cur_node.size(); j++)
        {
            // col_idx.push_back(cur_node[j]);
            col_idx[offset_counter] = cur_node[j];
            offset_counter++;
        }
    }

    // Add final offset for the end
    row_offset[num_nodes] = offset_counter;

}

csr::~csr()
{
    free(col_idx);
    free(row_offset);
}