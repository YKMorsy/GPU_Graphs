#include "../include/csc.h"

csc::csc(const char* filename)
{
    // read file
    std::ifstream infile(filename);

    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return;
    }
    
    // get number of nodes and edges
    int from, to;
    infile >> num_nodes;
    infile >> num_edges;

    // Create adjacency list
    col_offset = (int *)malloc((num_nodes+1) * sizeof(int));
    row_index = (int *)malloc(num_edges * sizeof(int));

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
            col_offset[i] = offset_counter;
            // row_offset.push_back(offset_counter);
        }
        else
        {
            col_offset[i] = -1;
            // row_offset.push_back(-1);
        }

        for (int j = 0; j < cur_node.size(); j++)
        {
            // col_idx.push_back(cur_node[j]);
            row_index[offset_counter] = cur_node[j];
            offset_counter++;
        }
    }

    // Add final offset for the end
    col_offset[num_nodes] = offset_counter;

    // change -1's to next offset
    for (int i = num_nodes; i >= 0; i--) {
        if (col_offset[i] == -1)
        {
            col_offset[i] = col_offset[i+1];
        }
    }

}

csc::~csc()
{
    free(col_offset);
    free(row_index);
}

void  csc::print_info()
{
    std::cout << "\n------CSS INFO------" << std::endl;

    std::cout << "Column offset: " << num_nodes << std::endl;
    for (int i = 0; i < num_nodes+1; i++) {
        std::cout << col_offset[i] << " | ";
    }
    std::cout << std::endl;

    std::cout << "Row index: " << num_edges << std::endl;
    for (int i = 0; i < num_edges; i++) {
        std::cout << row_index[i] << " | ";
    }
    std::cout << std::endl;
}