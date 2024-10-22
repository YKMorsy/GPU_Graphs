#include "../include/csc.h"

csc::csc(const char* filename, int R, int C)
{
    // read file
    std::ifstream infile(filename);

    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << "\n";
        return;
    }

    infile >> num_nodes;
    infile >> num_edges;

    // Initialize adjacency list
    std::vector<std::vector<int>> adjList(num_nodes);
    
    int from, to;
    while (infile >> from >> to)
    {
        // Store edge in adjacency list (1-based to 0-based index conversion)
        adjList[from-1].push_back(to-1);
    }
    infile.close(); // Close the file after reading

    num_pe_R = R;
    num_pe_C = C;
    int num_proc = R * C;

    int block_size_rows = num_nodes / (num_pe_R * num_pe_C);
    int block_size_cols = num_nodes / num_pe_C;

    csc_vect.resize(num_pe_R * num_pe_C);

    for (int proc = 0; proc < num_proc; proc++)
    {
        std::vector<std::vector<int>> curAdjList(num_nodes / num_pe_C);

        int p_i = proc / num_pe_C;
        int p_j = proc % num_pe_C;
        int start_col = (num_nodes / num_pe_C) * p_j;

        // Pij owns edges in blocks (mR + i, j)
        for (int m = 0; m < C; m++)
        {
            int startRow = (m * R + p_i) * block_size_rows;
            int startCol = p_j * block_size_cols;

            // Process only relevant edges from adjacency list
            for (int col = startCol; col < startCol + block_size_cols; ++col)
            {
                for (int row : adjList[col])
                {
                    if (row >= startRow && row < startRow + block_size_rows)
                    {
                        curAdjList[col - start_col].push_back(row);
                    }
                }
            }
        }

        // Convert adjacency list to CSC format
        std::vector<int> col_offset(num_nodes / num_pe_C + 1, 0);
        for (int col = 0; col < num_nodes / num_pe_C; ++col)
        {
            col_offset[col + 1] = curAdjList[col].size();
        }

        // Compute prefix sum to determine starting indices for each column
        for (int col = 1; col <= num_nodes / num_pe_C; ++col)
        {
            col_offset[col] += col_offset[col - 1];
        }

        int total_edges = col_offset[num_nodes / num_pe_C];

        std::vector<int> row_index(total_edges);

        // Fill the row_index vector with the row indices (neighbors)
        for (int col = 0; col < num_nodes / num_pe_C; ++col)
        {
            int start = col_offset[col];
            for (int i = 0; i < curAdjList[col].size(); ++i)
            {
                row_index[start + i] = curAdjList[col][i];
            }
        }

        // Store the CSC representation for the processor
        csc_vect[proc] = {col_offset, row_index};
    }
}
csc::~csc()
{
}

void  csc::print_info(int r, int c)
{
    std::cout << "\n------CSS INFO------" << std::endl;
    std::cout << "------ " << r << "x" << c << " ------" << std::endl;

    int proc_num = r*num_pe_C + c;

    std::vector<int> col_offset = csc_vect[proc_num].first;
    std::vector<int> row_index = csc_vect[proc_num].second;

    std::cout << "Column offset: " << std::endl;
    for (int i = 0; i < col_offset.size(); i++) 
    {
        std::cout << col_offset[i] << " | ";
    }
    std::cout << std::endl;

    std::cout << "Row index: " << std::endl;
    for (int i = 0; i < row_index.size(); i++) 
    {
        std::cout << row_index[i] << " | ";
    }
    std::cout << std::endl;
}