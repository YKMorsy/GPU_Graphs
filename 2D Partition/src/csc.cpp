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

    int from, to;
    std::vector<std::vector<int>> adjList(num_nodes+1);
    while (infile >> from >> to)
    {
        adjList[from-1].push_back(to-1);
    }
    infile.close(); // Close the file after reading

    num_pe_R = R;
    num_pe_C = C;
    int num_proc = R * C;

    // create adj list for each processor
    std::vector<int> total_edges_arr(num_proc);
    std::vector<std::vector<std::vector<int>>> processor_adjList(num_proc);
    csc_vect.resize(num_pe_R*num_pe_C);

    for (int p_i = 0; p_i < num_pe_R; p_i++)
    {
        for (int p_j = 0; p_j < num_pe_C; p_j++)
        {
            int vertex_block = p_j*num_pe_R + p_i; // 0-0, 1-2, 2-1, 3-3
            int nodes_per_col = num_nodes/num_pe_C; // 4
            int start_col_node = (vertex_block/num_pe_C) * nodes_per_col;
            int total_edges = 0;

            std::vector<std::vector<int>> cur_adj_list(nodes_per_col);

            for (int node_col = 0; node_col < nodes_per_col; node_col++)
            {                
                std::vector<int> col_shared_edges = adjList[node_col+start_col_node];

                int num_neighbors = 0;

                for (int neighbor_index = 0; neighbor_index < col_shared_edges.size(); neighbor_index++)
                {
                    int neighbor = col_shared_edges[neighbor_index];

                    int neighbor_i = (neighbor/(num_nodes/(num_pe_R*num_pe_C)))%num_pe_R;

                    if (p_i == neighbor_i)
                    {
                        cur_adj_list[node_col].push_back(neighbor);
                        num_neighbors++;
                        total_edges++;
                    }
                }

                if (num_neighbors == 0)
                {
                    cur_adj_list[node_col].push_back(-1);
                }
            }

            processor_adjList[p_i*num_pe_C+p_j] = cur_adj_list;
            total_edges_arr[p_i*num_pe_C+p_j] = total_edges;

        }
    }
    
    // create csc
    for (int p_i = 0; p_i < num_pe_R; p_i++)
    {
        for (int p_j = 0; p_j < num_pe_C; p_j++)
        {
            int vertex_block = p_j*num_pe_R + p_i; // 0-0, 1-2, 2-1, 3-3
            int nodes_per_col = num_nodes/num_pe_C; // 4
            int start_col_node = (vertex_block/num_pe_C) * nodes_per_col;
            int proc_num = p_i*num_pe_C+p_j;

            std::vector<int> col_offset;
            std::vector<int> row_index;
            std::vector<std::vector<int>> cur_adj_list = processor_adjList[proc_num];

            if (total_edges_arr[proc_num] != 0)
            {
                int cur_count = 0;

                for (int i = 0; i < cur_adj_list.size(); i++)
                {
                    col_offset.push_back(cur_count);

                    for (int j = 0; j < cur_adj_list[i].size(); j++)
                    {
                        if (cur_adj_list[i][j] != -1)
                        {
                            row_index.push_back(cur_adj_list[i][j]);
                            cur_count++;
                        }
                    }     
                }
                col_offset.push_back(cur_count);
            }
            else
            {
                for (int i = 0; i < (num_nodes/(num_pe_C)) + 1; i++)
                {
                    col_offset.push_back(0);
                }
            }

            csc_vect[proc_num] = {col_offset, row_index};
        }
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