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

    num_R = R;
    num_C = C;
    int num_proc = R * C;
    std::vector<std::vector<std::pair<int,int>>> processor_adjList(num_proc);
    std::vector<int> proc_number_rows(num_proc,0);
    std::vector<std::vector<int>> degree_vect(num_proc);
    csc_vect.resize(num_proc);

    int base_cols_per_proc = num_nodes / C;
    int extra_cols = num_nodes % C;
    int base_rows_per_proc = num_nodes / R;
    int extra_rows = num_nodes % R;
    std::vector<int> col_start(C + 1, 0);
    for (int i = 1; i <= C; ++i) 
    {
        col_start[i] = col_start[i - 1] + base_cols_per_proc + (i <= extra_cols ? 1 : 0);
    }

    for (int global_col = 0; global_col < num_nodes; global_col++)
    {
        std::vector<int> cur_node = adjList[global_col];
        
        int c = std::upper_bound(col_start.begin(), col_start.end(), global_col) - col_start.begin() - 1;

        for (int i = 0; i < cur_node.size(); i++)
        {
            int global_row = cur_node[i];
            int r = global_row%R;
            int proc_num = r*C+c;

            // global_col -> global_row in proc r*C+c
            processor_adjList[proc_num].push_back({global_col, global_row});
        }
    }

    for (int r = 0; r < R; r++)
    {
        for (int c = 0; c < C; c++)
        {
            int proc_num = r*C+c;
            int proc_num_cols = col_start[c + 1] - col_start[c];
            int proc_num_rows;
            if (r < extra_rows)
            {
                proc_num_rows = (base_rows_per_proc + 1);
            }
            else
            {
                proc_num_rows = base_rows_per_proc;
            }

            std::vector<std::pair<int,int>> proc_edges = processor_adjList[proc_num];
            std::vector<int> col_offset;
            std::vector<int> row_index;

            if (proc_edges.size() > 0)
            {
                col_offset.resize(proc_num_cols+1,-1);
                int count = 0;
                int offset_count = 0;

                for (int local_col = 0; local_col < proc_num_cols; local_col++)
                {
                    int global_col = col_start[c] + local_col;


                    col_offset[local_col] = offset_count;
                    
                    // find all source nodes in adj list that are equal to global_col and place in local_col
                    for (int edge = 0; edge < proc_edges.size(); edge++)
                    {
                        if (proc_edges[edge].first == global_col)
                        {
                            row_index.push_back(proc_edges[edge].second);
                            count++;
                            offset_count++;
                        }
                    }

                    if (count == 0)
                    {
                        col_offset[local_col] = -1;
                    }


                    count = 0;
                }

                col_offset[proc_num_cols] = offset_count;

                // change -1's to next offset
                for (int i = proc_num_cols; i >= 0; i--) 
                {
                    if (col_offset[i] == -1)
                    {
                        col_offset[i] = col_offset[i+1];
                    }
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

    int proc_num = r*num_C + c;

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