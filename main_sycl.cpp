#include "include/csr/csr.h"
#include "include/cpu/cpuBFS.h"
#include "include/gpu_sycl/syclBFS.h"

int main(int argc, char* argv[]) 
{
    if (argc != 3) 
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <source_num>" << std::endl;
        return 1;
    }

    const char* file = argv[1];
    int source = std::stoi(argv[2]);

    csr graph(file);

    std::cout << "Column indices: " << graph.num_edges << std::endl;
    for (int i = 0; i < graph.num_edges; i++) {
        std::cout << graph.col_idx[i] << " | ";
    }
    std::cout << std::endl;

    std::cout << "Row offset: " << graph.num_nodes << std::endl;
    for (int i = 0; i < graph.num_nodes+1; i++) {
        std::cout << graph.row_offset[i] << " | ";
    }
    std::cout << std::endl;

    std::cout << "\nRunning SYCL BFS\n";
    syclBFS syclBFS(graph, source);

    return 0;
}