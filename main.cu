#include "include/csr/csr.h"
#include "include/cpu/cpuBFS.h"
#include "include/gpu/gpuBFS.cuh"

#include <cstdlib>

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
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

    // std::cout << graph.num_nodes << std::endl;

    cpuBFS cpuBFS(graph, source);

    std::cout << "Distance vector: " << std::endl;
    for (int i = 0; i < graph.num_nodes; i++) {
        std::cout << cpuBFS.distance[i] << " | ";
    }
    std::cout << std::endl;

    // std::cout << "HI\n";

    // gpuBFS gpuBFS(graph, source);




    return 0;
}