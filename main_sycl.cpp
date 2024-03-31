#include "include/csr/csr.h"
#include "include/cpu/cpuBFS.h"
#include "include/gpu_sycl/syclBFS.h"
#include<unistd.h>

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

    std::cout << "Num Nodes: " << graph.num_nodes << std::endl;
    std::cout << "Num Edges: " << graph.num_edges << std::endl;

    // std::cout << "Column indices: " << graph.num_edges << std::endl;
    // for (int i = 0; i < graph.num_edges; i++) {
    //     std::cout << graph.col_idx[i] << " | ";
    // }
    // std::cout << std::endl;

    // std::cout << "Row offset: " << graph.num_nodes << std::endl;
    // for (int i = 0; i < graph.num_nodes+1; i++) {
    //     std::cout << graph.row_offset[i] << " | ";
    // }
    // std::cout << std::endl;

    std::cout << "\nRunning CPU BFS\n\n";
    cpuBFS cpuBFS(graph, source);
    std::cout << "Distance vector: " << std::endl;
    for (int i = 0; i < graph.num_nodes; i++) {
        std::cout << cpuBFS.distance[i] << " | ";
    }
    std::cout << std::endl;

    std::cout << "\nRunning SYCL BFS\n\n";
    std::cout.flush();
    syclBFS syclBFS(graph, source);
    // std::cout << std::endl;

    std::cout << "Distance vector: " << std::endl;
    for (int i = 0; i < graph.num_nodes; i++) {
        std::cout << syclBFS.host_distance[i] << " | ";
    }
    std::cout << std::endl;

    for (int i = 0; i < graph.num_nodes; i++) 
    {
        if (syclBFS.host_distance[i] != cpuBFS.distance[i])
        {
            std::cout << "mismatch at node " << i+1 << std::endl;
            std::cout << "cpu: " << cpuBFS.distance[i] << std::endl;
            std::cout << "gpu: " << syclBFS.host_distance[i] << std::endl;
            break;
        }
        
    }

    return 0;
}