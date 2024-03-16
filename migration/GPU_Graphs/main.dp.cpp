#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "include/csr/csr.h"
#include "include/cpu/cpuBFS.h"
#include "include/gpu/gpuBFS.dp.hpp"
#include <ctime>

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

    std::cout << "\nRunning CPU BFS\n";
    cpuBFS cpuBFS(graph, source);

    // std::cout << "Distance vector: " << std::endl;
    // for (int i = 0; i < graph.num_nodes; i++) {
    //     std::cout << cpuBFS.distance[i] << " | ";
    // }
    std::cout << std::endl;

    std::cout << "\nRunning GPU BFS\n";
    gpuBFS gpuBFS(graph, source);

    // std::cout << "Distance vector: " << std::endl;
    // for (int i = 0; i < graph.num_nodes; i++) {
    //     std::cout << gpuBFS.host_distance[i] << " | ";
    // }
    std::cout << std::endl;

    // double cpu_time = (end_cpu - start_cpu) / (double)CLOCKS_PER_SEC;
    std::cout << "CPU BFS Time: " << cpuBFS.exec_time << " ms" << std::endl;

    // double gpu_time = (end_gpu - start_gpu) / (double)CLOCKS_PER_SEC;
    std::cout << "GPU BFS Time: " << gpuBFS.exec_time << " ms" << std::endl;

    for (int i = 0; i < graph.num_nodes; i++) 
    {
        if (gpuBFS.host_distance[i] != cpuBFS.distance[i])
        {
            std::cout << "mismatch at node " << i+1 << std::endl;
            std::cout << "cpu: " << cpuBFS.distance[i] << std::endl;
            std::cout << "gpu: " << gpuBFS.host_distance[i] << std::endl;
            break;
        }
        
    }

    return 0;
}