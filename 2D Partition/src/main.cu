#include "../include/csc.h"
#include "../include/cpuBFS.h"
#include "../include/gpuBFS.cuh"

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cmath>

int main(int argc, char* argv[]) 
{
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <source_num>" << std::endl;
        return 1;
    }

    const char* file = argv[1];
    int source = std::stoi(argv[2]);
    int num_rows = std::stoi(argv[3]);
    int num_cols = std::stoi(argv[4]);

    // csc graph_cpu(file, 1, 1);
    // cpuBFS cpuBFS(graph_cpu, 0);
    // cpuBFS.print_distance(graph_cpu);

    csc graph_gpu(file, num_rows, num_cols);
    // graph_gpu.print_info(0,0);
    // graph_gpu.print_info(0,1);
    // graph_gpu.print_info(1,0);
    // graph_gpu.print_info(1,1);
    gpuBFS gpuBFS(graph_gpu, source, num_rows, num_cols);
    // gpuBFS.print_distance(graph_gpu);

    // std::cout << "CPU TIME: " << cpuBFS.exec_time << " Seconds" << std::endl;
    std::cout << gpuBFS.exec_time << " Seconds" << std::endl;

    // int num_mismatch = 0;
    // for (int i = 0; i < graph.num_nodes; i++) 
    // {

    //     if (gpuBFS.host_distance[i] != cpuBFS.distance[i])
    //     {
    //         num_mismatch++;
    //     }
        
    // }
    // std::cout << "Number of mismatches: " << num_mismatch << std::endl;

    return 0;
}