#include "../include/csc.h"
#include "../include/cpuBFS.h"
#include "../include/gpuBFS.cuh"

#include <ctime>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <source_num>" << std::endl;
        return 1;
    }

    const char* file = argv[1];
    int source = std::stoi(argv[2]);

    // csc graph(file, 2, 2);
    // graph.print_info(0,0);
    // graph.print_info(0,1);
    // graph.print_info(1,0);
    // graph.print_info(1,1);

    // csc graph(file, 1, 2);
    // graph.print_info(0,0);
    // graph.print_info(0,1);

    csc graph(file, 1, 1);
    graph.print_info(0,0);

    cpuBFS cpuBFS(graph, 0);
    cpuBFS.print_distance(graph);

    gpuBFS gpuBFS(graph, 0, 1, 1);
    gpuBFS.print_distance(graph);

    std::cout << "CPU TIME: " << cpuBFS.exec_time << " Seconds" << std::endl;
    std::cout << "GPU TIME: " << gpuBFS.exec_time << " Seconds" << std::endl;

    int num_mismatch = 0;
    for (int i = 0; i < graph.num_nodes; i++) 
    {

        if (gpuBFS.host_distance[i] != cpuBFS.distance[i])
        {
            num_mismatch++;
        }
        
    }
    std::cout << "Number of mismatches: " << num_mismatch << std::endl;

    return 0;
}