#include "include/csr/csr.h"
#include "include/cpu/cpuBFS.h"
#include "include/serialized/serialized.h"
#include "include/gpu/gpuBFS.cuh"

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

    csr graph(file);
    // graph.print_info();

    cpuBFS cpuBFS(graph, source);
    // cpuBFS.print_distance(graph);

    cpuBFS.cpuBFSAlt(graph, source);

    cpuBFS.testDistance(graph);

    gpuBFS gpuBFS(graph, source);  
    // gpuBFS.print_distance(graph);     

    std::cout << "\nCPU BFS Time: " << cpuBFS.exec_time << " ms" << std::endl;

    std::cout << nvshmem_my_pe() << " GPU BFS Time: " << gpuBFS.exec_time << " ms" << std::endl;

    // std::cout << "CPU BFS TE: " << cpuBFS.total_edges_traversed << " TE" << std::endl;

    // std::cout << "GPU BFS TE: " << gpuBFS.total_edges_traversed << " TE" << std::endl;

    // std::cout << "CPU BFS TEPS: " << (cpuBFS.total_edges_traversed/cpuBFS.exec_time)*1000 << " TEPS" << std::endl;

    // std::cout << "GPU BFS TEPS: " << (gpuBFS.total_edges_traversed/gpuBFS.exec_time)*1000 << " TEPS" << std::endl;

    int num_mismatch = 0;

    if (nvshmem_my_pe() == 0)
    {

        for (int i = 0; i < graph.num_nodes; i++) 
        {

            if (gpuBFS.host_distance[i] != cpuBFS.distance[i])
            {
                // std::cout << "mismatch at node " << i+1;
                // std::cout << " cpu: " << cpuBFS.distance[i];
                // std::cout << " gpu: " << gpuBFS.host_distance[i] << " | ";
                num_mismatch++;
                // break;
            }

            // if (serialized.distance[i] != cpuBFS.distance[i])
            // {
            //     // std::cout << "mismatch at node " << i+1;
            //     // std::cout << " cpu: " << cpuBFS.distance[i];
            //     // std::cout << " ser: " << serialized.distance[i] << " | ";
            //     num_mismatch++;
            //     // break;
            // }
            
        }

        std::cout << "Number of mismatches: " << num_mismatch << std::endl;

    }

    // nvshmem_barrier_all();
    // nvshmem_finalize();
    // gpuBFS.~gpuBFS();

    return 0;
}