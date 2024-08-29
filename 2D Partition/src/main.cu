#include "../include/csc.h"
#include "../include/cpuBFS.h"

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





    return 0;
}