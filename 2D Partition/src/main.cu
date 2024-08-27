#include "../include/csc.h"

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

    csc graph(file);
    graph.print_info();




    return 0;
}