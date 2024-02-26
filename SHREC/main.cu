#include "include/csr/csr.h"
#include "include/cpu/cpuBFS.h"

#include <cstdlib> // for std::stoi

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <source_num>" << std::endl;
        return 1;
    }

    const char* file = argv[1];
    int source = std::stoi(argv[2]);

    csr graph(file);
    std::vector<int> col_idx = graph.col_idx;
    std::vector<int> row_offset = graph.row_offset;

    cpuBFS cpuBFS(col_idx, row_offset, source);

    // std::cout << "Distance vector:" << std::endl;
    // for (int i = 0; i < cpuBFS.distance.size(); ++i) {
    //     std::cout << cpuBFS.distance[i] << " | ";
    // }
    // std::cout << std::endl;

    // std::cout << "Column indices:" << std::endl;
    // for (int i = 0; i < col_idx.size(); ++i) {
    //     std::cout << col_idx[i] << " | ";
    // }
    // std::cout << std::endl;

    // std::cout << "Row offset:" << std::endl;
    // for (int i = 0; i < row_offset.size(); ++i) {
    //     std::cout << row_offset[i] << " | ";
    // }
    // std::cout << std::endl;

    return 0;
}