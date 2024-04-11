#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --gres=gpu:1
#SBATCH --job-name=bfs_cuda

module purge
module load gcc/11.3.0
module load cuda/11.5.0
# module load nvidia-hpc-sdk/22.7

# Compile CUDA source files (.cu)
nvcc -c include/gpu/gpuBFS.cu -o gpuBFS.o

nvcc -c main.cu -o main.o

# Compile non-CUDA source files (.cpp)
g++ -c include/csr/csr.cpp -o csr.o
g++ -c include/cpu/cpuBFS.cpp -o cpuBFS.o

# Link objects to create executable
nvcc -o graphBFS gpuBFS.o cpuBFS.o csr.o main.o

# ncu --set full -o gpu_graph_profile ./graphBFS graphs/europe_osm.txt 47837212

./graphBFS graphs/tree.txt 0
# ./graphBFS graphs/wiki.txt 1485610
# ./graphBFS graphs/europe_osm.txt 47837212

# make -f Makefile_cuda run ARGS="graphs/tree.txt 0"
# make -f Makefile_cuda run ARGS="graphs/wiki.txt 6"
# make -f Makefile_cuda run ARGS="graphs/europe_osm.txt 47837212"