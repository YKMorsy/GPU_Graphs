#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_nvlink
#SBATCH --job-name=gpuBFS

module purge
module load gcc/8.2.0
module load cuda/11.0
#module load openmpi/4.0.5
module load nvshmem/2.6.0
module load hydra/3.2

NVSHMEM_HOME="/ihome/crc/install/nvshmem/2.6.0/gcc-8.2.0"
MPI_HOME="/ihome/crc/install/openmpi/4.0.5"
CUDA_HOME="/ihome/crc/install/cuda/11.0"
NVCC_GENCODE="-gencode arch=compute_70,code=[compute_70,sm_70]"
NVCCFLAGS="-rdc=true -ccbin g++ -lcuda -lnvshmem_host -lnvshmem_device --expt-relaxed-constexpr"

# Compile CUDA source files (.cu)
nvcc $NVCCFLAGS $NVCC_GENCODE -I$NVSHMEM_HOME/include -c include/gpu/gpuBFS_kernels.cu -o gpuBFS_kernels.o -L$NVSHMEM_HOME/lib
nvcc $NVCC_GENCODE -I$NVSHMEM_HOME/include -c include/gpu/gpuBFS.cu -o gpuBFS.o -L$NVSHMEM_HOME/lib
nvcc $NVCC_GENCODE -I$NVSHMEM_HOME/include -c main.cu -o main.o -L$NVSHMEM_HOME/lib

# Compile non-CUDA source files (.cpp)
g++ -fPIC -c include/csr/csr.cpp -o csr.o
g++ -fPIC -c include/cpu/cpuBFS.cpp -o cpuBFS.o

# Link objects to create executable
nvcc $NVCC_GENCODE -I$NVSHMEM_HOME/include -o graphBFS gpuBFS.o gpuBFS_kernels.o cpuBFS.o csr.o main.o -L$NVSHMEM_HOME/lib $NVCCFLAGS
# nvcc $NVCC_GENCODE -I$NVSHMEM_HOME/include -o graphBFS main.o -L$NVSHMEM_HOME/lib $NVCCFLAGS

# nvcc -rdc=true -ccbin g++ $NVCC_GENCODE -I$NVSHMEM_HOME/include main.cu -o graphBFS -L$NVSHMEM_HOME/lib -lcuda -lnvshmem_host -lnvshmem_device
# export NVSHMEM_SYMMETRIC_SIZE=30g

# Run the executable
# mpirun -np 2 ./graphBFS graphs/tree.txt 0
# mpirun -np 2 ./graphBFS graphs/test.txt 0
# mpirun -np 2 ./graphBFS graphs/test2.txt 0
# mpirun -np 2 ./graphBFS graphs/test3.txt 0
# mpirun -np 2 ./graphBFS graphs/test4.txt 0
# mpirun -np 2 ./graphBFS graphs/test_isolated_node.txt 1
# mpirun -np 2 ./graphBFS graphs/test_isolated_node.txt 0
# mpirun -np 2 ./graphBFS graphs/wiki.txt 5
mpirun -np 1 ./graphBFS graphs/europe_osm.txt 47837211
