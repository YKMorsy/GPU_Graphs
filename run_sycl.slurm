#!/bin/bash

#SBATCH --job-name=sycl_BFS      
#SBATCH --nodes=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=pvc

module purge
module load intel-compilers/2023.2.1

make -f Makefile_sycl run ARGS="graphs/tree.txt 0"
make -f Makefile_sycl run ARGS="graphs/wiki-Vote.txt 3"
make -f Makefile_sycl run ARGS="graphs/europe_osm.txt 47837212"