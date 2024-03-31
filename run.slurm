#!/bin/bash

#SBATCH --job-name=sycl_BFS      
#SBATCH --nodes=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=pvc

module purge
module load intel-compilers/2023.2.1

make run ARGS="graphs/tree.txt 0"
make run ARGS="graphs/wiki-Vote.txt 3"
make run ARGS="graphs/europe_osm.txt 47837212"