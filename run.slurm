#!/bin/bash

#SBATCH --job-name=sycl_BFS      
#SBATCH --nodes=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --partition=pvc

module purge
module load intel-compilers/2023.2.1

make run ARGS="graphs/europe_osm.txt 47837212"