#!/bin/bash

#SBATCH --job-name=sycl_BFS      
#SBATCH --nodes=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --partition=pvc

module load intel-compilers/2024.0.0
module load intel/2023b

make run ARGS="graphs/tree.txt 0"