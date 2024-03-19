#!/bin/bash

#SBATCH --job-name=sycl_demo      
#SBATCH --nodes=1
#SBATCH --gres=gpu:pvc:1
#SBATCH --partition=pvc

module load intel-compilers/2024.0.0

dpcpp -o hello_world hello_world.cpp
./hello_world