#!/bin/bash

export LD_LIBRARY_PATH="/home/shlok/gpgpu-sim_distribution/lib/gcc-7.5.0/cuda-11010/release"

source ~/gpgpu-sim_distribution/setup_environment

nvcc -lcudart pathfinder.cu

./a.out 1000 100 20 > output.txt