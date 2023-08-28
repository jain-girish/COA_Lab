#!/bin/bash

# Set environment variables
export LD_LIBRARY_PATH="/home/shlok/gpgpu-sim_distribution/lib/gcc-7.5.0/cuda-11010/release"

# Path to the directory containing the code
# PATH="~/Desktop/COA_Lab/Assignment4/testing/pathfinder"

# Change directory
# cd "$path"
# cd "${PATH}"

# Source setup_environment script
source ~/gpgpu-sim_distribution/setup_environment

# Compile CUDA code
nvcc -lcudart pathfinder.cu

# Run the executable
./a.out 1000 100 20 > output.txt