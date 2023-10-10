#!/bin/bash

cd ~/gpgpu-sim_distribution
source setup_environment
make

cd ~/Desktop/COA_Lab/MidSem_Proj/testing
source ~/gpgpu-sim_distribution/setup_environment
nvcc -lcudart pathfinder.cu
./a.out 100 10 20 > out.txt
python3 analysis.py