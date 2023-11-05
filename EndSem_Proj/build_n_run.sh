#!/bin/bash

cd ~/Desktop/COA_Lab/EndSem_Proj/gpgpu-sim_distribution-4.0.1
source setup_environment
make

cd ..
chmod +x run.sh
./run.sh

# cd ~/Desktop/COA_Lab/EndSem_Proj/gpgpu-sim_distribution-4.0.1/testing
# source ../setup_environment
# nvcc -lcudart pathfinder.cu
# ./a.out 100 100 20 > out.txt