#!/bin/bash

# cd ~/Desktop/COA_Lab/EndSem_Proj/testing/gpu-rodinia/cuda/hotspot
# source ../gpgpu-sim_distribution-4.0.1/setup_environment
# nvcc -lcudart hotspot.cu
# ./a.out > out.txt
# python3 ../../../clean.py

cd ~/Desktop/COA_Lab/EndSem_Proj/testing
source ../gpgpu-sim_distribution-4.0.1/setup_environment
nvcc -lcudart mmul.cu
./a.out > out_w_ka.txt
python3 clean.py