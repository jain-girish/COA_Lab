#!/bin/bash

cd ~/Desktop/COA_Lab/EndSem_Proj/testing
source ../gpgpu-sim_distribution-4.0.1/setup_environment
nvcc -lcudart mmul.cu
./a.out > out.txt
python3 clean.py