import os
import subprocess

# /home/shlok/gpgpu-sim_distribution/lib/gcc-7.5.0/cuda-11010/release/libcudart.so.11.0
# /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcudart.so.11.0


bash_script_path = os.path.abspath('benchmark.sh')
# Execute the Bash script
subprocess.run(['bash', 'benchmark.sh'])
