import os
import subprocess

GPU_CONFIG_PATH = os.path.expanduser('~/gpgpu-sim_distribution/configs/tested-cfgs')
DATA_STORE_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/Benchmarking')
CODE_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/testing/pathfinder/pathfinder.cu')
BASH_SCRIPT_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/runCode.sh')

cache_sizes = {"32KB":"N:64:128:4,L:L:m:N:H,S:64:8,8","8MB":"N:64:8192:16,L:L:m:N:H,S:64:8,8"}

for cache_size in cache_sizes:
    print(f"Processing cache size: {cache_size}")
    folder_path = os.path.join(DATA_STORE_PATH, f'{cache_size}_results')
    os.makedirs(folder_path, exist_ok=True)
    for folder in os.listdir(GPU_CONFIG_PATH):
        config_folder_path = os.path.join(GPU_CONFIG_PATH, folder)
        new_folder_path = os.path.join(folder_path, folder)
        os.makedirs(new_folder_path, exist_ok=True)
        if os.path.isdir(config_folder_path):
            print(f"Processing folder: {folder}")
            os.system(f'cp -r {config_folder_path}/* {new_folder_path}/')
            os.system(f'cp {CODE_PATH} {new_folder_path}')
            os.system(f'cp {BASH_SCRIPT_PATH} {new_folder_path}')

            config_file_path = os.path.join(new_folder_path, 'gpgpusim.config')

            with open(config_file_path, 'r') as f:
                lines = f.readlines()
            modified_lines = []
            for line in lines:
                if line.startswith('-gpgpu_cache:dl1  '):
                    modified_lines.append(f'-gpgpu_cache:dl1  {cache_sizes[cache_size]}\n')
                else:
                    modified_lines.append(line)
            with open(config_file_path, 'w') as f:
                f.writelines(modified_lines)
            
            os.chdir(new_folder_path)
            subprocess.run(['bash', 'runCode.sh'])
            print(f"Executed commands in: {new_folder_path}")

print("\nTasks completed for all folders and schedulers.")