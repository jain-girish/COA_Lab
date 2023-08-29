import os
import subprocess

GPU_CONFIG_PATH = os.path.expanduser('~/gpgpu-sim_distribution/configs/tested-cfgs')
DATA_STORE_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/Benchmarking')
CODE_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/testing/pathfinder/pathfinder.cu')
BASH_SCRIPT_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/runCode.sh')

schedulers = ['gto', 'lrr', 'two_level_active:6:0:1']

for folder in os.listdir(GPU_CONFIG_PATH):
    folder_path = os.path.join(GPU_CONFIG_PATH, folder)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder}")
        
        for scheduler in schedulers:
            scheduler_folder = os.path.join(DATA_STORE_PATH, f'{scheduler}_results')
            os.makedirs(scheduler_folder, exist_ok=True)
            
            new_folder_path = os.path.join(scheduler_folder, folder)
            os.makedirs(new_folder_path, exist_ok=True)
            
            print(f"  - Processing scheduler: {scheduler}")
            print(f"    Created folder: {new_folder_path}")
            
            os.system(f'cp -r {folder_path}/* {new_folder_path}/')
            print(f"    Copied GPU Configs to: {new_folder_path}")
            
            os.system(f'cp {CODE_PATH} {new_folder_path}')
            print(f"    Copied CUDA file to: {new_folder_path}")

            os.system(f'cp {BASH_SCRIPT_PATH} {new_folder_path}')
            
            config_file_path = os.path.join(new_folder_path, 'gpgpusim.config')
            
            print(f"    Modifying config file for scheduler: {scheduler}")
            with open(config_file_path, 'r') as f:
                lines = f.readlines()
            modified_lines = []
            for line in lines:
                if line.startswith('-gpgpu_scheduler'):
                    modified_lines.append(f'-gpgpu_scheduler {scheduler}\n')
                else:
                    modified_lines.append(line)
            with open(config_file_path, 'w') as f:
                f.writelines(modified_lines)
            
            print("    Running commands...")
            os.chdir(new_folder_path)
            
            subprocess.run(['bash', 'runCode.sh'])

            print(f"    Executed commands in: {new_folder_path}")

print("\nTasks completed for all folders and schedulers.")
