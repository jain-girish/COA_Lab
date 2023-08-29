import os
import subprocess

DATA_DIRECTORY_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/Benchmarking/32KB_results')

for folder in os.listdir(DATA_DIRECTORY_PATH):
    folder_path = os.path.join(DATA_DIRECTORY_PATH, folder)
    if os.path.isdir(folder_path):
        output_file_path = os.path.join(folder_path, 'output.txt')
        #get the path of file which starts with gpgpusim_power_report__ and ends with .log
        power_report_file_path = os.path.join(folder_path, [file for file in os.listdir(folder_path) if file.startswith('gpgpusim_power_report__') and file.endswith('.log')][0])
        with open(output_file_path, 'r') as f:
            lines = f.readlines()
        l1d_total_cache_miss_rate = 0
        for line in lines[::-1]:
            if("L1D_total_cache_miss_rate" in line):
                l1d_total_cache_miss_rate = float(line.split()[-1])
                break
        with open(power_report_file_path, 'r') as f:
            lines = f.readlines()
        kernel_average_power = 0
        for line in lines:
            if("kernel_avg_power" in line):
                kernel_average_power = float(line.split()[-1])
                break
        print(f'{folder}\t{1-l1d_total_cache_miss_rate}\t{kernel_average_power}')