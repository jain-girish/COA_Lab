import os
BENCHMARK_PATH = os.path.expanduser('~/Desktop/COA_Lab/Assignment4/Benchmarking')

cache_sizes = ["32KB", "8MB"]

for cache_size in cache_sizes:
    folder_path = os.path.join(BENCHMARK_PATH, f'{cache_size}_results')
    print(f'For {cache_size}')
    for config in os.listdir(folder_path):
        print(f'\tFor {config}')
        new_folder_path = os.path.join(folder_path, config)
        if os.path.isdir(new_folder_path):
            output_file_path = os.path.join(new_folder_path, 'output.txt')
            with open(output_file_path, 'r') as f:
                lines = f.readlines()
            f1 = 0
            f2 = 0
            l1d_total_cache_miss_rate = 0
            l2_total_cache_miss_rate = 0
            for line in lines[::-1]:
                if(not f1 and "L1D_total_cache_miss_rate" in line):
                    l1d_total_cache_miss_rate = float(line.split()[-1])
                    f1 = 1
                elif(not f2 and "L2_total_cache_miss_rate" in line):
                    l2_total_cache_miss_rate = float(line.split()[-1])
                    f2 = 1
                if(f1 and f2):
                    break
            print(f'\t\tL1D_total_cache_hit_rate: {1-l1d_total_cache_miss_rate}')
            print(f'\t\tL2_total_cache_hit_rate: {1-l2_total_cache_miss_rate}')
                    