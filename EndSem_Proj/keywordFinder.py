import os


folder_path = "~/Desktop/COA_Lab/EndSem_Proj/gpgpu-sim_distribution-4.0.1/src"

def search_files_for_string(folder_path, target_string):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if target_string in content:
                        print(f"String found in file: {file_path}")
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read (e.g., binary files) or have permission issues
                pass

# Example usage: searching for the string "example" in files in the folder "path/to/folder"
target_string = "->step"
search_files_for_string(os.path.expanduser(folder_path), target_string)
