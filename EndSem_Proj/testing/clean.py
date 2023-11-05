import os

folder_path = "~/Desktop/COA_Lab/EndSem_Proj/testing"

# delete all files which go like : _app_cuda_version_* or _cuobjdump_list_ptx_* or with .ptx or .ptxas extension

def delete_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".ptx") or file.endswith(".ptxas") or file.startswith("_app_cuda_version_") or file.startswith("_cuobjdump_list_ptx_"):
                    os.remove(file_path)
            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read (e.g., binary files) or have permission issues
                pass

delete_files(os.path.expanduser(folder_path))