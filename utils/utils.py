import os
def ensure_dir_exists(directory_path: str):
    """
    Check if a directory exists, and create it if it does not.

    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
