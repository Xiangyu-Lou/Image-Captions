import os

def get_file_names(folder_path):
    # List all files in the given folder
    file_names = []
    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isfile(full_path):
            file_names.append(item)
    return file_names

# Example usage
folder_path = 'your/folder/path'  # Replace with the actual path
file_list = get_file_names(folder_path='data/train2017')
print(len(file_list))