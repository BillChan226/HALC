import os

def count_files_in_folder(folder_path):
    """Counts the number of files in the given folder."""
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return 0

    total_files = 0
    for entry in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, entry)):
            total_files += 1

    return total_files

# Specify the folder path here
folder_path = '/home/czr/contrast_decoding_LVLMs/eval_dataset/val2014/'

# Call the function and print the result
file_count = count_files_in_folder(folder_path)
print(f"Number of files in '{folder_path}': {file_count}")
