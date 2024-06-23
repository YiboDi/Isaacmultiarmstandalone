import os

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count


directory = "/home/dyb/Thesis/expert" 
print(f"Counting files in directory: {directory}")
number_of_files = count_files(directory)
print(f"Total number of files: {number_of_files}")