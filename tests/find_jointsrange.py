import numpy as np
import os
import argparse

def expert_waypoints_range(expert_root_dir):
    max_values = np.array([0, 0, 0, 0, 0, 0])
    min_values = np.array([0, 0, 0, 0, 0, 0])
    count = 0
    
    # Iterate over all files in the expert_root_dir
    for filename in os.listdir(expert_root_dir):
        # Check if the file has a .npy extension
        if filename.endswith(".npy"):
            # Construct the full path to the file
            expert_path = os.path.join(expert_root_dir, filename)
            
            try:
                # Load the waypoints from the .npy file
                waypoints = np.load(expert_path)
                
                max = np.max(waypoints, axis=0)
                min = np.min(waypoints, axis=0)

                if max.shape!= (6,):
                    max = max.reshape((-1,6))
                    min = min.reshape((-1,6))
                    max = np.max(max, axis=0)
                    min = np.min(min, axis=0)

                max_values = np.maximum(max, max_values)
                min_values = np.minimum(min, min_values)

                count += 1
                print(f"counted {count}")

            except Exception as e:
                print(f"Error loading {expert_path}: {e}")
    
    return max_values, min_values



def main(expert_root_dir):
    max_values, min_values = expert_waypoints_range(expert_root_dir)
    print(f"Max values: {max_values}")
    print(f"Min values: {min_values}")

if __name__ == "__main__":
    expert_root_dir = "/home/dyb/Thesis/expert"
    main(expert_root_dir)
