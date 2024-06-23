import os
import json
import glob

def extract_target_eff_poses(data):
    return [pose[:][0] for pose in data["target_eff_poses"]]

def extract_base_poses(data):
    return [pose[:][0] for pose in data["base_poses"]]  # Take only x and y

def extract_start_config(data):
    return [pose[:] for pose in data["start_config"]]

def extract_goal_config(data):
    return [pose[:] for pose in data["goal_config"]]

def update_min_max(values, min_values, max_values):
    for i, value in enumerate(values):
        min_values[i] = min(min_values[i], value)
        max_values[i] = max(max_values[i], value)

def find_min_max(json_files):
    min_target_eff_poses = [-1.6, -1.6, 0.05]
    max_target_eff_poses = [1.6, 1.6, 0.87]
    min_base_poses = [-0.8, -0.8, 0]
    max_base_poses = [0.8, 0.8, 0]
    min_joint_config = [-6.1, -3.2, 0.8, -2.5, -3.5, -1.5]
    max_joint_config = [0.8, 0.8, 2.8, 0.7, 1.4, 1.4]

    for i, json_file in enumerate(json_files):

        with open(json_file, 'r') as f:
            data = json.load(f)
        
        target_eff_poses = extract_target_eff_poses(data)
        base_poses = extract_base_poses(data)
        start_config = extract_start_config(data)
        goal_config = extract_goal_config(data)

        for pose in target_eff_poses:
            update_min_max(pose, min_target_eff_poses, max_target_eff_poses)
        
        for pose in base_poses:
            update_min_max(pose, min_base_poses, max_base_poses)

        for config in start_config:
            update_min_max(pose, min_joint_config, max_joint_config)

        for config in goal_config:
            update_min_max(pose, min_joint_config, max_joint_config)

        print(i)

    return min_target_eff_poses, max_target_eff_poses, min_base_poses, max_base_poses, min_joint_config, max_joint_config

def main(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    json_files = [f for f in json_files if os.path.basename(f) != 'config.json']
    min_target_eff_poses, max_target_eff_poses, min_base_poses, max_base_poses, min_joint_config, max_joint_config = find_min_max(json_files)

    print("Min Target Effector Poses:", min_target_eff_poses)
    print("Max Target Effector Poses:", max_target_eff_poses)
    print("Min Base Poses:", min_base_poses)
    print("Max Base Poses:", max_base_poses)
    print("Min joint Config:", min_joint_config)
    print("Max joint Config:", max_joint_config)

if __name__ == "__main__":
    directory = "/home/dyb/Thesis/tests"  # Update this with your directory path
    main(directory)