import numpy as np


expert_root_dir = '/home/tp2/papers/multiarm_dataset/expert/'

task_id = '000001'
expert_path = expert_root_dir + task_id + ".npy"

rrt_waypoints = np.load(expert_path)

print(rrt_waypoints)

