import os, sys

path = os.getcwd() + 'tasks/'
# path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/tasks'
print(f'********{sys.path}***************')
sys.path.append(path)