import os, sys

path = os.getcwd() + 'tasks/'
# path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/tasks'
print(f'********{sys.path}***************')
sys.path.append(path)