import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/networks')

import gym
from vec_env_base_custom import VecEnvBase
import torch
# from omni.isaac.gym.vec_env import VecEnvMT
from stable_baselines3 import SAC

from lstm_torch_layers import LSTMExtractor

env = VecEnvBase(headless=False)

# create task and register task
# from Di_custom.multiarmRL.tasks.multiarm_task import MultiarmTask
from multiarm_task import MultiarmTask

task = MultiarmTask(name="Multiarm")
env.set_task(task, backend="torch")

model = SAC('MlpPolicy',
            env = env,
            policy_kwargs={"features_extractor_class": LSTMExtractor, 
                        #    "features_extractor_kwargs": {"features_dim": 64}
                           })

model.learn(total_timesteps=100000)
model.save()
