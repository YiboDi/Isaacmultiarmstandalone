# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase
# from omni.isaac.gym.vec_env import VecEnvMT

env = VecEnvBase(headless=False)

# create task and register task
from multiarm_task import MultiarmTask

# from utils import load_config

# config = load_config()
task = MultiarmTask(name="Cartpole", env = env)
env.set_task(task, backend="torch")

# import stable baselines
# from stable_baselines3 import PPO

# create agent from stable baselines
# model = PPO(
#     "MlpPolicy",
#     env,
#     n_steps=1000,
#     batch_size=1000,
#     n_epochs=20,
#     learning_rate=0.001,
#     gamma=0.99,
#     device="cuda:0",
#     ent_coef=0.0,
#     vf_coef=0.5,
#     max_grad_norm=1.0,
#     verbose=1,
#     tensorboard_log="./cartpole_tensorboard",
# )
# model.learn(total_timesteps=10000)
# model.save("ppo_cartpole")

# env.close()

if __name__ == "__main__":
    sac
