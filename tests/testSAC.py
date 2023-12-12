import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/algorithms')

import gym
from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
# from omni.isaac.gym.vec_env import VecEnvMT
# from stable_baselines3 import SAC

env = VecEnvBase(headless=False)

# create task and register task
from Di_custom.multiarmRL.tasks.multiarm_task import MultiarmTask

task = MultiarmTask(name="Multiarm")
env.set_task(task, backend="torch")

model = SAC()


num_episodes = 10  # Define the number of episodes for testing

for episode in range(num_episodes):
    observations, actions = env.reset()
    done = False
    while not done:
        # Select a random action
        # action = env.action_space.sample()

        #Select an action
        observations = env.get_observations()
        actions = []

        for i in range(observations.shape[0]):
            observation = observations[i].squeeze(0)
            action = model.inference(observation) # get the action based on the current observation
            actions.append(action)

        actions = torch.cat(actions, dim=0)

        # Step through the environment
        next_observations, rewards, done, info = env.step(actions)

        # for i in range(next_observations.shape[0]):
        data_dic = {
                'obsercations' : [row.squeeze(0) for i, row in enumerate(observations)],
                'actions' : [row.squeeze(0) for i, row in enumerate(actions)],
                'rewards' : [row.squeeze(0) for i, row in enumerate(rewards)], 
                'next_observations' : [row.squeeze(0) for i, row in enumerate(next_observations)]
            }
        model.replay_buffer.extend(data_dic) # rewards may not be torch tensor
        

        observations = next_observations

        # Optionally print out step information
        print(f"Episode: {episode}, Step: {action}, Reward: {reward}")

    print(f"Episode {episode} finished")

env.close()
