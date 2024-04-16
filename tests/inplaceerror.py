import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/tasks')
# sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/algorithms')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/networks')
import gym
from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
# from multiarmRL.algorithms.SAC import SAC
# from algorithms.SAC import SAC
# from omni.isaac.gym.vec_env import VecEnvMT
# from stable_baselines3 import SAC
from BaseNet import StochasticActor, Q
# from multiarm_task import MultiarmTask
import json


# from BaseNet import create_network
from net_utils import create_testnet

# if __name__ == 'main':
num_episodes = 75000  # Define the number of episodes for testing

env = VecEnvBase()

from multiarm_task import MultiarmTask
task = MultiarmTask(name="Multiarm")
env.set_task(task, backend = 'torch')

file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/config/default.json'
# Load JSON file
with open(file_path, 'r') as file:
    config = json.load(file)
    training_config = config['training']

network = create_testnet()
# network = network.to('cuda')
print(network)
model = SAC(network=network)

for episode in range(num_episodes):
    observations = env.reset()
    done = False
    while not done:
        # Select a random action
        # action = env.action_space.sample()

        #Select an action
        observations = env._task.get_observations() # num_robots * num_robots * 107
        # actions = []
        zero_tensor = torch.zeros((env._task.num_agents, 4-env._task.num_agents, 107), device = 'cuda')
        observations = torch.cat((observations, zero_tensor), dim = 1) # size [n,4,107]

        # for i in range(observations.shape[0]):
        #     observation = observations[i].squeeze(0)
        #     action = model.inference(observation) # get the action based on the current observation
        #     actions.append(action)

        # actions = torch.cat(actions, dim=0)

        actions = model.inference(observations) # input in network has shape of batch_size * seq_len * input_size = num_robots * num_robots * 107
        actions = actions.reshape(env._task.num_agents, 6) # num_robots * num_robots *
        # Step through the environment
        next_observations, rewards, done, info, is_terminals = env.step(actions)
        next_observations = torch.cat((next_observations, zero_tensor), dim = 1)

        # for i in range(next_observations.shape[0]):
        data_dic = {
                'observations' : [row.squeeze(0) for i, row in enumerate(observations)],
                'actions' : [row.squeeze(0) for i, row in enumerate(actions)],
                'rewards' : [row.squeeze(0) for i, row in enumerate(rewards)], 
                'next_observations' : [row.squeeze(0) for i, row in enumerate(next_observations)],
                'is_terminal' : [row for i, row in enumerate(is_terminals)]
            }
        model.replay_buffer.extend(data_dic) # rewards may not be torch tensor
        

        observations = next_observations

        # Optionally print out step information
        print(f"Episode: {episode}, Step: {actions}, Reward: {rewards}")

    print(f"Episode {episode} finished")

env.close()
