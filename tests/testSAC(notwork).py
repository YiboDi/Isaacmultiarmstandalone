import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/tasks')
# sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/algorithms')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/networks')
import gym
from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
# from omni.isaac.gym.vec_env import VecEnvMT
# from stable_baselines3 import SAC
from BaseNet import StochasticActor, Q
# from multiarm_task import MultiarmTask
import json

def create_env():
    env = VecEnvBase(headless=False)
    return env

# create task and register task
# from multiarm_task import MultiarmTask

def create_task(env):
    task = MultiarmTask(name="Multiarm")
    env.set_task(task, backend="torch")

def create_network(training_config, actor_obs_dim = 107, action_dim = 6, critic_obs_dim = 107):
    policy_net = StochasticActor(
            obs_dim=actor_obs_dim,
            action_dim=action_dim,
            action_variance_bounds=training_config['action_variance'],
            network_config=training_config['network']['actor'])
    
    Q1 = Q(obs_dim=critic_obs_dim + 6
                 if training_config['centralized_critic'] # default to be false
                 else critic_obs_dim, 
                 action_dim=action_dim,
                 network_config=training_config['network']['critic'])
    
    Q2 = Q(obs_dim=critic_obs_dim + 6
                 if training_config['centralized_critic'] # default to be false
                 else critic_obs_dim, 
                 action_dim=action_dim,
                 network_config=training_config['network']['critic'])
    
    network = {
        'policy':policy_net,
        'Q1' : Q1,
        'Q2' : Q2,
    }

    return network

def create_training_model(network):
    model = SAC(network)   
    return model



if __name__ == 'main':
    num_episodes = 10  # Define the number of episodes for testing

    env = create_env()

    from multiarm_task import MultiarmTask

    env = create_task(env=env)

    file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/config/default.json'
    # Load JSON file
    with open(file_path, 'r') as file:
        config = json.load(file)
        training_config = config['training']

    network = create_network(training_config=training_config)
    model = create_training_model(network=network)

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
