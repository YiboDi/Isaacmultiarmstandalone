import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/algorithms')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/networks')
import gym
# from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
# from Supervision import supervision
from BaseNet import StochasticActor, Q
import json
# from expertSupervisionEnv import expertSupervisionEnv, angle
from expertmultiEnv import expertmultiEnv, angle
from numpy.linalg import norm
import numpy as np
import os
from tensorboardX import SummaryWriter


# from BaseNet import create_network
from net_utils import create_lstm

num_episodes = 75000  # Define the number of episodes for testing
# rather than define a number, num_episode should be up to number of tasks used as training data
training_data = os.listdir('/home/tp2/papers/multiarm_dataset/tasks')
num_episodes = len(training_data)*2-5

# env = expertSupervisionEnv()
env = expertmultiEnv()

# from multiarm_task import MultiarmTask
from multiarm_with_supervision import MultiarmSupervision
# task = MultiarmSupervision(name="MultiarmSupervision")
from multiarm_paraenvs import MultiarmTask
task = MultiarmTask(name="MultiarmParaenvs", env=env)
env.set_task(task, backend = 'torch')

file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/config/default.json'
# Load config JSON file
with open(file_path, 'r') as file:
    config = json.load(file)
    training_config = config['training']

network = create_lstm(training_config=training_config)
# print(network)
# modify for each experiment
experiment_name = '02.12test'

experiment_dir = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRLdata/experiments/' + experiment_name
log_dir = experiment_dir + '/logs'
# checkpoint_dir = experiment_dir + '/checkpoints'
model = SAC(network=network, experiment_dir=experiment_dir,
            # load_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/experiments/01.24/checkpoints/ckpt_sac_lstm_00337'
            )
writer = SummaryWriter(log_dir=log_dir)

# torch.autograd.set_detect_anomaly(True)

for episode in range(num_episodes):
    observations = env.reset()
    reset = False # flag indicate the reset

    mode = env.mode
    cumulative_reward = torch.zeros(env._task._num_envs, device='cuda')
    end = torch.zeros(env._task._num_envs, device='cuda')

    while not reset: # loop in one episode

        # mode = env.mode

        observations = env._task.get_observations() # num_robots * num_robots * 107
        # with normal mode, take an action which NN output.
        if env._task.mode == 'normal':
            # print(env._task.current_task.id)
            if observations.dim()==4: # num_envs, num_agnets, num_agents, num_obs_per_agent
                observations = observations.reshape(-1, *observations.shape[2:])
            actions = model.inference(observations) # input in network has shape of batch_size * seq_len * input_size = num_robots * num_robots * 107
            # shape of actions is (batch size, 6)
            # dispatch the batch size back to (num_envs, num_robots)
            actions = actions.reshape(env._task._num_envs, env._task.num_agents, *actions.shape[1:])


        # with supervision mode, take an action based on expert_waypoints
        elif env._task.mode == 'supervision':
            actions = env.act_experts()
            
        # Step through the environment
        # actions_reshaped = actions.reshape(env._task._num_envs, env._task.num_agents, *actions.shape[1:])
        next_observations, rewards, reset, info, is_terminals, dones = env.step(actions)

        
        # data_dic = {
        #         'observations' : [row for i, row in enumerate(observations)], # observation should be 1*n*107
        #         'actions' : [row for i, row in enumerate(actions)],
        #         'rewards' : [row for i, row in enumerate(rewards)], # each robot has a reward, size is (n,)
        #         'next_observations' : [row for i, row in enumerate(next_observations)],
        #         'is_terminal' : [row for i, row in enumerate(is_terminals)]
        #     }

        # for multi envs setting :
        data_dic = {
            'observations' : [obs_agent for obs_agents in observations if obs_agents not in end_indices for obs_agent in obs_agents], # here n*107
            'actions' : [act_agent for act_agents in actions if act_agents not in end_indices for act_agent in act_agents], # here 6
            'rewards' : [rew_agent for rew_agents in rewards if rew_agents not in end_indices for rew_agent in rew_agents ], # here 1
            'next_observations' : [next_obs_agent for next_obs_agents in next_observations if next_obs_agents not in end_indices for next_obs_agent in next_obs_agents ],
            'is_done' : [done_agent for done_agents in dones if done_agents not in end_indices for done_agent in done_agents ] # self.is_terminal has shape of [num_envs], representing if terminal for each env. but here should use done with shape of [num_envs,num_agents]
        }
        model.replay_buffer.extend(data_dic) # rewards are not torch tensor, but when using in training, loaded as torch tensor
        end = is_terminals
        end_indices = torch.where(end==1)

        mean_reward = torch.mean(rewards, dim=1)
        cumulative_reward += mean_reward # average reward across all robots in one env
        cumulative_reward_logged = cumulative_reward.sum()
        # Optionally print out step information
        # print(f"Episode: {episode}, Step: {actions}, Reward: {rewards}")
    if env._task.mode == 'normal':
        """add scaler across all tasks with different num_envs"""
        # writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar('cumulative_reward', cumulative_reward_logged, episode)
        # should be :
        writer.add_scalar('average_cumulative_reward', cumulative_reward_logged/env._task.progress_buf, episode) # average reward across all robots in one env
        writer.add_scalar('success', env._task.success.sum(), episode) # 
        """should also add scaler distinguish tasks with different num_envs"""

    print(f"Episode {episode} finished")

env.close()