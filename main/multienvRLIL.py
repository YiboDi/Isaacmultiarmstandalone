import sys 
import time

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/algorithms')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/networks')
# import gym
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

file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/config/default.json'
# Load config JSON file
with open(file_path, 'r') as file:
    config = json.load(file)
    training_config = config['training']

network = create_lstm(training_config=training_config)
# print(network)
# modify for each experiment
experiment_name = '0423test'

experiment_dir = '/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Di_custom/multiarmRLdata/experiments/' + experiment_name
log_dir = experiment_dir + '/logs'
# checkpoint_dir = experiment_dir + '/checkpoints'
model = SAC(network=network, experiment_dir=experiment_dir,
            # load_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/experiments/01.24/checkpoints/ckpt_sac_lstm_00337'
            )
writer = SummaryWriter(log_dir=log_dir)

# torch.autograd.set_detect_anomaly(True)

for episode in range(num_episodes):
    observations = env.reset()
    reset = False # flag indicate the reset

    mode = env.mode
    # cumulative_reward = torch.zeros(env._task._num_envs, device='cuda')
    cumulative_reward = torch.zeros(1, device='cuda')
    end = torch.zeros(env._task._num_envs, device='cuda')

    step_count = torch.zeros(env._task._num_envs, device='cuda')

    while not reset: # loop in one episode
        step_start = time.time()
        # mode = env.mode
        end = env._task.is_terminals
        end_mask = end!=1
        # observations = env._task.get_observations() # num_robots * num_robots * 107
        # with normal mode, take an action which NN output.
        if env._task.mode == 'normal':
            # print(env._task.current_task.id)
            if observations.dim()==4: # num_envs, num_agnets, num_agents, num_obs_per_agent
                obs = observations.reshape(-1, *observations.shape[2:])

            inference_start = time.time()
            actions = model.inference(obs) # input in network has shape of batch_size * seq_len * input_size = num_robots * num_robots * 107
            inference_end = time.time()
            # self.inference_time = inference_end - inference_start
            print('inference time: ', inference_end-inference_start)

            # shape of actions is (batch size, 6)
            # dispatch the batch size back to (num_envs, num_robots)
            # actions = actions.reshape(env._task._num_envs, env._task.num_agents, *actions.shape[1:])
            actions = actions.reshape(env._task._num_envs, env._task.num_agents, 6)


        # with supervision mode, take an action based on expert_waypoints
        elif env._task.mode == 'supervision':
            actions = env.act_experts(step_count).to('cuda')
            
        # Step through the environment
        # actions_reshaped = actions.reshape(env._task._num_envs, env._task.num_agents, *actions.shape[1:])
        envstep_start = time.time()
        next_observations, rewards, reset, info, is_terminals, dones = env.step(actions)
        envstep_end = time.time()
        print('env.step time: ', envstep_end-envstep_start)

        
        # data_dic = {
        #         'observations' : [row for i, row in enumerate(observations)], # observation should be 1*n*107
        #         'actions' : [row for i, row in enumerate(actions)],
        #         'rewards' : [row for i, row in enumerate(rewards)], # each robot has a reward, size is (n,)
        #         'next_observations' : [row for i, row in enumerate(next_observations)],
        #         'is_terminal' : [row for i, row in enumerate(is_terminals)]
        #     }

        # for multi envs setting :
        # try:
        #     end_mask
        # except:
        #     pass
        # else:
        rpextend_start = time.time()
        obs = observations[end_mask]
        actions = actions[end_mask]
        rewards = rewards[end_mask]
        next_observations = next_observations[end_mask]
        dones = dones[end_mask]
        # alternatively using .flatten()
        data_dic = {
        'observations' : [obs_agent for obs_agents in obs for obs_agent in obs_agents], # here n*107
        'actions' : [act_agent for act_agents in actions for act_agent in act_agents], # here 6
        'rewards' : [rew_agent for rew_agents in rewards for rew_agent in rew_agents ], # here 1
        'next_observations' : [next_obs_agent for next_obs_agents in next_observations for next_obs_agent in next_obs_agents ],
        'is_terminal' : [done_agent for done_agents in dones for done_agent in done_agents ] # self.is_terminal has shape of [num_envs], representing if terminal for each env. but here should use done with shape of [num_envs,num_agents]
        }
        model.replay_buffer.extend(data_dic) # rewards are not torch tensor, but when using in training, loaded as torch tensor
        rpextend_end = time.time()
        print('rpextend time: ', rpextend_end-rpextend_start)
        # end = is_terminals
        # end_mask = end!=1
        step_end = time.time()
        print('step time: ', step_end-step_start)
        print()
        # mean_reward = torch.mean(rewards, dim=1)
        # cumulative_reward += mean_reward # average reward across all robots in one env
        # cumulative_reward_logged = cumulative_reward.sum()
        mean_reward = torch.mean(rewards)
        cumulative_reward += mean_reward
        step_count += 1-end 
        # Optionally print out step information
        # print(f"Episode: {episode}, Step: {actions}, Reward: {rewards}")
        
    if env._task.mode == 'normal':
        """add scaler across all tasks with different num_envs"""
        # writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar('cumulative_reward', cumulative_reward, episode)
        # should be :
        writer.add_scalar('average_cumulative_reward', cumulative_reward/step_count.sum(), episode) # average reward across all robots in one env
        writer.add_scalar('success', env._task.success.sum(), episode) # 
        """should also add scaler distinguish tasks with different num_envs"""

    print(f"Episode {episode} finished")

env.close()