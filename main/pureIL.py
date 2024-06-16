import sys 

sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/dataset')
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/envs')
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/tasks')
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/algorithms')
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/networks')
/home/dyb/Thesis/Isaacmultiarmstandalone/envs/pureILenv.py
import gym
from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
# from Supervision import supervision
from BaseNet import StochasticActor, Q
import json
from pureILenv import pureILenv, angle
from numpy.linalg import norm
import numpy as np
import os
from tensorboardX import SummaryWriter


# from BaseNet import create_network
from net_utils import create_lstm

# rather than define a number, num_episode should be up to number of tasks used as training data
training_data = os.listdir('/home/tp2/papers/multiarm_dataset/tasks')
num_episodes = len(training_data)*2-500

env = pureILenv()

# from multiarm_task import MultiarmTask
from multiarm_task import MultiarmTask
task = MultiarmTask(name="MultiarmSupervision")
env.set_task(task, backend = 'torch')

file_path = '/home/dyb/Thesis/Isaacmultiarmstandalone/config/default.json'
# Load config JSON file
with open(file_path, 'r') as file:
    config = json.load(file)
    training_config = config['training']

network = create_lstm(training_config=training_config)
# print(network)
# modify for each experiment
experiment_name = '0321ILtest'

experiment_dir = '/home/dyb/Thesis/Isaacmultiarmstandalonedata/experiments/' + experiment_name
log_dir = experiment_dir + '/logs'
# checkpoint_dir = experiment_dir + '/checkpoints'
model = SAC(network=network, experiment_dir=experiment_dir,
            # load_path = '/home/dyb/Thesis/Isaacmultiarmstandalonedata/experiments/0306continue0305/checkpoints/ckpt_sac_lstm_01278'
            )
writer = SummaryWriter(log_dir=log_dir)

# torch.autograd.set_detect_anomaly(True)

for episode in range(num_episodes):
    observations = env.reset()
    done = False

    # mode = env.mode
    cumulative_reward = 0

    while not done:

        # mode = env.mode

        observations = env._task.get_observations().clone() # num_robots * num_robots * 107
        # with normal mode, take an action which NN output.
        # if env._task.mode == 'normal':
        #     # print(env._task.current_task.id)
        #     actions = model.inference(observations) # input in network has shape of batch_size * seq_len * input_size = num_robots * num_robots * 107
        #     # writer = SummaryWriter(log_dir)
        #     # writer.add_scalar('reward', env._task., episode)

        """train the model when meets requirements"""
        if model.replay_buffer is not None and len(model.replay_buffer) >= model.warmup_steps and model.replay_buffer.freshness > model.minimum_replay_buffer_freshness: 
            model.train()
            model.last_train_size = len(model.replay_buffer)

        # with supervision mode, take an action based on expert_waypoints
        # elif env._task.mode == 'supervision':
        actions = env.act_expert()
            
        # Step through the environment
        next_observations, rewards, done, info, is_terminals = env.step(actions)

        
        data_dic = {
                'observations' : [row for i, row in enumerate(observations)], # observation should be 1*n*107
                'actions' : [row for i, row in enumerate(actions)],
                'rewards' : [row for i, row in enumerate(rewards)], # each robot has a reward, size is (n,)
                'next_observations' : [row for i, row in enumerate(next_observations)],
                'is_terminal' : [row for i, row in enumerate(is_terminals)]
            }
        model.replay_buffer.extend(data_dic) # rewards are not torch tensor, but when using in training, loaded as torch tensor
        

        mean_reward = np.mean(rewards)
        cumulative_reward += mean_reward

        # Optionally print out step information
        # print(f"Episode: {episode}, Step: {actions}, Reward: {rewards}")

    # writer = SummaryWriter(log_dir=log_dir)
    writer.add_scalar('cumulative_reward', cumulative_reward, episode)
    # should be :
    writer.add_scalar('average_cumulative_reward', cumulative_reward/env._task.progress_buf, episode)
    writer.add_scalar('success', env._task.success, episode)

    print(f"Episode {episode} finished")

env.close()