import sys 
import time

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/algorithms')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/networks')
import gym
from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
# from Supervision import supervision
from BaseNet import StochasticActor, Q
import json
from expertSupervisionEnv import expertSupervisionEnv, angle
from numpy.linalg import norm
import numpy as np
import os
from tensorboardX import SummaryWriter


# from BaseNet import create_network
from net_utils import create_lstm

num_episodes = 10  # Define the number of episodes for testing
# rather than define a number, num_episode should be up to number of tasks used as training data
training_data = os.listdir('/home/tp2/papers/multiarm_dataset/tasks')
num_episodes = len(training_data)*2-500

env = expertSupervisionEnv()

# from multiarm_task import MultiarmTask
from multiarm_with_supervision import MultiarmSupervision
task = MultiarmSupervision(name="MultiarmSupervision")
env.set_task(task, backend = 'torch')

file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/config/default.json'
# Load config JSON file
with open(file_path, 'r') as file:
    config = json.load(file)
    training_config = config['training']

network = create_lstm(training_config=training_config)
# print(network)
# modify for each experiment
experiment_name = '0328testtimeconsumption'

experiment_dir = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRLdata/experiments/' + experiment_name
log_dir = experiment_dir + '/logs'
# checkpoint_dir = experiment_dir + '/checkpoints'
model = SAC(network=network, experiment_dir=experiment_dir,
            load_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRLdata/experiments/0306continue0305/checkpoints/ckpt_sac_lstm_01278'
            )
writer = SummaryWriter(log_dir=log_dir)

# torch.autograd.set_detect_anomaly(True)

for episode in range(num_episodes):
    observations = env.reset()
    done = False

    mode = env.mode
    cumulative_reward = 0
    ep_step = 0
    ep_start_time = time.time()

    while not done: # one episode
        start_time = time.time()

        # mode = env.mode

        # observations = env._task.get_observations().clone() # num_robots * num_robots * 107
        # with normal mode, take an action which NN output.
        if env._task.mode == 'normal':
            # print(env._task.current_task.id)
            RL_start = time.time()
            actions = model.inference(observations) # input in network has shape of batch_size * seq_len * input_size = num_robots * num_robots * 107
            # writer = SummaryWriter(log_dir)
            # writer.add_scalar('reward', env._task., episode)
            RL_end = time.time()
            print("RL action time: ", RL_end - RL_start)
            

        # with supervision mode, take an action based on expert_waypoints
        elif env._task.mode == 'supervision':
            IL_start = time.time()

            actions = env.act_expert()

            IL_end = time.time()
            print("IL action time: ", IL_end - IL_start)
            
        # Step through the environment
        envstep_start = time.time()
        next_observations, rewards, done, info, is_terminals = env.step(actions)
        envstep_end = time.time()
        print("env step time: ", envstep_end - envstep_start)
        # why squeeze?
        # data_dic = {
        #         'observations' : [row.squeeze(0) for i, row in enumerate(observations)],
        #         'actions' : [row.squeeze(0) for i, row in enumerate(actions)],
        #         'rewards' : [row.squeeze(0) for i, row in enumerate(rewards)], 
        #         'next_observations' : [row.squeeze(0) for i, row in enumerate(next_observations)]
        #     }
        rp_start = time.time()
        data_dic = {
                'observations' : [row for i, row in enumerate(observations)], # observation should be 1*n*107
                'actions' : [row for i, row in enumerate(actions)],
                'rewards' : [row for i, row in enumerate(rewards)], # each robot has a reward, size is (n,)
                'next_observations' : [row for i, row in enumerate(next_observations)],
                'is_terminal' : [row for i, row in enumerate(is_terminals)]
            }
        model.replay_buffer.extend(data_dic) # rewards are not torch tensor, but when using in training, loaded as torch tensor
        rp_end = time.time()
        print("replay buffer time: ", rp_end - rp_start)
        # seems unnecessary, because at the start of each loop, observation will be got from Isaac
        observations = next_observations

        mean_reward = np.mean(rewards)
        cumulative_reward += mean_reward

        end_time = time.time()
        if env._task.mode == 'normal':
            print(f"normal mode, time taken for one step: {end_time - start_time}")
        elif env._task.mode =='supervision':
            print(f"supervision mode, time taken for one step: {end_time - start_time}")
        print()

        ep_step += 1

        # Optionally print out step information
        # print(f"Episode: {episode}, Step: {actions}, Reward: {rewards}")
    if env._task.mode == 'normal':
        # writer = SummaryWriter(log_dir=log_dir)
        writer.add_scalar('cumulative_reward', cumulative_reward, episode)
        # should be :
        writer.add_scalar('average_cumulative_reward', cumulative_reward/env._task.progress_buf, episode)
        writer.add_scalar('success', env._task.success, episode)

    print(f"Episode {episode} finished")
    
    print(f"Average time taken for episode {episode}: {(end_time - ep_start_time)/ep_step} seconds")

env.close()
