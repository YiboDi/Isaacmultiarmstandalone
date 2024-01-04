import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/dataset')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/envs')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/algorithms')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/networks')
import gym
from vec_env_base_custom import VecEnvBase
import torch
from SAC import SAC
from BaseNet import StochasticActor, Q
import json
from expertSupervisionEnv import expertSupervisionEnv, angle
from numpy.linalg import norm


from BaseNet import create_network

num_episodes = 75000  # Define the number of episodes for testing

env = expertSupervisionEnv()

# from multiarm_task import MultiarmTask
from multiarm_with_supervision import MultiarmSupervision
task = MultiarmSupervision(name="MultiarmSupervision")
env.set_task(task, backend = 'torch')

file_path = '/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/config/default.json'
# Load JSON file
with open(file_path, 'r') as file:
    config = json.load(file)
    training_config = config['training']

network = create_network(training_config=training_config)
model = SAC(network=network)

for episode in range(num_episodes):
    observations = env.reset()
    done = False

    # mode = env.mode

    while not done:

        mode = env.mode

        observations = env._task.get_observations() # num_robots * num_robots * 107
        if mode == 'normal':
            actions = model.inference(observations) # input in network has shape of batch_size * seq_len * input_size = num_robots * num_robots * 107
        elif mode == 'supervision':
            #load waypoints
            if expert_waypoints == None or next_wp_idx != None and next_wp_idx >= len(expert_waypoints):
                expert_waypoints = env.load_expert_waypoints_for_task(
                    task_id=env._task.current_task.id) # 1 dimensional data, num_robots * 6
            # if next_wp_idx == None:
            #     next_wp_idx = 0
            # curr_j = env._task.get_joint_positions() # change franka_list
            curr_j = torch.empty(shape=(env.task.num_agents * 6,)) 
            for i, agent in enumerate(env._task._franka_list[0:env.task.num_agents]):
                dof = agent.get_joint_positions()
                curr_j[i*6:i*6+6] = dof
            curr_j = curr_j.numpy()


            result = [(idx, norm(wp - curr_j))
                  for idx, wp in enumerate(expert_waypoints)]
            result.sort(
                key=lambda v: v[1])
            next_wp_idx = result[0][0]

            while curr_j - expert_waypoints[next_wp_idx] < 0.01:
                next_wp_idx += 1
            target_wp_idx = next_wp_idx
            next_dir_j = expert_waypoints[next_wp_idx] - curr_j
            while True: # find the most far away expert wp (target_j) in dataset that fullfil the conditions(tolerance config)
                target_j = expert_waypoints[target_wp_idx]
                target_dir_j = target_j - curr_j
                max_action = 0.5
                joint_tolerance = 0.1
                if target_wp_idx < len(expert_waypoints) - 1 and \
                    all([delta_j < max_action for delta_j in abs(target_dir_j)])\
                        and angle(next_dir_j, target_dir_j) < joint_tolerance:
                    target_wp_idx += 1
                else:
                    break
            if next_wp_idx < len(expert_waypoints) - 1:
                next_wp_idx += 1
            # actions = target_j - curr_j
            # env.step(actions)
            else:
                mode = 'normal'

            #calculate the actions based on the waypoints
            actions = target_j - curr_j

        # Step through the environment
        next_observations, rewards, done, info = env.step(actions)

        # for i in range(next_observations.shape[0]):
        data_dic = {
                'observations' : [row.squeeze(0) for i, row in enumerate(observations)],
                'actions' : [row.squeeze(0) for i, row in enumerate(actions)],
                'rewards' : [row.squeeze(0) for i, row in enumerate(rewards)], 
                'next_observations' : [row.squeeze(0) for i, row in enumerate(next_observations)]
            }
        model.replay_buffer.extend(data_dic) # rewards may not be torch tensor
        

        observations = next_observations

        # Optionally print out step information
        print(f"Episode: {episode}, Step: {actions}, Reward: {rewards}")

    print(f"Episode {episode} finished")

env.close()
