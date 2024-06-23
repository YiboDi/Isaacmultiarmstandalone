from vec_env_base_custom import VecEnvBase
import numpy as np
import torch
from torch import FloatTensor
from math import acos, cos, sin
from numpy.linalg import norm

import time

def angle(a, b):
    # Angle between two vectors1
        return acos(min(np.dot(a, b) / (norm(a) * norm(b)), 1.0))

class expertSupervisionEnv(VecEnvBase):
    
    def __init__(self, headless=False):
        super(expertSupervisionEnv, self).__init__(headless=headless)
        # self.mode = self._task.mode
        self.mode = None
        self.expert_root_dir = '/home/dyb/Thesis/expert/'
        self.max_action = 1.0
        self.joint_tolerance = 0.2


    def reset(self):
        self.mode = self._task.mode
        return super().reset()
    
    def load_expert_waypoints_for_task(self, task_id):
        expert_path = self.expert_root_dir + task_id + ".npy"
        try:
            rrt_waypoints = np.load(expert_path)
        except Exception:
            return None
        return rrt_waypoints # joint_position
    
    # def set_task(self, task, backend="numpy", sim_params=None, init_sim=True) -> None:
    #      return super().set_task(task, backend, sim_params, init_sim)
    
    def act_expert(self):
        loadexpert_start = time.time()
        expert_waypoints = self.load_expert_waypoints_for_task(
                    task_id=self._task.current_task.id)
        loadexpert_end = time.time()
        print('load expert time:' + str(loadexpert_end - loadexpert_start))

        # if next_wp_idx == None:
        #     next_wp_idx = 0
        # curr_j = self._task.get_joint_positions() # change franka_list
            
        # load curr_j from Isaac sim    
        loadjp_start = time.time()
        curr_j = torch.empty(self._task.num_agents * 6,) 
        for i, agent in enumerate(self._task._franka_list[0:self._task.num_agents]):
            dof = agent.get_joint_positions()
            curr_j[i*6:i*6+6] = dof.clone()
        curr_j = curr_j.numpy()
        loadjp_end = time.time()
        print('load joint position time:' + str(loadjp_end - loadjp_start))

        # find the nearest wp in expert_waypoints and set it as next_wp and its idx to be next_wp_idx
        closest_start = time.time()
        result = [(idx, norm(wp - curr_j))
                for idx, wp in enumerate(expert_waypoints)]
        result.sort(
            key=lambda v: v[1])
        closest_end = time.time()
        print('find closest waypoint time:' + str(closest_end - closest_start))
        # for i in range(5):
        #     print('waypoint{} is:'.format(i)+str(expert_waypoints[i]))
        next_wp_idx = result[0][0] + 1 if result[0][0] < len(expert_waypoints)-1 else result[0][0]# ensure that the initial next_wp is next to curr_j

        # if next_wp is too close to curr_j, check next one in expert_waypoint. when condition is satisfied, initialize the target_j as next_wp.
        # if dis between next_wp and curr_j smaller than threshold (here 0.01), then we regard robots are current in this pose, so check next wp.
        # dis = curr_j - expert_waypoints[next_wp_idx] # dis check if robots already at this pose
        # seem 0.01 is too small
        # while dis.any() < 0.01:
        while np.all(curr_j - expert_waypoints[next_wp_idx] < 0.01):
            # next_wp_idx += 1
            if next_wp_idx >= len(expert_waypoints) - 1:
                break
            next_wp_idx += 1
            
        target_wp_idx = next_wp_idx
        next_dir_j = expert_waypoints[next_wp_idx] - curr_j
        # find the most far away expert wp (target_j) in dataset that fullfil the conditions(tolerance config)
        # comfirm the final target_j in this step
        target_start = time.time()
        while True: 
            target_j = expert_waypoints[target_wp_idx]
            target_dir_j = target_j - curr_j

            if target_wp_idx < len(expert_waypoints) - 1 and \
                all([delta_j < self.max_action for delta_j in abs(target_dir_j)])\
                    and angle(next_dir_j, target_dir_j) < self.joint_tolerance:
                target_wp_idx += 1
            else:
                break
        target_end = time.time()
        print('find target waypoint time:' + str(target_end - target_start))
        # if next_wp_idx (which is nearest waypoint to the curr_j) is the last one, change the mode to normal
        # if next_wp_idx < len(expert_waypoints) - 1:
        #     next_wp_idx += 1
        # # actions = target_j - curr_j
        # # self.step(actions)
            
        # # so when else, should reset the task, modify
        # else:
        #     mode = 'normal'

        #calculate the actions based on the waypoints
        actions = target_j - curr_j
        # actions should be target_j instead of target_j - curr_j?
        # actions = target_j
        actions = actions.reshape((self._task.num_agents, 6))
        actions = torch.from_numpy(actions).to('cuda')
        return actions