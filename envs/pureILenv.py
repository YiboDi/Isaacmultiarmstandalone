from vec_env_base_custom import VecEnvBase
import numpy as np
import torch
from torch import FloatTensor
from math import acos, cos, sin
from numpy.linalg import norm

def angle(a, b):
    # Angle between two vectors1
        return acos(min(np.dot(a, b) / (norm(a) * norm(b)), 1.0))

class pureILenv(VecEnvBase):
    
    def __init__(self, headless=False):
        super(pureILenv, self).__init__(headless=headless)
        self.expert_root_dir = '/home/dyb/Thesis/expert/'
        self.max_action = 1.0
        self.joint_tolerance = 0.2

    def load_expert_waypoints_for_task(self, task_id):
        expert_path = self.expert_root_dir + task_id + ".npy"
        try:
            rrt_waypoints = np.load(expert_path)
        except Exception:
            return None
        return rrt_waypoints # joint_position
    
    

    def load_expert_waypoints_for_tasks(self, task_ids):
        waypoints_list = []
        for task_id in task_ids:
            expert_path = self.expert_root_dir + task_id + ".npy"
            try:
                # Load the expert waypoints for the given task ID
                rrt_waypoints = np.load(expert_path)
                rrt_waypoints = torch.from_numpy(rrt_waypoints)
            except Exception as e:
                # Handle the exception as needed (e.g., log an error message, return None, etc.)
                print(f"Error loading waypoints for task {task_id}: {e}")
                rrt_waypoints = None  # Or consider an alternative handling strategy

            waypoints_list.append(rrt_waypoints)

        return waypoints_list # return a list (size is num_envs) of np.array/torch.tensor (shape is [num_agents, num_steps])

    
    def act_expert(self):
        expert_waypoints = self.load_expert_waypoints_for_task(
                    task_id=self._task.current_task.id)
            
        # load curr_j from Isaac sim    
        curr_j = torch.empty(self._task.num_agents * 6,) 
        for i, agent in enumerate(self._task._franka_list[0:self._task.num_agents]):
            dof = agent.get_joint_positions()
            curr_j[i*6:i*6+6] = dof.clone()
        curr_j = curr_j.numpy()

        # find the nearest wp in expert_waypoints and set it as next_wp and its idx to be next_wp_idx
        result = [(idx, norm(wp - curr_j))
                for idx, wp in enumerate(expert_waypoints)]
        result.sort(
            key=lambda v: v[1])
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
        while True: 
            target_j = expert_waypoints[target_wp_idx]
            target_dir_j = target_j - curr_j


            if target_wp_idx < len(expert_waypoints) - 1 and \
                all([delta_j < self.max_action for delta_j in abs(target_dir_j)])\
                    and angle(next_dir_j, target_dir_j) < self.joint_tolerance:
                target_wp_idx += 1
            else:
                break


        #calculate the actions based on the waypoints
        actions = target_j - curr_j
        actions = actions.reshape((self._task.num_agents, 6))
        actions = torch.from_numpy(actions).clone()
        return actions
    
