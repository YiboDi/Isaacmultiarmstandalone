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

class expertmultiEnv(VecEnvBase):
    
    def __init__(self, headless=False):
        super(expertmultiEnv, self).__init__(headless=headless)
        # self.mode = self._task.mode
        self.mode = None
        self.expert_root_dir = '/home/dyb/Thesis/expert/'
        self.max_action = 1.0
        self.joint_tolerance = 0.4

        self.failed_count = 0

        self.max_joint = torch.tensor([3.6, 0.1, 2.9, 3.1, 4.0, 4.2], device='cuda')
        self.min_joint = torch.tensor([-6.2, -3.3, -1.8, -4.3, -4.7, -4.4], device='cuda')
    
    def step(self, actions):
        """ Basic implementation for stepping simulation. 
            Can be overriden by inherited Env classes
            to satisfy requirements of specific RL libraries. This method passes actions to task
            for processing, steps simulation, and computes observations, rewards, and resets.

        Args:
            actions (Union[numpy.ndarray, torch.Tensor]): Actions buffer from policy.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            rewards(Union[numpy.ndarray, torch.Tensor]): Buffer of rewards data.
            dones(Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
            info(dict): Dictionary of extras data.
        """
        prephysics_start = time.time()
        self._task.pre_physics_step(actions)
        prephysics_end = time.time()
        print('pre_physics_time: ', prephysics_end - prephysics_start)

        worldstep_start = time.time()
        self._world.step(render=self._render) # steps the physics simulation
        worldstep_end = time.time()
        print('worldstep_time: ', worldstep_end - worldstep_start)

        self.sim_frame_count += 1

        obs_start = time.time()
        observations = self._task.get_observations()
        obs_end = time.time()
        print('obs_time: ', obs_end - obs_start)

        rewards_start = time.time()
        rewards = self._task.calculate_metrics() # also update self.done
        rewards_end = time.time()
        print('rewards_time: ', rewards_end - rewards_start)
        
        resets = self._task.is_done() # self.reset_buf change if some env meet the conditions
        info = {}

        done = self._task.done
        
        is_terminals = self._task.is_terminals

        # self.progress_buf[:] += 1

        return observations, rewards, resets, info, is_terminals, done


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
    
    # def load_expert_waypoints_for_task_multienv(self, task_ids):
    #     expert_waypoints = []
    #     for task_id in task_ids:
    #         expert_path = self.expert_root_dir + task_id + ".npy"
    #         try:
    #             rrt_waypoints = np.load(expert_path)
    #         except Exception:
    #             return None
    #         expert_waypoints.append(rrt_waypoints)
    #     return expert_waypoints
    
    # def load_expert_waypoints_for_task_multienv(self, task_ids):
    #     expert_paths = []
    #     for task_id in task_ids:
    #         expert_path = self.expert_root_dir + task_id + ".npy"
    #         expert_paths.append(expert_path)
    #     try:
    #         rrt_waypoints = np.array([np.load(expert_path)] for expert_path in expert_paths)
    #     except Exception:
    #         return None
        
    #     return rrt_waypoints
    

    def load_expert_waypoints_for_tasks(self, task_ids):
        waypoints_list = []
        max_len = 0
        for task_id in task_ids:
            expert_path = self.expert_root_dir + task_id + ".npy"
            try:
                # Load the expert waypoints for the given task ID
                rrt_waypoints = np.load(expert_path)
                # rrt_waypoints = torch.from_numpy(rrt_waypoints)
                length = rrt_waypoints.shape[0]
                if length > max_len:
                    max_len = length
            except Exception as e:
                # Handle the exception as needed (e.g., log an error message, return None, etc.)
                print(f"Error loading waypoints for task {task_id}: {e}")
                # rrt_waypoints = None  # Or consider an alternative handling strategy
                return None

            waypoints_list.append(rrt_waypoints)
        # pad the trajectories to max_length
        padded_waypoints_list = self.waypoints_preprocessing(waypoints_list, max_len)

        return padded_waypoints_list # return a list (size is num_envs) of np.array/torch.tensor (shape is [num_agents, num_steps])
    
    def waypoints_preprocessing(self, waypoints_list, max_len):
        # pad all the trajectories to max_length
        padded_waypoints_list = []
        for i, waypoints in enumerate(waypoints_list):
            last_element = waypoints[-1]
            padding = np.repeat(last_element[np.newaxis, :], max_len - waypoints.shape[0], axis=0)
            padded = np.concatenate([waypoints, padding], axis=0)
            padded_waypoints_list.append(padded)
        return padded_waypoints_list

    
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
    
    def act_experts(self, step_count):
        if torch.all(step_count == 0):
        # Assuming `load_expert_waypoints_for_tasks` can load multiple trajectories for parallel tasks
            expert_waypoints_batch = self.load_expert_waypoints_for_tasks(task_ids=[task.id for task in self._task.current_tasks]) # a list of np.array with same size [max_size, 6]
            if expert_waypoints_batch is None:
                self.failed_count += 1
                return None
            else:
                expert_waypoints = np.stack(expert_waypoints_batch, axis=0) # shape: [num_envs, max_size, 6]
                self.expert_waypoints = torch.tensor(expert_waypoints).to('cuda') # shape: [num_envs, max_size, num_agents*6]
                max_size = expert_waypoints.shape[1]
                self.expert_waypoints = self.expert_waypoints.view(self._task._num_envs, max_size, self._task.num_agents, 6) # shape: [num_envs, max_size, num_agents*6]
        # Initialize a tensor to store current joint positions from all parallel Isaac sim environments
        curr_js = self._task.dof_pos # [num_envs, num_agents, 6]
        # curr_js = curr_js.unsqueeze(1) # [num_envs, 1, num_agents, 6]
        distances = torch.norm(self.expert_waypoints - curr_js.unsqueeze(1), dim=-1) # [num_envs, max_size, num_agents], unsqueeze for broadcasting
        distances = torch.sum(distances, dim=-1) # [num_envs, max_size]
        next_wp_idx = torch.argmin(distances, dim=1)  # [num_envs]

        next_wp_idx = torch.where(next_wp_idx >= self.expert_waypoints.shape[1]-1, next_wp_idx, next_wp_idx+1)


        target_wp_idx = next_wp_idx.clone()


        while True:
            target_wp_idx_reshaped = target_wp_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,1,self.expert_waypoints.shape[2], self.expert_waypoints.shape[3]) # [num_envs, max_size, num_agents, 6]
            target_j = torch.gather(self.expert_waypoints, 1, target_wp_idx_reshaped) # [num_envs, num_agents, 6]
            target_j = target_j.squeeze(1)

            cond1 = target_wp_idx < self.expert_waypoints.shape[1] - 1
            cond2 = torch.all(torch.abs(target_j - curr_js) < self.max_action, dim=-1).squeeze()
            # cond2 = torch.all(cond2, dim=-1).squeeze()
            cond2 = torch.all(cond2, dim=-1)
            cond = cond1 & cond2
            if not cond.any():
                break
            target_wp_idx = torch.where(cond, target_wp_idx+1, target_wp_idx)

        actions = target_j #[num_envs, num_agents, 6]
        actions = 2*(actions - self.min_joint) / (self.max_joint - self.min_joint) - 1 #[num_envs, num_agents, 6]


        return actions.clone()