from vec_env_base_custom import VecEnvBase
import numpy as np
import torch
from torch import FloatTensor
from math import acos, cos, sin
from numpy.linalg import norm

def angle(a, b):
    # Angle between two vectors1
        return acos(min(np.dot(a, b) / (norm(a) * norm(b)), 1.0))

class expertmultiEnv(VecEnvBase):
    
    def __init__(self, headless=False):
        super(expertmultiEnv, self).__init__(headless=headless)
        # self.mode = self._task.mode
        self.mode = None
        self.expert_root_dir = '/home/tp2/papers/multiarm_dataset/expert/'
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
        for task_id in task_ids:
            expert_path = self.expert_root_dir + task_id + ".npy"
            try:
                # Load the expert waypoints for the given task ID
                rrt_waypoints = np.load(expert_path)
            except Exception as e:
                # Handle the exception as needed (e.g., log an error message, return None, etc.)
                print(f"Error loading waypoints for task {task_id}: {e}")
                rrt_waypoints = None  # Or consider an alternative handling strategy

            waypoints_list.append(rrt_waypoints)

        return waypoints_list

    
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
    
    def act_experts(self):
        # Assuming `load_expert_waypoints_for_tasks` can load multiple trajectories for parallel tasks
        expert_waypoints_batch = self.load_expert_waypoints_for_tasks(task_ids=[task.id for task in self._task.current_tasks])

        # Initialize a tensor to store current joint positions from all parallel Isaac sim environments
        curr_js = torch.empty(len(self._task.current_tasks), self._task.num_agents * 6)
        for task_idx, task in enumerate(self._task.current_tasks):
            for i, agent in enumerate(self._task._franka_list[0:self._task.num_agents]):
                dof = agent.get_joint_positions()
                curr_js[task_idx, i * 6:(i + 1) * 6] = torch.tensor(dof)

        # Process each trajectory in the batch
        actions_batch = torch.empty_like(curr_js)
        for idx, (curr_j, expert_waypoints) in enumerate(zip(curr_js, expert_waypoints_batch)):
            # Find the nearest waypoint to the current position for each trajectory
            distances = torch.norm(expert_waypoints - curr_j.unsqueeze(0).repeat(len(expert_waypoints), 1), dim=1)
            next_wp_idx = torch.argmin(distances)

            # Ensure the next waypoint is not too close to the current position
            while next_wp_idx < len(expert_waypoints) - 1 and torch.all(torch.abs(curr_j - expert_waypoints[next_wp_idx]) < 0.01):
                next_wp_idx += 1

            # Initialize target waypoint index
            target_wp_idx = next_wp_idx

            # Calculate direction to the next waypoint
            next_dir_j = expert_waypoints[next_wp_idx] - curr_j

            # Find a target waypoint that is within action and joint tolerance limits
            while target_wp_idx < len(expert_waypoints) - 1:
                target_j = expert_waypoints[target_wp_idx]
                target_dir_j = target_j - curr_j

                if torch.all(torch.abs(target_dir_j) < self.max_action) and torch.dot(next_dir_j, target_dir_j) / (torch.norm(next_dir_j) * torch.norm(target_dir_j)) > torch.cos(self.joint_tolerance):
                    target_wp_idx += 1
                else:
                    break

            # Calculate actions based on the selected target waypoint
            actions = (expert_waypoints[target_wp_idx] - curr_j).reshape((self._task.num_agents, 6))
            actions_batch[idx] = actions

        return actions_batch.clone()