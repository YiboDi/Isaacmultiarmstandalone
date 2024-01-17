from vec_env_base_custom import VecEnvBase
import numpy as np
from torch import FloatTensor
from math import acos, cos, sin
from numpy.linalg import norm

def angle(a, b):
    # Angle between two vectors1
        return acos(min(np.dot(a, b) / (norm(a) * norm(b)), 1.0))

class expertSupervisionEnv(VecEnvBase):
    
    def __init__(self, headless=False):
        super(expertSupervisionEnv, self).__init__(headless=headless)
        # self.mode = self._task.mode
        self.mode = None
        self.expert_root_dir = '/home/tp2/papers/multiarm_dataset/expert/'


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
    
    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True) -> None:
         return super().set_task(task, backend, sim_params, init_sim)
    
    # def step(self, actions):
    #      observations, rewards, dones, info = super().step(actions)

    #      is_terminals = self._task.is_terminals

    #      return observations, rewards, dones, info, is_terminals

    

    # def compute_actions_to_waypoint(
    #     self,
    #     ur5s,
    #     waypoint, # target_j
    #     centralized_policy=False,
    #     action_type='delta'):
    #     if not centralized_policy:
    #         waypoints = np.split(np.array(waypoint), len(ur5s))
    #         if action_type == 'target-norm': # false, delta
    #             # Normalize waypoint w.r.t UR5 joint limits
    #             # 
    #             waypoints = [UR5.normalize_joint_values(wp)
    #                         for wp in waypoints]
    #             return [(FloatTensor(target_joint))
    #                     for ur5, target_joint in zip(ur5s, waypoints)]
    #         return [FloatTensor(
    #             np.array(target_joint) -
    #             np.array(ur5.get_joint_positions()))
    #             for ur5, target_joint in zip(ur5s, waypoints)]
    #     else:
    #         # waypoint = np.array(waypoint)
    #         # current_joint_values = np.array(list(chain.from_iterable(
    #         #     [ur5.get_arm_joint_values() for ur5 in ur5s])))
    #         # return [FloatTensor(waypoint - current_joint_values)]
    #         pass
        
    # def perform_expert_actions_variable_threshold(
    #         self,
    #         env,
    #         expert_waypoints,
    #         joint_tolerance,
    #         threshold,
    #         max_action,
    #         action_type,
    #         log=False):
    #     def done():
    #         return ur5s_at_waypoint( # true if the current joint pos is the same as the target
    #             env.active_ur5s,
    #             expert_waypoints[-1],
    #             threshold=0.001)
    #     next_wp_idx = 0
    #     curr_j = np.array(list(chain.from_iterable( # current joint pos
    #         [ur5.get_arm_joint_values() for ur5 in env.active_ur5s])))
    #     rv = None
    #     while (not env.terminate_episode) and \
    #             (not done()):

    #         # Closest waypoint ahead of current
    #         next_j = expert_waypoints[next_wp_idx] # next joint pos

    #         # If already at waypoint, then skip
    #         # and expert waypoints not run out
    #         if ur5s_at_waypoint(
    #             env.active_ur5s,
    #             next_j,
    #             threshold=threshold) and\
    #                 next_wp_idx < len(expert_waypoints) - 1:
    #             next_wp_idx += 1
    #             continue

    #         # next direction to move in joint space
    #         next_dir_j = next_j - curr_j

    #         # Find next target joint that is within action
    #         # magnitude and joint direction tolerance
    #         target_wp_idx = next_wp_idx
    #         while True: # find the most far away expert wp (target_j) in dataset that fullfil the conditions(tolerance config)
    #             target_j = expert_waypoints[target_wp_idx]
    #             target_dir_j = target_j - curr_j
    #             if target_wp_idx < len(expert_waypoints) - 1 and \
    #                 all([delta_j < max_action for delta_j in abs(target_dir_j)])\
    #                     and angle(next_dir_j, target_dir_j) < joint_tolerance:
    #                 target_wp_idx += 1
    #             else:
    #                 break
    #         actions = compute_actions_to_waypoint(
    #             ur5s=env.active_ur5s,
    #             waypoint=target_j,
    #             centralized_policy=env.centralized_policy,
    #             action_type=action_type)

    #         rv = env.step(actions)

    #         curr_j = np.array(list(chain.from_iterable(
    #             [ur5.get_arm_joint_values() for ur5 in env.active_ur5s])))
    #         # Set current waypoint index to the waypoint closest to curr_j
    #         result = [(idx, norm(wp - curr_j))
    #                 for idx, wp in enumerate(expert_waypoints)]
    #         result.sort(
    #             key=lambda v: v[1])
    #         next_wp_idx = result[0][0]
    #         if next_wp_idx < len(expert_waypoints) - 1:
    #             next_wp_idx += 1
    #     if env.current_step >= env.episode_length and log:
    #         print("expert out of time!")
    #     return rv

    # def perform_expert_actions(
    #     self,
    #     env,
    #     expert_waypoints,
    #     expert_config):
    #     if expert_config['waypoint_conversion_mode'] == 'tolerance': # true
    #         # Variable threshold
    #         # using the threshold in config to limit the actions
    #         tolerance_config = expert_config['tolerance_config']
    #         return self.perform_expert_actions_variable_threshold(
    #             env=env,
    #             expert_waypoints=expert_waypoints,
    #             joint_tolerance=tolerance_config['tolerance'],
    #             threshold=tolerance_config['threshold'],
    #             max_action=tolerance_config['max_magnitude'],
    #             action_type=env.action_type)
    #     elif expert_config['action_type'] == 'threshold': # default to be delta
    #         # threshold_config = expert_config['threshold_config']
    #         # return perform_expert_actions_fixed_threshold(
    #         #     env=env,
    #         #     expert_waypoints=expert_waypoints,
    #         #     expert_action_threshhold=threshold_config['threshold'],
    #         #     action_type=env.action_type)
    #         pass
    #     else:
    #         raise NotImplementedError
        
