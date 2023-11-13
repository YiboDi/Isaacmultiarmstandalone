# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.cloner import Cloner
import omni.kit
from pxr import Usd, UsdGeom

from gym import spaces
import numpy as np
import torch
import math
from tasks import TaskLoader
from utils import load_config

from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim

from ur5 import UR5
from ur5_view import UR5View
from omni.isaac.core.objects import VisualCylinder



class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None) -> None:

        self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')

        self.taskloader = TaskLoader(root_dir='/home/tp2/papers/multiarm_dataset/tasks', shuffle=True)
        self.current_task = self.taskloader.get_next_task()

        self.action_scale = 1.0
        self.dt = 1/60 # difference in time between two consecutive states or updates
        self.episode_length = self.config['environment']['episode_length']
        self.progress_buf = 0

        self._cloner = GridCloner(spacing=3)
        self._cloner.define_base_env("/World/envs")
        define_prim("/World/envs/env_0")

        # self._env = env
        self._device = "cuda"
        self._num_envs = 1

        self.collision_penalty = -0.05
        self.delta_pos_reward = 0
        self.delta_ori_reward = 0
        self.activation_radius = 100
        self.indiv_reach_target_reward = 0.01
        self.coorp_reach_target_reward = 1
        self.position_tolerance = 0.04
        self.orientation_tolerance = 0.1

        self.num_franka_dofs = 6

        self.observation_space = None
        self.action_space = None


        # trigger __init__ of parent class
        BaseTask.__init__(self, name=name, offset=offset)

    def init_task(self):

        # task-specific parameters for multiagent
        

        # task-specific parameters
        # self._cartpole_position = [0.0, 0.0, 2.0]
        # self._reset_dist = 3.0 #reset when cart is more than 3m away from start position
        # self._max_push_effort = 400.0

        #values used for defining RL buffers for multiagent and dynamic
        self._num_observation = 0 
        for item in self.config['training']['observations']['items']:
            self._num_observation += item['dimensions'] * (item['history'] + 1)

        self._num_observations = self._num_observation * self.current_task.ur5_count # num of obs in single env
        

        # values used for defining RL buffers

        self._num_action = 6 # 6 joint on ur5
        self._num_actions = self._num_action * self.current_task.ur5_count # num of actions in single env

        # a few class buffers to store RL-related states
        self.ob = torch.zeros((self.num_agents, self._num_observation))
        self.obs = torch.zeros((self.num_agents, self.num_agents, self._num_observation))
        # self.resets = torch.zeros((self._num_envs, 1))
        self.resets = torch.zeros((1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)  #[-1, +1]
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        ) #(-oo,+oo)

        stage = get_current_stage()

        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_actions), device=self._device)



    def set_up_scene(self, scene, replicate_physics=True) -> None:

        # eliminate all existing scene firstly
        if scene != None:
            scene.clear()

        self.num_agents=self.current_task.ur5_count

        self.get_franka(self.num_agents)

        self._franka_list=[]
        # self.ee_list = []
        # self.targets_list = []
        # usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/assets/ur5/ur5.usd"

        for i in range(4):

            if self.current_task.target_eff_poses[i][0] is not None:
                franka = UR5View(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i),
                                target_pos = self.current_task.target_eff_poses[i][0],
                                target_ori = self.current_task.target_eff_poses[i][1]
                                )
                scene.add(franka)
                self._franka_list.append(franka)

            # add the following variable in UR5View init
            # self._franka_list[-1].
                self._franka_list[-1].target_eff_pose = self.current_task.target_eff_poses[i] # shape of num_agents*(translation3+orientation4), precisely a list of num_agents * [3.4]
                self._franka_list[-1].goal_config = self.current_task.goal_config[i] # goal joint states, shape same with above

            # self._franka_list[-1].ee = scene.add(VisualCylinder(prim_path='/World/Franka/franka{}/ee_link/point'.format(i), radius=0.02, height=0.1))
            
            # self._franka_list[-1].target = scene.add(VisualCylinder(prim_path='World/Franka/franka/target', radius=0.02, height=0.1, color=[0.8, 0.8, 0.8],
            #                                   translation = self.current_task.target_eff_poses[i,0:3],
            #                                   orientation = self.current_task.target_eff_poses[i,3:7]))
            
            # scene.add(self.ee)
            # scene.add(self.target)
            elif self.current_task.target_eff_poses[i][0] is None:
                franka = UR5View(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i),
                                # target_pos = self.current_task.target_eff_poses[i][0],
                                # target_ori = self.current_task.target_eff_poses[i][1]
                                )
                scene.add(franka)

        scene.add_default_ground_plane()
        

        # set default camera viewport position and target
        self.set_initial_camera_params()

        self.init_task()

    # def target_poses(self, idx):
    #     return self.taskloader.c


    def get_franka(self, num_agents):
        

        # assets_root_path = get_assets_root_path()
        # usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/Di_custom/multiarmRL/assets/ur5/ur5.usd"

        for i in range(4):
            if self.current_task.base_poses[i][0] is not None:
                position = self.current_task.base_poses[i][0]
                orientation = self.current_task.base_poses[i][1]
                default_dof_pos = self.current_task.start_config[i]
                UR5(prim_path="/World/Franka/franka{}".format(i), translation=position, orientation=orientation, usd_path=usd_path, default_dof_pos=default_dof_pos) 

                sensor_result, sensor = omni.kit.commands.execute(
                        "IsaacSensorCreateContactSensor",
                        path="/sensor",
                        parent = "/World/Frankas/Franka{}".format(i),
                        min_threshold=0,
                        max_threshold=10000000,
                        color= (1, 0, 0, 1),
                        radius=0.05, # ? what for?
                        sensor_period=self.dt,
                        # sensor_period=-1,
                        # offset= Gf.Vec3d(0,0,0),
                        translation=Gf.Vec3f(0, 0, 0),
                        visualize=True,)
                
            elif self.current_task.base_poses[i][0] is None:
                UR5(prim_path="/World/Franka/franka{}".format(i), translation=[0, 0, -10], 
                #    orientation=orientation, 
                   usd_path=usd_path, 
                #    default_dof_pos=default_dof_pos
                   )



    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")


    def reset(self):
        self.current_task = self.taskloader.get_next_task()
        # self.set_up_scene()
        self.num_agents=self.current_task.ur5_count
        self.base_poses = self.current_task.base_poses
        self.start_config = self.current_task.start_config
        self.goal_config = self.current_task.goal_config
        self.target_eff_poses = self.current_task.target_eff_poses

        self._num_observations = self._num_observation * self.current_task.ur5_count
        self._num_actions = self._num_action * self.current_task.ur5_count


        # self.num_franka_dofs = self._franka_list[0].num_dof
        self.num_franka_dofs = 6
        self.franka_dof_pos = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)
        dof_limits = self._franka_list[0].get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[-1] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self.num_agents, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        self.franka_dof_targets = self.franka_dof_pos
        # dof_vel = torch.zeros((self.num_agents, self._franka_list[0].num_dof), device=self._device)
        dof_vel = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)

        for i, agent in enumerate(self._franka_list):
            if i < self.num_agents:
                self._franka_list[i].set_world_poses(self.base_poses[i])
                self._franka_list[i].set_joint_position_targets(self.start_config[i])
                self._franka_list[i].set_joint_positions(self.start_config[i])
                self._franka_list[i].set_joint_velocities(dof_vel[i])

                self._franka_list[i].target.set_world_poses(self.target_eff_poses[i])

            elif i >= self.num_agents:
                self._franka_list[i].set_world_poses([0,0,-10])

        self.progress_buf = 0

    # def post_reset(self):

    #     self.current_task = self.taskloader.get_next_task()
    #     self.set_up_scene()

    #     # self.num_franka_dofs = self._franka_list[0].num_dof
    #     self.num_franka_dofs = 6
    #     self.franka_dof_pos = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)
    #     dof_limits = self._franka_list[0].get_dof_limits()
    #     self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
    #     self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
    #     self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
    #     self.franka_dof_speed_scales[-1] = 0.1
    #     self.franka_dof_targets = torch.zeros(
    #         (self.num_agents, self.num_franka_dofs), dtype=torch.float, device=self._device
    #     )

    #     self.reset()

    # def reset(self, env_ids=None): # env.reset() -> self._task.reset() here
    #     if env_ids is None:
    #         env_ids = torch.arange(self._num_envs, device=self._device)
    #     num_resets = len(env_ids)

    #     self.franka_dof_targets = self.franka_dof_pos
    #     # dof_vel = torch.zeros((self.num_agents, self._franka_list[0].num_dof), device=self._device)
    #     dof_vel = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)

    #     for i, agent in enumerate(self._franka_list):
    #         self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[i, :], indices=indices)
    #         self._franka_list[i].set_joint_positions(self.franka_dof_pos[i, :], indices=indices)
    #         self._franka_list[i].set_joint_velocities(dof_vel[i, :], indices=indices)

    #     # # randomize DOF positions
    #     # dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
    #     # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
    #     # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

    #     # # randomize DOF velocities
    #     # dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
    #     # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
    #     # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

    #     # apply resets
    #     indices = env_ids.to(dtype=torch.int32)
            
    #     # bookkeeping
    #     self.resets[env_ids] = 0 # self.resets = 0
        self.progress_buf = 0

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.resets.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset(reset_env_ids)

        actions = torch.tensor(actions)
        self.actions = actions.clone().to(self._device)

        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        for i, agent in enumerate(self._franka_list):
            self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[i, :])

        self.progress_buf += 1


        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observation(self, this_franka):
        pos = np.array(this_franka.get_world_poses()) # the base position of this_franka
        sorted_franka_list = self._franka_list.sort(reverse=True, key=lambda agent:
                                                    np.linalg.norm(pos - np.array(agent.get_world_poses()[0]))) # or without [0] or [0:3]
        for i, agent in enumerate(sorted_franka_list):
            dof_pos = agent.get_joint_positions()
            ee_pos, ee_rot = agent.ee.get_world_poses()
            target_eff_pose = agent.target_eff_pose
            # goal_config = agent.goal_config

            # link position is the center of mass, for simplification using pos of links here
            link_position = agent.get_link_positions() # get link positions of links, 30 for ur5, why 30? reason: 10*3
            
            base_pose = agent.get_world_poses() # get the position of the base
            if self.ob == torch.zeros((self.num_agents, self._num_observation)): # if first step (no history yet)
                self.ob[i, 0:6] = dof_pos
                self.ob[i, 6:12] = dof_pos
                self.ob[i, 12:15] = ee_pos
                self.ob[i, 15:19] = ee_rot
                self.ob[i, 19:22] = ee_pos
                self.ob[i, 22:26] = ee_rot
                self.ob[i, 26:40] = target_eff_pose # 7*2
                self.ob[i, 40:70] = link_position
                self.ob[i, 70:100] = link_position
                self.ob[i, 100:107] = base_pose

            else:
                self.ob[i, 0:6] = self.ob[i, 6:12]
                self.ob[i, 6:12] = dof_pos
                self.ob[i, 12:15] = self.ob[i, 19:22]
                self.ob[i, 15:19] = self.ob[i, 22:26]
                self.ob[i, 19:22] = ee_pos
                self.ob[i, 22:26] = ee_rot
                self.ob[i, 26:40] = target_eff_pose # 7*2
                self.ob[i, 40:70] = self.ob[i, 70:100]
                self.ob[i, 70:100] = link_position
                self.ob[i, 100:107] = base_pose


        return self.ob # observation of a single franka (this_franka), shape of num_agents*ob of a single agent


    def get_observations(self):
        # do not consider multi env temporally
        
        # firstly sort the self._franka_list by base distance, furthest to closest.


        for i, agent in enumerate(self._franka_list):
            
            self.obs[i, :, :] = self.get_observation(this_franka=agent)


            # hand_pos, hand_rot = agent._hands.get_world_poses()
            # dof_pos = agent.get_joint_positions()
            # # dof_vel = agent.get_joint_velocities()


            # if dof_pos.shape[0]+dof_vel.shape[0] != self._num_observation:
            #     raise ValueError('dim of observation does not match')
            
            # self.obs[i,:] = torch.cat(dof_pos, hand_pos, hand_rot) # shape of self.obs is num_robots * (num_joint_pos * num_joint_vel)


        return self.obs # observation of the whole system, shape of num_agents*num_agents*ob of a single agent

    def check_collision(self, agent):

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        # for agent in self._franka_list:
        #     raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(agent.prim_path + "/sensor")
        #     if raw_readings.shape[0]:                
        #         for reading in raw_readings:
        #             if "franka" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
        #                 return True # Collision detected with some part of the robot
        #             if "franka" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
        #                 return True # Collision detected with some part of the robot
        raw_readings = self._contact_sensor_interface.get_sensor_reading(agent.prim_path + '/sensor')
        if raw_readings[-1] == 0:
            return 0
        else:
            return 1

    
    # def all_reach_targets(self):
    #     for i, agent in enumerate(self._franka_list):
    #         pos_delta = np.linalg.norm(agent.get_world_pose()[0:3] - self.current_task.target_eff_poses[i, 0])
    #         ori_delat = np.linalg.norm(agent.get_world_pose()[3:7] - self.current_task.target_eff_poses[i, 1])
    #         if pos_delta > self.position_tolerance or ori_delat > self.orientation_tolerance:
    #             return 0

    #     return 1

    def all_reach_targets(self):
        for i, agent in enumerate(self._franka_list):
            pos_delta = np.linalg.norm(agent.ee_link.get_world_pose()[0:3] - self.agent.target_eff_poses[0:3])
            ori_delat = np.linalg.norm(agent.ee_link.get_world_pose()[3:7] - self.agent.target_eff_poses[3:7])
            if pos_delta > self.position_tolerance or ori_delat > self.orientation_tolerance:
                return 0

        return 1
    
    def indiv_reach_targets(self, agent):
        pos_delta = np.linalg.norm(agent.ee_link.get_world_pose()[0:3] - self.agent.target_eff_poses[0:3])
        ori_delat = np.linalg.norm(agent.ee_link.get_world_pose()[3:7] - self.agent.target_eff_poses[3:7])
        if pos_delta < self.position_tolerance and ori_delat < self.orientation_tolerance:
            return 1
        else:
            return 0


    def calculate_metrics(self) -> None:
        cart_pos = self.obs[:, 0]
        cart_vel = self.obs[:, 1]
        pole_angle = self.obs[:, 2]
        pole_vel = self.obs[:, 3]

        # compute reward based on angle of pole and cart velocity
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # apply a penalty if cart is too far from center
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # apply a penalty if pole is too far from upright
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)


        current_ur5_ee_residuals = self.get_ur5_eef_residuals()
        if self.prev_ur5_ee_residuals is None:
            self.prev_ur5_ee_residuals = current_ur5_ee_residuals


        # collision_penalties = np.array([
        #     (self.collision_penalty if ur5_state['colliding'] else 0)
        #     for ur5_state in state['ur5s']
        # ])

        # collision_penalties = np.array(self.collision_penalty if self.check_collision() else 0)
        collision_penalties = np.zeros(self.num_agents)
        for i, agent in enumerate(self._franka_list):
            # collision_check[i] = self.check_collision(agent=agent) 
            if self.check_collision(agent=agent) == 1:
                collision_penalties[i] = self.collision_penalty
            elif self.check_collision(agent=agent) == 0:
                collision_penalties[i] = 0
            else:
                raise ValueError('The reading of the contact sensor makes no sense')


        # individually_reached_target_rewards = np.array([
        #     (self.individually_reach_target
        #         if ur5_state['reached_target']
        #         else 0)
        #     for ur5_state in state['ur5s']
        # ])
        indiv_reach_target_rewards = np.zeros(self.num_agents)
        for i, agent in enumerate(self._franka_list):
            if self.indiv_reach_targets(agent=agent):
                indiv_reach_target_rewards[i] = self.indiv_reach_target_reward
            elif self.indiv_reach_targets(agent=agent) == 0:
                indiv_reach_target_rewards[i] = 0
            else:
                raise ValueError('The agent should either reach its target or not')

        
        # Only give delta rewards if ee is within a radius
        # away from the target ee
        # delta_position_rewards = \
        #     [(prev - curr) * self.delta_reward['position']
        #      if curr < self.delta_reward['activation_radius']
        #      else 0.0
        #      for curr, prev in zip(
        #         current_ur5_ee_residuals[0],
        #         self.prev_ur5_ee_residuals[0])]
        # delta_orientation_rewards = \
        #     [(prev - curr) * self.delta_reward['orientation']
        #      if curr_pos_res < self.delta_reward['activation_radius']
        #      else 0.0
        #      for curr, prev, curr_pos_res in zip(
        #         current_ur5_ee_residuals[1],
        #         self.prev_ur5_ee_residuals[1],
        #         current_ur5_ee_residuals[0])]


        # collectively_reached_targets = (
        #     state['reach_count'] == len(self.active_ur5s))
        # collective_reached_targets_rewards = np.array(
        #     [(self.collectively_reach_target
        #         if collectively_reached_targets
        #       else 0)
        #      for _ in range(len(self.active_ur5s))])

        if self.all_reach_targets:
            collectively_reach_targets_reward = np.full((self.num_agents, ), self.coorp_reach_target_reward)
        else:
            collectively_reach_targets_reward = np.zeros(self.num_agents)

        
        # self.prev_ur5_ee_residuals = current_ur5_ee_residuals

        franka_rewards_sum = \
            collision_penalties + indiv_reach_target_rewards +\
            collectively_reach_targets_reward 
            # + survival_penalties + \
            # proximity_penalties + \
            # delta_position_rewards + delta_orientation_rewards

        # if self.centralized_policy:
        #     return np.array([ur5_rewards_sum.sum()])
        # else:
        #     return ur5_rewards_sum
        reward = franka_rewards_sum

        return reward

    def is_done(self):

        # resets = torch.where(self.check_collision(), torch.ones_like(self.resets), self.resets)
        # resets = torch.where(self.progress_buf >= self.episode_length - 1, torch.ones_like(self.resets), self.resets)
        resets = torch.where(self.check_collision(), 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        self.resets = resets


        return resets.item()
