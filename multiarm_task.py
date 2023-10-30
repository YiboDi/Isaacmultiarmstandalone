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



class MultiarmTask(BaseTask):
    def __init__(self, name, env, offset=None) -> None:

        self.config = load_config(path='/home/tp2/papers/decentralized-multiarm/configs/default.json')

        self.taskloader = TaskLoader(root_dir='/home/tp2/papers/decentralized-multiarm/tasks', shuffle=True)
        self.current_task = self.taskloader.get_next_task()

        self.action_scale = 1.0
        self.dt = 1/60 # difference in time between two consecutive states or updates
        self.episode_length = self.config['environment']['episode_length']
        self.progress_buf = 0

        self._cloner = GridCloner(spacing=3)
        self._cloner.define_base_env("/World/envs")
        define_prim("/World/envs/env_0")

        self._env = env
        self._device = "gpu"
        self._num_envs = 1

        self.collision_penalty = -0.05
        self.delta_pos_reward = 0
        self.delta_ori_reward = 0
        self.activation_radius = 100
        self.indiv_reach_target_reward = 0.01
        self.coorp_reach_target_reward = 1
        self.position_tolerance = 0.04
        self.orientation_tolerance = 0.1


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
        # self._num_observations = 4
        self._num_action = 6 # 6 joint on ur5
        self._num_actions = self._num_action * self.current_task.ur5_count # num of actions in single env


        # self._device = "cpu"
        # self._device = "gpu"
        # self.num_envs = 1

        # a few class buffers to store RL-related states
        # self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.ob = torch.zeros((self.num_agents, self._num_observation))
        self.obs = torch.zeros((self.num_agents, self.num_agents, self._num_observation))
        self.resets = torch.zeros((self._num_envs, 1))
        # self.resets = torch.zeros((1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)  #[-1, +1]
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        ) #(-oo,+oo)

        def get_env_local_pose(env_pos, xformable, device):  # in parallel envs, get the env related pos based on the world related pos and env pos 
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()
            
            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)


        # def get_pose(xformable, device):
        #     world_transform = xformable.ComputerLocalToWorldTransform(0)
        #     world_pos = world_transform.ExtractTranslation()
        #     world_quat = world_transform.ExtractRotationQuat()

        #     px = world_pos[0] 
        #     py = world_pos[1] 
        #     pz = world_pos[2] 
        #     qx = world_quat.imaginary[0]
        #     qy = world_quat.imaginary[1]
        #     qz = world_quat.imaginary[2]
        #     qw = world_quat.real

        #     return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)


        stage = get_current_stage()
        hand_pose = get_env_local_pose(self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_link7")), self._device)
        lfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_leftfinger")), self._device
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0], UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda_rightfinger")), self._device
        )

        # stage = get_current_stage()
        # hand_pose = get_pose(UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/franka/panda_link7")), self._device)
        # lfinger_pose = get_pose(
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/franka/panda_leftfinger")), self._device
        # )
        # rfinger_pose = get_pose(
        #     UsdGeom.Xformable(stage.GetPrimAtPath("/World/Franka/franka/panda_rightfinger")), self._device)

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(hand_pose[3:7], hand_pose[0:3]))

        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3])
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_actions), device=self._device)



    def set_up_scene(self, scene, replicate_physics=True) -> None:

        # eliminate all existing scene firstly
        self.scene.clear()

        self.num_agents=self.current_task.ur5_count

        self.get_franka(self.num_agents)

        self._franka_list=[]

        for i in self.num_agents:
            franka = ArticulationView(prim_paths_expr="/World/Franka/franka{}".format(i), name="franka{}_view".format(i))
            scene.add(franka)
            self._franka_list.append(franka)
            self._franka_list[-1].target_eff_pose = self.current_task.target_eff_poses[i,:]
            self._franka_list[-1].goal_config = self.current_task.goal_config[i,:]
        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        # self._franka = ArticulationView(prim_paths_expr="/World/Cartpole*", name="franka_view")
        # add Cartpole ArticulationView and ground plane to the Scene
        # scene.add(self._franka)
        scene.add_default_ground_plane()

        # collision_filter_global_paths = list()
        # # if self._sim_config.task_config["sim"].get("add_ground_plane", True):
        # self._ground_plane_path = "/World/defaultGroundPlane"
        # collision_filter_global_paths.append(self._ground_plane_path)
        # scene.add_default_ground_plane(prim_path=self._ground_plane_path)
        # prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        # self._env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics)
        # self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        # self._cloner.filter_collisions(
        #     self._env._world.get_physics_context().prim_path, "/World/collisions", prim_paths, collision_filter_global_paths)


        # set default camera viewport position and target
        self.set_initial_camera_params()

        self.init_task()

    def target_poses(self, idx):
        return self.taskloader.c


    def get_franka(self, num_agents=1):
        
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"
        # add the Cartpole USD to our stage
        for i in num_agents:
           position = self.current_task.base_poses[i][0]
           orientation = self.current_task.base_poses[i][0]
           create_prim(prim_path="/World/Frankas/Franka{}".format(i), prim_type="Xform", position=position, orientation=orientation, usd_path=usd_path) 

           result, sensor = omni.kit.commands.execute(
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

       
        # add_reference_to_stage(usd_path, "/World/Cartpole")


    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):

        self.current_task = self.taskloader.get_next_task()
        self.set_up_scene()

        self.num_franka_dofs = self._franka_list[0].num_dof
        self.franka_dof_pos = torch.zeros((self.num_agents, self.num_franka_dofs), device=self._device)
        dof_limits = self._franka_list.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[-1] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self.num_agents, self.num_franka_dofs), dtype=torch.float, device=self._device
        )

        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        num_resets = len(env_ids)

        self.franka_dof_targets = self.franka_dof_pos
        dof_vel = torch.zeros((self.num_agents, self._franka_list[0].num_dof), device=self._device)

        for i, agent in enumerate(self._franka_list):
            self._franka_list[i].set_joint_position_targets(self.franka_dof_targets[i, :], indices=indices)
            self._franka_list[i].set_joint_positions(self.franka_dof_pos[i, :], indices=indices)
            self._franka_list[i].set_joint_velocities(dof_vel[i, :], indices=indices)



        # # randomize DOF positions
        # dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # # randomize DOF velocities
        # dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        # self._cartpoles.set_joint_velocities(dof_vel, indices=indices)
        # for i, agent in enumerate(self._franka_list):
            


        # bookkeeping
        self.resets[env_ids] = 0 # self.resets = 0
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
        pos = np.array(this_franka.get_world_pose()) # the base position of this_franka
        sorted_franka_list = self._franka_list.sort(reverse=True, key=lambda agent:
                                                    np.linalg.norm(pos - np.array(agent.get_world_pose()[0]))) # or without [0] or [0:3]
        for i, agent in enumerate(sorted_franka_list):
            dof_pos = agent.get_joint_positions()
            hand_pos, hand_rot = agent._hands.get_world_poses()
            target_eff_pose = agent.target_eff_pose
            goal_config = agent.goal_config
            link_position = agent.get_link_positions() # get link positions of links, 30 for ur5
            base_pose = agent.get_world_positions() # get the position of the base
            if self.ob == torch.zeros((self.num_agents, self._num_observation)): # if first step (no history yet)
                self.ob[i, 0:6] = dof_pos
                self.ob[i, 6:12] = dof_pos
                self.ob[i, 12:15] = hand_pos
                self.ob[i, 15:19] = hand_rot
                self.ob[i, 19:22] = hand_pos
                self.ob[i, 22:26] = hand_rot
                self.ob[i, 26:40] = target_eff_pose # 7*2
                self.ob[i, 40:70] = link_position
                self.ob[i, 70:100] = link_position
                self.ob[i, 100:107] = base_pose

            else:
                self.ob[i, 0:6] = self.ob[i, 6:12]
                self.ob[i, 6:12] = dof_pos
                self.ob[i, 12:15] = self.ob[i, 19:22]
                self.ob[i, 15:19] = self.ob[i, 22:26]
                self.ob[i, 19:22] = hand_pos
                self.ob[i, 22:26] = hand_rot
                self.ob[i, 26:40] = target_eff_pose # 7*2
                self.ob[i, 40:70] = self.ob[i, 70:100]
                self.ob[i, 70:100] = link_position
                self.ob[i, 100:107] = base_pose


        return self.ob # observation of a single franka (this_franka), shape of num_agents*ob of a single agent


    def get_observations(self):
        # do not consider multi env temporally
        
        # firstly sort the self._franka_list by base distance, furthest to closest.


        for i, agent in enumerate(self._franka_list):
            
            self.obs[i, :, :] = self.get_observation(agent)


            # hand_pos, hand_rot = agent._hands.get_world_poses()
            # dof_pos = agent.get_joint_positions()
            # # dof_vel = agent.get_joint_velocities()


            # if dof_pos.shape[0]+dof_vel.shape[0] != self._num_observation:
            #     raise ValueError('dim of observation does not match')
            
            # self.obs[i,:] = torch.cat(dof_pos, hand_pos, hand_rot) # shape of self.obs is num_robots * (num_joint_pos * num_joint_vel)


        return self.obs # observation of the whole system, shape of num_agents*num_agents*ob of a single agent

    def check_collision(self):

        self._contact_sensor_interface = _sensor.acquire_contact_sensor_interface()

        for agent in self._franka_list:
            raw_readings = self._contact_sensor_interface.get_contact_sensor_raw_data(agent.prim_path + "/sensor")
            if raw_readings.shape[0]:                
                for reading in raw_readings:
                    if "franka" in str(self._contact_sensor_interface.decode_body_name(reading["body1"])):
                        return True # Collision detected with some part of the robot
                    if "franka" in str(self._contact_sensor_interface.decode_body_name(reading["body0"])):
                        return True # Collision detected with some part of the robot
                    
        return False
    
    def all_reach_targets(self):
        for i, agent in enumerate(self._franka_list):
            agent.


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

        collision_penalties = np.array(self.collision_penalty if self.check_collision() else 0)

        if self.cooperative_individual_reach_target:
            individually_reached_target_rewards = np.array(
                [self.individually_reach_target * state['reach_count']
                    for _ in range(len(self.active_ur5s))])
        else:
            individually_reached_target_rewards = np.array([
                (self.individually_reach_target
                    if ur5_state['reached_target']
                 else 0)
                for ur5_state in state['ur5s']
            ])

        
        # Only give delta rewards if ee is within a radius
        # away from the target ee
        delta_position_rewards = \
            [(prev - curr) * self.delta_reward['position']
             if curr < self.delta_reward['activation_radius']
             else 0.0
             for curr, prev in zip(
                current_ur5_ee_residuals[0],
                self.prev_ur5_ee_residuals[0])]
        delta_orientation_rewards = \
            [(prev - curr) * self.delta_reward['orientation']
             if curr_pos_res < self.delta_reward['activation_radius']
             else 0.0
             for curr, prev, curr_pos_res in zip(
                current_ur5_ee_residuals[1],
                self.prev_ur5_ee_residuals[1],
                current_ur5_ee_residuals[0])]
        # proximity penalty
        proximity_penalties = np.array([
            sum([(1 - closest_points_to_other[0][8]
                  / self.proximity_penalty_distance)
                 * self.proximity_penalty
                 for closest_points_to_other in ur5.closest_points_to_others
                 if len(closest_points_to_other) > 0 and
                 closest_points_to_other[0][8] <
                 self.proximity_penalty_distance])
            for ur5 in self.active_ur5s])

        collectively_reached_targets = (
            state['reach_count'] == len(self.active_ur5s))
        collective_reached_targets_rewards = np.array(
            [(self.collectively_reach_target
                if collectively_reached_targets
              else 0)
             for _ in range(len(self.active_ur5s))])
        self.prev_ur5_ee_residuals = current_ur5_ee_residuals
        ur5_rewards_sum = \
            collision_penalties + individually_reached_target_rewards +\
            collective_reached_targets_rewards + survival_penalties + \
            proximity_penalties + \
            delta_position_rewards + delta_orientation_rewards
        if self.centralized_policy:
            return np.array([ur5_rewards_sum.sum()])
        else:
            return ur5_rewards_sum

        return reward.item()

    def is_done(self) -> None:

        resets = torch.where(self.check_collision(), torch.ones_like(self.resets), self.resets)

        resets = torch.where(self.progress_buf >= self.episode_length - 1, torch.ones_like(self.resets), self.resets)

        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        self.resets = resets


        return resets.item()
