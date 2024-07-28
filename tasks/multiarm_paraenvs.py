import sys
import time
# sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/exts')

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
from omni.isaac.core.prims import XFormPrimView, GeometryPrimView, RigidPrimView, RigidContactView

from gym import spaces
import numpy as np
import torch
# import math
from math import pi

# import sys 
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone')
sys.path.append('/home/dyb/Thesis/Isaacmultiarmstandalone/robots')
from taskloader import TaskLoader

from thesis_utils import load_config

# from omni.isaac.sensor import _sensor
import omni.kit.commands
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim

# from ur5withEEandTarget import UR5withEEandTarget
from ur5 import UR5
from ur5_view import UR5View
from ur5_view_multienv import UR5MultiarmEnv
from omni.isaac.core.objects import VisualCylinder
import omni

from omni.isaac.core.utils.types import ArticulationActions


class MultiarmTask(BaseTask):
    def __init__(self, name, offset=None, env=None, num_envs=256, train=True, expert_integration=True) -> None:

        """env setting"""
        self._env = env

        self.config = load_config(path='/home/dyb/Thesis/Isaacmultiarmstandalone/config/default.json')

        self.taskloader = TaskLoader(root_dir='/home/dyb/Thesis/tasks', shuffle=True)
        self._num_envs = num_envs
        self._env_spacing = 3
        self.train = train

        self.dt = 1/60 # difference in time between two consecutive states or updates

        self.progress_buf = 0

        self.default_zero_env_path = '/World/envs/env_0'
        self.default_base_env_path = '/World/envs'

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(self.default_zero_env_path)

        self._device = "cuda"

        self.collision_penalty = -10
        self.indiv_reach_target_reward = 5
        self.coorp_reach_target_reward = 10
        self.position_tolerance = 0.07 # modify based on the experiment result
        self.orientation_tolerance = 0.01

        self.ee_velocities_tolerance = 0.02
        self.joint_velocities_tolerance = 0.01
        self.actions_tolerance = 0.001

        self.dof_vel_scale = 0.1
        self.num_franka_dofs = 6

        self._max_episode_length = 150

        self.dof_lower_limits = torch.tensor([-2 * pi, -2 * pi, -2 * pi, -2 * pi, -2 * pi, -2 * pi], device=self._device) # true for real ur5
        self.dof_upper_limits = torch.tensor([2 * pi, 2 * pi, -2 * pi, 2 * pi, 2 * pi, 2 * pi], device=self._device) # true for real ur5

        self.success = torch.zeros((self._num_envs), device=self._device)

        # self.max_velocity = torch.tensor([3.15, 3.15, 3.15, 3.2, 3.2, 3.2], device=self._device) # true for real ur5
        self._num_action = 6 # 6 joint on ur5

        self.drive = "velocity"
        if self.drive == 'velocity':
            self._num_observation = 45 #107, modified to 47 without link position, modify to 47+6(joint_velocity) =53 
            self.joint_velocity_limits = torch.tensor([5/6*pi, 5/6*pi, 5/6*pi, 5/6*pi, pi, pi], device=self._device)
            self.clipAction = torch.ones(self._num_action, device=self._device)
        elif self.drive == 'position':
            self._num_observation = 47

        self.observation_space = None
        self.action_space = None

        self.max_ee_pos = torch.tensor([1.6, 1.6, 0.9], device=self._device)
        self.min_ee_pos = torch.tensor([-1.6, -1.6, 0.0], device=self._device)
        self.max_base_pos = torch.tensor([0.8, 0.8], device=self._device)
        self.min_base_pos = torch.tensor([-0.8, -0.8], device=self._device)

        """task setting"""

        #! consider range of joint config of data in expert trajectories
        self.max_joint = torch.tensor([3.6, 0.1, 2.9, 3.1, 4.0, 4.2], device=self._device)
        self.min_joint = torch.tensor([-6.2, -3.3, -1.8, -4.3, -4.7, -4.4], device=self._device)

        self.fix_agent_num = True
        self.fixed_agent_num = 1

        self.dof_speed_scales = 0.1
        self.action_scale = 7.5
        # self.action_scale = 1.0
        self.expert_integration = expert_integration
        if self.expert_integration == True:
            self.mode = 'supervision'
        elif self.expert_integration == False:
            self.mode = 'normal'

        
        BaseTask.__init__(self, name=name, offset=offset)


    def set_up_scene(self, scene, replicate_physics=True) -> None:

        self.get_franka()
        self.get_target()
        
        # cloner class to create multiple envs
        collision_filter_global_paths = list()
        # if self._sim_config.task_config["sim"].get("add_ground_plane", True):
        self._ground_plane_path = "/World/defaultGroundPlane"
        collision_filter_global_paths.append(self._ground_plane_path)
        scene.add_default_ground_plane(prim_path=self._ground_plane_path) # comsume lots of time
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs) 
        # position of all envs
        self._env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics) 
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path, "/World/collisions", prim_paths, collision_filter_global_paths)



        # create a FrankaView for all robots under num_envs and num_agents
        self.frankaview = ArticulationView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*", name="frankas_view")
        self.frankaview.base_link = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/base_link", name="frankas_view_base_link")
        self.frankaview.links = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/.*_link|tool0|world", name="frankas_view_links")
        self.frankaview.link_for_contact = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/.*_link", name="frankas_view_link_for_contact", track_contact_forces= True)
        self.frankaview.ee_link = RigidPrimView(prim_paths_expr=self.default_base_env_path + "/.*/franka.*/ee_link", name="frankas_view_ee_link")
        self.targetview = GeometryPrimView(prim_paths_expr=self.default_base_env_path + "/.*/target.*", name="targets_view")

        scene.add(self.frankaview)
        scene.add(self.targetview)
        scene.add(self.frankaview.base_link)
        scene.add(self.frankaview.links)
        scene.add(self.frankaview.link_for_contact)
        scene.add(self.frankaview.ee_link)
        
        # set default camera viewport position and target
        self.set_initial_camera_params()

        # self.init_task()
        self.reset()

    def get_franka(self):
        

        usd_path = "/home/dyb/Thesis/Isaacmultiarmstandalone/assets/ur5/ur5.usd"

        for i in range(4):
            ur5 = UR5(prim_path = self.default_zero_env_path + '/franka{}'.format(i), usd_path=usd_path)
            # ee = VisualCylinder(prim_path=self.default_zero_env_path + "/franka{}/ee_link/ee".format(i), radius=0.02, height=0.1, name='UR5EE')

    def get_target(self):
        for i in range(4):
            target = VisualCylinder(prim_path=self.default_zero_env_path + "/target{}".format(i), radius=0.02, height=0.1,
                                    color=np.array([0.8, 0.8, 0.8]),
                                    name='UR5Target')


    def set_initial_camera_params(self, camera_position=[5, 5, 2], camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def update_tasks(self, fix_agent_num, fixed_agent_num):
        if self.expert_integration == True:
            if self.mode == 'supervision':
                self.mode = 'normal'
                self.current_tasks = []
                for i in range(self._num_envs):
                    current_task = self.taskloader.get_next_task()
                    if fix_agent_num:
                        while len(current_task.start_config) != fixed_agent_num: # test only environments with single robot
                            current_task = self.taskloader.get_next_task()
                    else:
                        while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
                            current_task = self.taskloader.get_next_task()
                    self.current_tasks.append(current_task)
            # no need to change to 'supervision' when all success
            elif self.mode == 'normal':
                self.mode = 'supervision'
        elif self.expert_integration == False:
            self.current_tasks = []
            for i in range(self._num_envs):
                current_task = self.taskloader.get_next_task()
                if fix_agent_num:
                    while len(current_task.start_config) != fixed_agent_num: # test only environments with single robot
                        current_task = self.taskloader.get_next_task()
                else:
                    while i != 0 and len(current_task.start_config) != len(self.current_tasks[0].start_config):
                        current_task = self.taskloader.get_next_task()
                self.current_tasks.append(current_task)
    


    def reset(self):
        """
        Reset the environments.
        self.num_agents, self.collision, self.success, self.is_terminals
        self.ob, self.obs, self.actions

        Set : franka.set_local_pose(current_task.base_poses)
              franka.set_joint_positions(current_task.start_config)
              franka.set_joint_position_targets(current_task.start_config)
              franka.set_joint_velocities(zeros)

              franka.target.set_local_pose(current_task.target_eff_poses)

        
        """

        #updata tasks list
        self.update_tasks(self.fix_agent_num, self.fixed_agent_num)
        self.num_agents=len(self.current_tasks[0].start_config)
        complement = 4 - self.num_agents
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_agents, self._num_action))
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_agents, self._num_observation))

        self.collision = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        self.success = torch.zeros((self._num_envs), device=self._device)  
        self.is_terminals = torch.zeros(self._num_envs, device=self._device)    

        self.done = torch.zeros((self._num_envs, self.num_agents), device=self._device)

        self.ob = torch.zeros((self._num_envs, self.num_agents, self._num_observation), device=self._device)
        self.obs = torch.zeros((self._num_envs, self.num_agents, self.num_agents, self._num_observation), device=self._device)
        self.actions = torch.zeros((self._num_envs, self.num_agents, self._num_action), device=self._device)
        self.resets = torch.zeros((1), device=self._device)


        # method below involve more overhead due to the repeated tensor initializations

        # start_config is the initial joint configurations
        start_config = [task.start_config for task in self.current_tasks]
        start_config = torch.tensor(start_config, device=self._device) # [num_envs, num_agents, num_dof]
        start_config_complement = torch.zeros(self._num_envs, complement, self.num_franka_dofs, device=self._device)
        start_config_complemented = torch.cat([start_config, start_config_complement], dim=1) # [num_envs, 4, num_dof]

        self.franka_dof_targets = start_config
        # dof_vel = torch.zeros((self._num_envs, self.num_agents, self.num_franka_dofs), device=self._device)
        dof_vel_complemented = torch.zeros((self._num_envs, 4, self.num_franka_dofs), device=self._device)


        """
        Reset the franka (positions, position targets, velocities, and local pose) to the start configuration.
        """
        # shape of base_pos is (num_envs, num_agents, 3), shape of base_ori is (num_envs, num_agents, 4), same to target_eff
        # robot base_position and target_eff_position
        base_pos_list_envs, base_ori_list_envs = [], []
        target_pos_list_envs, target_ori_list_envs = [], []

        for current_task in self.current_tasks:
            # Append the positions and orientations directly without converting to tensors yet
            base_pos_list_envs += [current_task.base_poses[i][0] for i in range(self.num_agents)]
            base_ori_list_envs += [current_task.base_poses[i][1] for i in range(self.num_agents)]
            target_pos_list_envs += [current_task.target_eff_poses[i][0] for i in range(self.num_agents)]
            target_ori_list_envs += [current_task.target_eff_poses[i][1] for i in range(self.num_agents)]

        # Convert the lists of lists into tensors
        pos_complement = torch.tensor([0,0,-10],device=self._device)
        pos_complement = torch.stack([pos_complement]*complement, dim=0)
        pos_complement = torch.stack([pos_complement]*self._num_envs, dim=0)
        # pos_complement = torch.zeros([self._num_envs, complement, 3], device=self._device)
        rot_complement = torch.tensor([0,0,0,1],device=self._device)
        rot_complement = torch.stack([rot_complement]*complement, dim=0)
        rot_complement = torch.stack([rot_complement]*self._num_envs, dim=0)
        # rot_complement = torch.tensor([self._num_envs, complement, 4], device=self._device)
        base_pos_e = torch.tensor(base_pos_list_envs, device=self._device).view(self._num_envs, self.num_agents, 3) #[num_envs, num_agents, 3]
        base_pos_e_complemented = torch.cat([base_pos_e, pos_complement], dim=1)
        self.world_transform = self._env_pos.unsqueeze(1)
        base_pos_w_complemented = base_pos_e_complemented + self.world_transform
        base_ori = torch.tensor(base_ori_list_envs, device=self._device).view(self._num_envs, self.num_agents, 4)[:, :, [3, 0, 1, 2]]
        base_ori_complemented = torch.cat([base_ori, rot_complement], dim=1)
        target_eff_pos_e = torch.tensor(target_pos_list_envs, device=self._device).view(self._num_envs, self.num_agents, 3)
        target_eff_pos_e_complemented = torch.cat([target_eff_pos_e, pos_complement], dim=1)
        target_eff_pos_w_complemented = target_eff_pos_e_complemented + self.world_transform
        target_eff_ori = torch.tensor(target_ori_list_envs, device=self._device).view(self._num_envs, self.num_agents, 4)[:, :, [3, 0, 1, 2]]
        target_eff_ori_complemented = torch.cat([target_eff_ori, rot_complement], dim=1)


        self.frankaview.set_world_poses(positions = base_pos_w_complemented.view(self._num_envs*4, 3), orientations = base_ori_complemented.view(self._num_envs*4, 4))

        self.frankaview.set_joint_positions(start_config_complemented.view(self._num_envs*4, 6))
        self.frankaview.set_joint_position_targets(start_config_complemented.view(self._num_envs*4, 6))
        self.frankaview.set_joint_velocities(dof_vel_complemented.view(self._num_envs*4, 6))

        self.targetview.set_world_poses(positions = target_eff_pos_w_complemented.view(self._num_envs*4, 3), orientations = target_eff_ori_complemented.view(self._num_envs*4, 4))

        self.progress_buf = 0

        self.base_pos_e_task = base_pos_e
        self.target_eff_pos_e_task = target_eff_pos_e

        complement = torch.zeros(4,device=self._device) # 4
        complement = torch.stack([complement]*self._num_envs, dim=0) # n_e, 4
        complement = torch.cat([self._env_pos, complement],dim=1) # n_e, 7
        complement = complement.unsqueeze(dim=1) # n_e, 1 ,7
        self.complement = torch.cat([complement]*self.num_agents, dim=1) # n_e, n_a ,7

    def pre_physics_step(self, actions) -> None: # actions should have size of (self._num_envs, self.num_agent, 6)

        actions = actions.to(self._device)
        # actions = tensor_clamp(actions, -self.clipAction, self.clipAction)

        if self.drive == 'position':
        # scale the actions from (-1,1) back to joint range
            actions = (actions+1)/2 * (self.max_joint - self.min_joint) + self.min_joint
            targets = actions
            self.franka_dof_targets[:] = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)

        if self.drive == 'velocity':
            targets = self.franka_dof_targets + self.dof_speed_scales * self.dt * actions * self.action_scale # self.dof_speed_scales = self.dof_lower_limits
            self.franka_dof_targets[:] = tensor_clamp(targets, self.dof_lower_limits, self.dof_upper_limits)

        # complement to shape [num_envs, 4(full size), num_action]
        franka_dof_targets = torch.cat([self.franka_dof_targets, torch.zeros(self._num_envs, (4-self.num_agents), self._num_action, device=self._device)], dim=1)
        self.frankaview.set_joint_position_targets(franka_dof_targets.view(self._num_envs*4, self._num_action))

        self.actions = actions.clone()
        self.progress_buf += 1


    
    def pre_observations_nolinks(self): 
        dof_pos = self.frankaview.get_joint_positions() # [num_envs, 4*6]
        dof_pos = dof_pos.view(self._num_envs, 4, 6)[:,:self.num_agents,:].to(self._device)
        self.dof_pos = dof_pos # self.dof_pos is the real joint position of the robot 
        # normalization
        dof_pos_norm = 2*(dof_pos - self.min_joint)/(self.max_joint - self.min_joint) - 1

        dof_vel = self.frankaview.get_joint_velocities()
        dof_vel = dof_vel.view(self._num_envs, 4, 6)[:,:self.num_agents,:].to(self._device)
        dof_vel = dof_vel * self.dof_vel_scale
        self.dof_vel = dof_vel
        
        self.ee_pos_w, self.ee_rot = self.frankaview.ee_link.get_world_poses()
        self.ee_pos_w = self.ee_pos_w.view(self._num_envs, 4, 3)[:,:self.num_agents,:].to(self._device)
        self.ee_pos_e = self.ee_pos_w - self.complement[:,:,:3]
        self.ee_rot = self.ee_rot.view(self._num_envs,4,4)[:,:self.num_agents,:].to(self._device)

        # normalization
        self.ee_pos_e_norm = 2*(self.ee_pos_e - self.min_ee_pos)/(self.max_ee_pos - self.min_ee_pos) - 1

        # target_eff_pose = self.targetview.get_world_poses()
        # target_eff_pose = torch.cat(target_eff_pose, dim=-1).to(self._device)
        # target_eff_pose = target_eff_pose.view(self._num_envs, 4, 7)[:,:self.num_agents,:]
        # self.target_eff_pose = target_eff_pose # num_envs, num_agent, 7
        # # check if target eff has been in the correct pose
        # # print(str(self.target_eff_pos_task)+str(target_eff_pose))
        # # transform from world frame to env frame
        # target_eff_pose -= self.complement
        
        # # normalization
        # target_eff_pose[:,:,:3] = 2*(target_eff_pose[:,:,:3] - self.min_ee_pos)/(self.max_ee_pos - self.min_ee_pos) - 1
        # target_eff_pose_norm = torch.cat([target_eff_pose, target_eff_pose], dim=-1) # observation contains historical frame of target_eff_pose

        if self.progress_buf <= 1:
            """base"""
            base_pose_w = self.frankaview.get_world_poses()
            # # normalization
            # base_pose[0] = 2 * (base_pose[0] - self.min_base_pos)/(self.max_base_pos - self.min_base_pos) - 1
            base_pose_w = torch.cat(base_pose_w, dim=-1).squeeze().to(self._device)
            base_pose_w = base_pose_w.view(self._num_envs, 4, 7)[:,:self.num_agents,:]
            # transform from world frame to env frame
            base_pose_e = base_pose_w - self.complement
            # check if reset correct
            # print(str(self.base_pos)+str(base_pose))           
            # normalization of x and y
            base_pose_e_norm = base_pose_e.clone()
            base_pose_e_norm[:,:,:2] = 2*(base_pose_e[:,:,:2] - self.min_base_pos)/(self.max_base_pos - self.min_base_pos) - 1

            """target_eff"""
            target_eff_pose_w = self.targetview.get_world_poses()
            target_eff_pose_w = torch.cat(target_eff_pose_w, dim=-1).to(self._device)
            target_eff_pose_w = target_eff_pose_w.view(self._num_envs, 4, 7)[:,:self.num_agents,:]
            self.target_eff_pose_w = target_eff_pose_w # num_envs, num_agent, 7
            # check if target eff has been in the correct pose
            # print(str(self.target_eff_pos_task)+str(target_eff_pose))
            # transform from world frame to env frame
            target_eff_pose_e = target_eff_pose_w - self.complement
            self.target_eff_pose_e = target_eff_pose_e
        
            # normalization
            target_eff_pose_e_norm = target_eff_pose_e.clone()
            target_eff_pose_e_norm[:,:,:3] = 2*(target_eff_pose_e[:,:,:3] - self.min_ee_pos)/(self.max_ee_pos - self.min_ee_pos) - 1
            target_eff_pose_e_norm_stack = torch.cat([target_eff_pose_e_norm, target_eff_pose_e_norm], dim=-1) # observation contains historical frame of target_eff_pose

            

            # normalization
            if self.drive == "position":
                self.ob[:, :, 0:6] = dof_pos # joint_position
                self.ob[:, :, 6:12] = dof_pos 
                self.ob[:, :, 12:15] = self.ee_pos_norm # world pos, need to transpose to pos to env
                self.ob[:, :, 15:19] = self.ee_rot
                self.ob[:, :, 19:22] = self.ee_pos_norm
                self.ob[:, :, 22:26] = self.ee_rot
                self.ob[:, :, 26:40] = target_eff_pose # 7*2 # local pos to the env
                self.ob[:, :, 40:47] = base_pose
            elif self.drive == "velocity":
                self.ob[:, :, 0:6] = dof_pos_norm.clone() # joint_position
                self.ob[:, :, 6:12] = dof_pos_norm.clone()
                self.ob[:, :, 12:18] = dof_vel.clone()
                self.ob[:, :, 18:24] = dof_vel.clone()
                # self.ob[:, :, 24:27] = self.ee_pos_norm # world pos, need to transpose to pos to env
                # self.ob[:, :, 27:31] = self.ee_rot
                # self.ob[:, :, 31:34] = self.ee_pos_norm
                # self.ob[:, :, 34:38] = self.ee_rot
                self.ob[:, :, 24:38] = target_eff_pose_e_norm_stack.clone() # 7*2 # local pos to the env
                self.ob[:, :, 38:45] = base_pose_e_norm.clone()

        else:
            if self.drive == 'position':
                self.ob[:, :, 0:6] = self.ob[:, :, 6:12]
                self.ob[:, :, 6:12] = dof_pos
                self.ob[:, :, 12:15] = self.ob[:, :, 19:22]
                self.ob[:, :, 15:19] = self.ob[:, :, 22:26]
                self.ob[:, :, 19:22] = self.ee_pos_norm
                self.ob[:, :, 22:26] = self.ee_rot
                # self.ob[:, :, 26:40] = target_eff_pose # 7*2
                # self.ob[:, :, 40:70] = self.ob[:, :, 70:100]
                # self.ob[:, :, 70:100] = link_position
            elif self.drive == 'velocity':
                self.ob[:, :, 0:6] = self.ob[:, :, 6:12].clone()
                self.ob[:, :, 6:12] = dof_pos_norm.clone()
                self.ob[:, :, 12:18] = self.ob[:, :, 18:24].clone()
                self.ob[:, :, 18:24] = dof_vel.clone()
                # self.ob[:, :, 24:27] = self.ob[:, :, 31:34] # world pos, need to transpose to pos to env
                # self.ob[:, :, 27:31] = self.ob[:, :, 34:38]
                # self.ob[:, :, 31:34] = self.ee_pos_norm
                # self.ob[:, :, 34:38] = self.ee_rot
                # if with dynamic target, then reset the target_eff_pose at each time step:
                # self.ob[:, :, 24:31] = self.ob[:, :, 31:38]
                # self.ob[:, :, 31:38] = target_eff_pose_norm[:,:,:7]

        return self.ob
    
    def get_observations(self):
        """
        shape of self.obs is (self._num_envs, self.num_agents, self.num_agents, self._num_observation)
        """
        obs_start = time.time()
        # firstly sort the self._franka_list by base distance, furthest to closest, for each env
        ob_start = time.time()
        # self.get_observation() # get dof_pos 
        self.pre_observations_nolinks() # get other ob, give up after testing
        ob_end = time.time()
        print('get_observation time:', ob_end - ob_start)

        self.obs = self.ob.unsqueeze(1).expand(-1,self.num_agents, -1, -1)

        distance = torch.cdist(self.base_pos_e_task, self.base_pos_e_task) # self.base_pose has shape of [n_e,n_a,3], dis has shape of [n_e,n_a,n_a]
        sorted_index = distance.argsort(descending = True).to(self._device) #[n_e,n_a,n_a]
        expanded_index = sorted_index.unsqueeze(-1).expand(-1,-1, -1, self._num_observation).to(self._device) 

        self.obs = torch.gather(self.obs, dim=2, index=expanded_index)

        obs_end = time.time()
        print('get_observations time:', obs_end - obs_start)

        return self.obs


    def check_collision(self):
        # if self.frankaview.link_for_contact.get_net_contact_forces() is not None:
        contact_force = self.frankaview.link_for_contact.get_net_contact_forces()
        contact_force = torch.norm(contact_force, dim=-1).to(self._device)
        contact_force = contact_force.view(self._num_envs, 4, 8)[:, :self.num_agents, :]
        collision = torch.where(torch.any(contact_force, dim = -1) > 0.3, 1, 0) # check if any link of robot is contacted
        self.collision = torch.where(collision == 1, 1, self.collision) # check if any link of robot is contacted
        # print('collision happens with link at path:' + str(self.frankaview.link_for_contact.prims))
        """
        set self.is_terminals to be 1 if collision happens in an env, in calculate_metrics()
        """
        # self.is_terminals = torch.where(self.collision == 1, 1, self.is_terminals)
        return collision



    def all_reach_targets(self):

        indiv_reach_targets = self.indiv_reach_targets()
        all_reach_targets = torch.all(indiv_reach_targets, dim=1).int()

        """
        set self.is_terminals to be 1 if all agents in an env reach their targets
        """
        # self.is_terminals = torch.where(all_reach_targets == 1, 1, self.is_terminals)
        return all_reach_targets
    
    def indiv_reach_targets(self):
        indiv_reach_targets = torch.zeros((self._num_envs, self.num_agents), device = self._device)
        # self.ee_pos and self.target_eff_pos are in world frame
        self.pos_delta = torch.norm(self.ee_pos_e - self.target_eff_pose_e[:,:,:3], dim=-1, keepdim=True).squeeze(dim=-1)

        self.ori_delta = self.quaternion_angle_difference(self.ee_rot, self.target_eff_pose_e[:,:,3:])

        # ee_velocities
        # self.ee_velocities = self.frankaview.ee_link.get_velocities(clone=False)
        # self.ee_velocities = self.ee_velocities.view(self._num_envs,4,6)[:,:self.num_agents,:].to(self._device)
        # self.ee_velocities = torch.norm(self.ee_velocities, p=2, dim=-1)

        # joints_velocities
        # self.joint_velocities = self.frankaview.get_joint_velocities(clone=False)
        # self.joint_velocities = self.joint_velocities.view(self._num_envs,4,6)[:,:self.num_agents,:].to(self._device)
        # self.joint_velocities = torch.norm(self.joint_velocities, p=1, dim=-1)

        # actions
        self.actions_norm = torch.norm(self.actions, p=1, dim=-1)


        # if pos_delta < self.position_tolerance and ori_delta < self.orientation_tolerance:
        #     # the agent terminates if reaches its target
        #     self.is_terminals[self._franka_list.index(agent)] = 1
        #     return 1
        # else:
        #     return 0
            # pos_delta = torch.from_numpy(pos_delta).to(self._device).squeeze(dim=-1)
            # ori_delta = torch.from_numpy(ori_delta).to(self._device).squeeze(dim=-1)
        indiv_reach_targets[:,:] = torch.where((self.pos_delta < self.position_tolerance) & 
                                               (self.ori_delta < self.orientation_tolerance), #&
                                            #    (self.ee_velocities < self.ee_velocities_tolerance) &
                                            #    (self.joint_velocities < self.joint_velocities_tolerance) &
                                            #    (self.actions_norm < self.actions_tolerance), 
                                               1, 0)

        return indiv_reach_targets
    
    def quaternion_angle_difference(self, q1, q2):
        q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
        q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
        dot_product = torch.abs(torch.sum(q1 * q2, dim=-1))
        angle = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))
        return angle


    def calculate_metrics(self) -> None: # calculate the rewards in each env.step()

        reward = torch.zeros((self._num_envs, self.num_agents), device = self._device)

        
        collision_penalties = torch.zeros((self._num_envs, self.num_agents), device = self._device)
        if self.progress_buf > 1:
            # self.check_collision()
            collision = self.check_collision()
            collision_penalties = torch.where(collision == 1, self.collision_penalty, 0)
            # set the is_terminal of env with collision to 1
            self.is_terminals = torch.where(self.collision.any(dim=1) == 1, 1, self.is_terminals)


        indiv_reach_target_rewards = torch.zeros((self._num_envs, self.num_agents))
        indiv_reach_target = self.indiv_reach_targets()
        indiv_reach_target_rewards = torch.where(indiv_reach_target==1, self.indiv_reach_target_reward, 0)

        pos_rewards = torch.zeros((self._num_envs, self.num_agents), device=self._device)
        ori_rewards = torch.zeros((self._num_envs, self.num_agents), device=self._device)

        # Smooth, continuous reward for getting closer to the target position
        # pos_rewards[:, :] = torch.exp(-self.pos_delta / self.position_tolerance)
        pos_rewards[:, :] = -1.0 * self.pos_delta
        # Smooth, continuous reward for aligning orientation to the target
        # ori_rewards[:, :] = torch.exp(-self.ori_delta / self.orientation_tolerance)
        ori_rewards[:, :] = -0.5 * self.ori_delta


        collectively_reach_targets_reward = torch.where(self.all_reach_targets() == 1, self.coorp_reach_target_reward, 0)
        
        self.success = torch.where(collectively_reach_targets_reward == self.coorp_reach_target_reward, 1, self.success)
        self.is_terminals = torch.where(collectively_reach_targets_reward == self.coorp_reach_target_reward, 1, self.is_terminals)

        # update self.done
        self.done = torch.where(self.collision == 1, 1, self.done)
        self.done = torch.where(indiv_reach_target == 1, 1, self.done)
        self.done = torch.ones_like(self.done) if self.progress_buf >= self._max_episode_length else self.done

        franka_rewards_sum = \
            collision_penalties + indiv_reach_target_rewards +\
            collectively_reach_targets_reward.unsqueeze(dim=-1) \
            + pos_rewards + ori_rewards
        # -1 + 1 +
        # 2
        # + (0, 1/e) + (0, 1/e)

        reward = franka_rewards_sum

        return reward # reward for each robot
    


    def is_done(self): 
        # reset all envs when all envs are is_terminals
        resets = 0
        # all envs either success or collide
        if self.train and torch.all(self.is_terminals == 1):
            resets = 1
            # print('end episode because of all envs success or collision')
            if torch.all(self.success == 1):
                print('end episode because of all envs succeed')
            elif torch.all(torch.any(self.collision == 1, dim=-1)):
                print('end episode because of all envs collided')
                if self.mode == 'supervision':
                    print('expert make collision')
        
            

        # reset when reach max steps
        # resets = 1 if self.progress_buf >= self._max_episode_length else resets
        if self.progress_buf >= self._max_episode_length:
            self.done = torch.ones((self._num_envs, self.num_agents), device=self._device)
            resets = 1
            print('end episode because of max steps')

        self.resets = resets
        if self.resets == 1:
            print('progress_buf: ', self.progress_buf)

        return resets