from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.objects import VisualCylinder
import numpy as np
import torch

class UR5MultiarmEnv(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "UR5View",
    ) -> None:
        """[summary]
        """
        # define the child variables for ur5: target_eff_pose, goal_config, all links
        # self.ee = RigidPrimView(prim_paths_expr = prim_paths_expr + "/ee_link")
        # self.target_eff_pose = None
        # self.goal_config = None
        #the order of links might be different with pybullet sim
        self.base_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/base_link", name = name + '_base_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.shoulder_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/shoulder_link", name = name + '_shoulder_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.upper_arm_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/upper_arm_link", name= name + '_upper_arm_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.forearm_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/forearm_link", name=name+'_forearm_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.wrist_1_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/wrist_1_link", name=name+'_wrist_1_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.wrist_2_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/wrist_2_link", name = name+'_wrist_2_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.wrist_3_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/wrist_3_link", name=name+'_wrist_3_link', track_contact_forces=True, prepare_contact_sensors=True, reset_xform_properties=False,)
        self.ee_link = RigidPrimView(prim_paths_expr = prim_paths_expr + "/ee_link", 
                                        # track_contact_forces=True, prepare_contact_sensors=True,
                                        )
        self.tool0 = RigidPrimView(prim_paths_expr = prim_paths_expr + "/tool0")
        self.world = RigidPrimView(prim_paths_expr = prim_paths_expr + "/world")

        self.link_list = [self.base_link,
                          self.shoulder_link,
                          self.upper_arm_link,
                          self.forearm_link,
                          self.wrist_1_link,
                          self.wrist_2_link,
                          self.wrist_3_link,
                          self.ee_link,
                          self.tool0,
                          self.world
                          ]
        
        self.link_for_contact = [
                          self.base_link,
                          self.shoulder_link,
                          self.upper_arm_link,
                          self.forearm_link,
                          self.wrist_1_link,
                          self.wrist_2_link,
                          self.wrist_3_link,
                          ]
        
        # for link in self.link_for_contact:
        #     link.initialize()
        
        self.ee = None
        self.target = None

        # self.link_position = []
        # for i,link in self.link_list:
        #     self.link_position += link.get_coms()

        self.prim_path = prim_paths_expr

        super().__init__(
            prim_paths_expr = prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )
        #set all link as instantiates of RigidPrimView class, the prim_paths_expr = prim_path_expr + '/link_name'
        # self.ee = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_link7", name="ee_view", reset_xform_properties=False)

        # self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_link7", name="hands_view", reset_xform_properties=False)
        # self._lfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_leftfinger", name="lfingers_view", reset_xform_properties=False)
        # self._rfingers = RigidPrimView(prim_paths_expr="/World/envs/.*/franka/panda_rightfinger",  name="rfingers_view", reset_xform_properties=False)

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        # self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), self.get_dof_index("panda_finger_joint2")]

    # here use pos of links instead of coms of links    
    def get_link_positions(self):
        self.link_position = []
        for link in self.link_list:
            # com_np = link.get_coms()
            # com = list(com_np)
            # self.link_position += com
            link_pos = link.get_local_poses()[0]
            # link_pos = torch.cat(link_pos)
            self.link_position.append(link_pos)

        self.link_position = torch.cat(self.link_position, dim=1)

        return self.link_position
    # @property
    # def gripper_indices(self):
    #     return self._gripper_indices

    def get_link_coms(self):
        self.link_coms = []
        for link in self.link_list:
            try:
                link_com = link.get_coms()[0]
            except:
                link_com = link.get_local_poses()[0]
            self.link_coms.append(link_com)
        self.link_coms = torch.cat(self.link_coms, dim=1)
        return self.link_coms

    
