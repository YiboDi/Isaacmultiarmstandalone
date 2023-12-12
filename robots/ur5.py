
from typing import Optional
import math
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import PhysxSchema

from omni.isaac.core.prims import RigidPrim

class UR5(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "ur5",
        usd_path: Optional[str] = None,
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
        default_dof_pos: dict = None, # default_dof_pos should follow the start_config of the specific ur5
    ) -> None:
        """[summary]
        """

        self._usd_path = usd_path
        self._name = name

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.7071068, 0, 0, 0.7071068]) if orientation is None else orientation
        self._default_dof_pos = torch.tensor([0, 0, 0, 0, 0, 0]) if default_dof_pos is None else default_dof_pos
        add_reference_to_stage(usd_path=self._usd_path, prim_path=prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        # self.base_link = RigidPrim(prim_path=prim_path + "/base_link")
        # self.shoulder_link = RigidPrim(prim_path=prim_path + "/shoulder_link")
        # self.upper_arm_link = RigidPrim(prim_path=prim_path + "/upper_arm_link")
        # self.forearm_link = RigidPrim(prim_path=prim_path + "/forearm_link")
        # self.wrist_1_link = RigidPrim(prim_path=prim_path + "/wrist_1_link")
        # self.wrist_2_link = RigidPrim(prim_path=prim_path + "/wrist_2_link")
        # self.wrist_3_link = RigidPrim(prim_path=prim_path + "/wrist_3_link")
        # self.ee_link = RigidPrim(prim_path=prim_path + "/ee_link")
        # self.tool0 = RigidPrim(prim_path=prim_path + "/tool0")
        # self.world = RigidPrim(prim_path=prim_path + "/world")

        dof_paths = [
            "base_link/shoulder_pan_joint",
            "shoulder_link/shoulder_lift_joint",
            "upper_arm_link/elbow_joint",
            "forearm_link/wrist_1_joint",
            "wrist_1_link/wrist_2_joint",
            "wrist_2_link/wrist_3_joint"
        ]

        # drive_type = ["angular"] * 7 + ["linear"] * 2
        drive_type = ["angular"] * 6
        # default_dof_pos = [math.degrees(x) for x in [0.0, -1.0, 0.0, -2.2, 0.0, 2.4, 0.8]] + [0.02, 0.02]
        stiffness = [400*np.pi/180] * 6
        # damping = [80*np.pi/180] * 6
        # stiffness = [0] * 6
        damping = [0] * 6
        max_force = [150, 150, 150, 28, 28, 28] # from ur5.py
        max_velocity = [math.degrees(x) for x in [3.15, 3.15, 3.15, 3.2, 3.2, 3.2]] # radians to degrees
        default_dof_pos = torch.tensor(self._default_dof_pos)
        default_dof_pos = [math.degrees(x) for x in self._default_dof_pos] #radians to degrees 

        for i, dof in enumerate(dof_paths):
            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i]
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(max_velocity[i])
    # # test below
    # def prepare_contacts(self, stage, prim):
    #     for link_prim in prim.GetChildren():
    #         if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
    #             if "_HIP" not in str(link_prim.GetPrimPath()):
    #                 rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
    #                 rb.CreateSleepThresholdAttr().Set(0)
    #                 cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
    #                 cr_api.CreateThresholdAttr().Set(0)
    # # test
    # def set_anymal_properties(self, stage, prim):
    #     for link_prim in prim.GetChildren():
    #         if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
    #             rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
    #             rb.GetDisableGravityAttr().Set(False)
    #             rb.GetRetainAccelerationsAttr().Set(False)
    #             rb.GetLinearDampingAttr().Set(0.0)
    #             rb.GetMaxLinearVelocityAttr().Set(1000.0)
    #             rb.GetAngularDampingAttr().Set(0.0)
    #             rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)
