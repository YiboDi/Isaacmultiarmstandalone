from ur5 import UR5

class UR5withEEandTarget(UR5):
    def __init__(self, name, offset=None, target_pos=None, target_ori=None) -> None:
        super(). __init__(
            prim_path: str,
            name: Optional[str] = "ur5",
            usd_path: Optional[str] = None,
            translation: Optional[torch.tensor] = None,
            orientation: Optional[torch.tensor] = None,
            default_dof_pos: dict = None, # default_dof_pos should follow the start_config of the specific ur5
            )   
        
        self.ee = VisualCylinder(prim_path=prim_paths_expr + "/ee_link/ee", radius=0.02, height=0.1, name=name + 'EE')
        self.target = VisualCylinder(prim_path=prim_paths_expr + "/target", radius=0.02, height=0.1,
                                    color=np.array([0.8, 0.8, 0.8]),
                                    translation=target_pos if target_pos is not None else [0,0,-5],
                                    orientation=target_ori if target_ori is not None else None,
                                    name=name + 'Target')
