import sys
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2022.2.1/exts')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/robots')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/envs')

from vec_env_base_custom import VecEnvBase

env = VecEnvBase()
from ur5_view import UR5View
from ur5 import UR5

usd_path = "/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/assets/ur5/ur5.usd"
ur5 = UR5(prim_path="/ur5", usd_path=usd_path)

ur5_view = UR5View(prim_paths_expr = "/ur5", name = "ur5")
dof_limits = ur5_view.get_dof_limits()
lower_limits = dof_limits[0, :, 0]
upper_limits = dof_limits[0, :, 1]

print('lower limits: ', lower_limits)
print('upper limits: ', upper_limits)