from stable_baselines3.common.env_checker import check_env
from omni.isaac.gym.vec_env import VecEnvBase
# from omni.isaac.gym.vec_env import VecEnvMT

env = VecEnvBase(headless=False)

# create task and register task
from cartpole_task import CartpoleTask

task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")

check_env(env)