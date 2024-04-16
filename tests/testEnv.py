import sys 

sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/tasks')
sys.path.append('/home/tp2/.local/share/ov/pkg/isaac_sim-2023.1.1/Isaacmultiarmstandalone/envs')
import gym
from vec_env_base_custom import VecEnvBase
# from omni.isaac.gym.vec_env import VecEnvMT

env = VecEnvBase(headless=False)

# create task and register task
from multiarm_task import MultiarmTask

task = MultiarmTask(name="Multiarm")
env.set_task(task, backend="torch")


num_episodes = 10  # Define the number of episodes for testing

for episode in range(num_episodes):
    # observation, action = env.reset()
    observation = env.reset()
    done = False
    while not done:
        # Select a random action
        action = env.action_space.sample()

        # Step through the environment
        observation, reward, done, info = env.step(action)

        # Optionally print out step information
        print(f"Episode: {episode}, Step: {action}, Reward: {reward}")

    print(f"Episode {episode} finished")

env.close()
