# from omni.isaac.kit import SimulationApp

# simulation_app = SimulationApp({"headless": False})

import gym
from vec_env_base_custom import VecEnvBase
# from omni.isaac.gym.vec_env import VecEnvMT

env = VecEnvBase(headless=False)

# create task and register task
from multiarm_task import MultiarmTask

task = MultiarmTask(name="Multiarm")
env.set_task(task, backend="torch")


num_episodes = 100  # Define the number of episodes for testing

for episode in range(num_episodes):
    # observation, action = env.reset()
    if episode > 0:
        # observation, action = env.reset()
        env.reset()
    done = False
    while not done:
        # Select a random action
        action = env.action_space.sample() # need to reshape the action to num_agents * 6

        # Step through the environment
        observation, reward, done, info = env.step(action)

        # Optionally print out step information
        print(f"Episode: {episode}, Step: {action}, Reward: {reward}")

    print(f"Episode {episode} finished")

env.close()
