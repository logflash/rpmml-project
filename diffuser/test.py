import environments.maze2d_env
import gymnasium as gym

env = gym.make("Maze2D-v0")
obs, info = env.reset()
print("obs:", obs)
obs, reward, done, truncated, info = env.step(env.action_space.sample())
print("step output:", obs, reward, done, truncated)
