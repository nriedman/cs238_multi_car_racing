import ray
import gymnasium as gym
# import gym_multi_car_racing
import numpy as np
from gym import spaces
from ray.rllib.algorithms import ppo

class DummyEnv(gym.Env):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    def reset(self, seed, options):
        return self.env.reset(seed, options)
    def step(self, action):
        return self.env.step(action)

# class DummyEnv(gym.Env):
#     def __init__(self, env_config):
#         super(DummyEnv, self).__init__()
#         # Define a simple observation space and action space
#         self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
#         self.action_space = spaces.Discrete(2)

#     def reset(self):
#         # Return an initial random observation
#         return np.random.rand(4)

#     def step(self, action):
#         # Return a random observation, zero reward, not done, and empty info
#         obs = np.random.rand(4)
#         reward = 0
#         done = False
#         info = {}
#         return obs, reward, done, info

print(type(DummyEnv))
ray.init()
algo = ppo.PPO(env=DummyEnv, config={
    "env_config": {},  # config to pass to env class
})

for i in range(len(2)):
    algo.train()
    print("hi")
print("finished")

ray.shutdown()
