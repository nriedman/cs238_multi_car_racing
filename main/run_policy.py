import gym
import gym_multi_car_racing
from stable_baselines3 import PPO
import numpy as np

env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
               use_random_direction=True, backwards_flag=True, h_ratio=0.25,
               use_ego_color=False)
# Load the model for evaluation
model = PPO.load("ppo_multi_car_racing")

# Evaluate the trained model
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
