import gym
import gym_multi_car_racing
import numpy as np

# Create the environment
env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

# Reset environment
obs = env.reset()
done = False
total_reward = 0

# Dummy policy function: return random actions for each agent
def my_policy(observation):
    # For testing purposes, we'll just return random actions for each agent
    # The action is a 2D array: (num_agents, 3)
    num_agents = observation.shape[0]  # Get the number of agents
    # Actions are continuous and between -1 and 1, for the 3 control outputs (steering, throttle, brake)
    return np.random.uniform(-1, 1, (num_agents, 3))

# Main loop to interact with the environment
for i in range(10000):
    # Get actions from the dummy policy
    action = my_policy(obs)
    
    # Take a step in the environment with the generated actions
    obs, reward, done, info = env.step(action)
    
    # Accumulate the total reward
    total_reward += reward
    
    # Render the environment for visualization (optional)
    env.render()

# Print the total reward across all agents
print("individual scores:", total_reward)
