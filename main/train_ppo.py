import ray
from ray import tune
from ray.rllib.env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import gym
import gym_multi_car_racing
import numpy as np
from gym import spaces

class DummyEnv(gym.Env):
    def __init__(self, env_config):
        super(DummyEnv, self).__init__()
        # Define a simple observation space and action space
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        # Return an initial random observation
        return np.random.rand(4)

    def step(self, action):
        # Return a random observation, zero reward, not done, and empty info
        obs = np.random.rand(4)
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

# Define a function to create the environment
def dummy_env_creator(env_config):
    return DummyEnv(env_config)

#======
# Define a custom multi-agent environment class
class MultiCarRacingEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env = gym.make("MultiCarRacing-v0", num_agents=2, direction='CCW',
               use_random_direction=True, backwards_flag=True, h_ratio=0.25,
               use_ego_color=False)
        self.num_agents = 2
        self.agents = ["agent_0", "agent_1"]
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        return {f"agent_{i}": obs[i] for i in range(self.num_agents)}

    def step(self, action_dict):
        actions = [action_dict[f"agent_{i}"] for i in range(self.num_agents)]
        obs, rewards, done, _ = self.env.step(actions)
        obs_dict = {f"agent_{i}": obs[i] for i in range(self.num_agents)}
        rewards_dict = {f"agent_{i}": rewards[i] for i in range(self.num_agents)}
        dones_dict = {f"agent_{i}": done for i in range(self.num_agents)}
        dones_dict["__all__"] = done
        return obs_dict, rewards_dict, dones_dict, {}

# Define a function to create the environment
def multi_car_racing_env_creator(env_config):
    return MultiCarRacingEnv(env_config)
#========

# Initialize Ray
ray.init()

# Register the environment
register_env("multi_car_racing", multi_car_racing_env_creator)

# Configure the PPO algorithm
config = PPOConfig()
config = config.environment("multi_car_racing", env_config={})
config = config.framework("torch")  # Use "tf" if you prefer TensorFlow
config = config.rollouts(num_rollout_workers=1)  # Increase for more parallelism
config = config.training(
    gamma=0.99,
    lr=0.0003,
    train_batch_size=4000,
    clip_param=0.2,
    use_gae=True,
    lambda_=0.95,
    vf_clip_param=10.0,
    vf_loss_coeff=1.0,
    entropy_coeff=0.01,
    sgd_minibatch_size=128,
    num_sgd_iter=10
)
config = config.multi_agent(
    policies={"shared_policy": (None, MultiCarRacingEnv({}).observation_space, MultiCarRacingEnv({}).action_space, {})},
    policy_mapping_fn=lambda agent_id: "shared_policy"  # Map both agents to the shared policy
)

# # Build the PPO algorithm
algo = config.build()

# Train the model
for i in range(1000):  # Number of iterations
    result = algo.train()
    print(f"Iteration {i}: mean reward {result['episode_reward_mean']}")

# Save the trained model
algo.save("multi_car_racing_ppo_model")

# Shut down Ray
ray.shutdown()
