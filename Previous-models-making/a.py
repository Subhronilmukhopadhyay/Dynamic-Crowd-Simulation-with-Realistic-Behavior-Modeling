import gym
import numpy as np
import torch
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import random

# -------------------------------
# Define the Custom Gym Environment with Fixed Observation Space
# -------------------------------
class CrowdSimEnv(gym.Env):
    """
    A crowd simulation environment that uses a pre-trained GNN+Transformer model 
    to predict pedestrian positions. The observation consists of the agent's 2D position 
    and the predicted 2D positions of a fixed number of pedestrians (max_agents).
    The agent's action is a continuous 2D velocity command.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, model, pedestrian_data, goal=np.array([1.0, 1.0], dtype=np.float32),
                 max_steps=50, max_agents=6):
        """
        Args:
          model: Pre-trained GNN+Transformer model.
          pedestrian_data: List of samples from the dataset.
          goal: The goal position (normalized).
          max_steps: Maximum steps per episode.
          max_agents: The fixed number of pedestrian agents used for observations.
        """
        super(CrowdSimEnv, self).__init__()
        self.model = model  # pre-trained model; ensure it is set to eval() mode outside
        self.goal = goal.astype(np.float32)
        self.max_steps = max_steps
        self.max_agents = max_agents
        
        # pedestrian_data: list of samples, where each sample is (x_dict_list, edge_index_dict_list, ground_truth)
        self.pedestrian_data = pedestrian_data
        
        # Define observation space: agent 2D position + fixed number of pedestrian positions (max_agents x 2)
        self.obs_dim = 2 + self.max_agents * 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        # Action: continuous 2D velocity command
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        
        self.current_sample = None
        self.agent_pos = None
        self.predicted_pedestrians = None  # will be of shape (num_agents, 2) from model
        self.current_step = 0
        
        self.reset()
        
    def reset(self):
        # Use Python's random.choice to select a sample
        self.current_sample = random.choice(self.pedestrian_data)
        self.x_dict_list, self.edge_index_dict_list, self.ground_truth = self.current_sample
        
        # Use the pre-trained model to predict pedestrian trajectories for the window.
        # We use the first time step's prediction.
        with torch.no_grad():
            output, all_agent_ids, src_key_padding_mask = self.model(self.x_dict_list, self.edge_index_dict_list)
            # output: (T, num_agents, output_dim); we take time 0 and first 2 dims (pos_x, pos_y)
            pred = output[0, :, :2].cpu().numpy()  # shape: (n_agents, 2)
        
        # Now, adjust predicted pedestrian positions to have fixed size max_agents:
        n_agents = pred.shape[0]
        if n_agents < self.max_agents:
            # pad with zeros (or a neutral value) for missing agents
            pad_width = self.max_agents - n_agents
            padded_pred = np.concatenate([pred, np.zeros((pad_width, 2))], axis=0)
        else:
            # if there are more than max_agents, take the first max_agents
            padded_pred = pred[:self.max_agents]
        self.predicted_pedestrians = padded_pred  # shape: (max_agents, 2)
        
        # Reset agent position (for example, starting at [0, 0])
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.current_step = 0
        
        # Build initial observation: agent position (2D) concatenated with flattened pedestrian positions.
        obs = np.concatenate([self.agent_pos, self.predicted_pedestrians.flatten()])
        return obs
        
    def step(self, action):
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update agent position
        self.agent_pos = self.agent_pos + action
        # Observation remains constant for pedestrians in this simplified example
        obs = np.concatenate([self.agent_pos, self.predicted_pedestrians.flatten()])
        
        # Reward: negative distance to goal plus penalty for close proximity to pedestrians
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal)
        reward = -dist_to_goal
        collision_threshold = 0.05
        for ped in self.predicted_pedestrians:
            if np.linalg.norm(self.agent_pos - ped) < collision_threshold:
                reward -= 1.0
        
        self.current_step += 1
        done = False
        if dist_to_goal < 0.05:
            reward += 10.0  # bonus for reaching goal
            done = True
        elif self.current_step >= self.max_steps:
            done = True
        
        return obs, reward, done, {}
    
    def render(self, mode="human"):
        plt.figure(figsize=(6,6))
        plt.scatter(self.agent_pos[0], self.agent_pos[1], c='red', label="Agent")
        plt.scatter(self.goal[0], self.goal[1], c='green', label="Goal")
        plt.scatter(self.predicted_pedestrians[:, 0], self.predicted_pedestrians[:, 1],
                    c='blue', label="Pedestrians")
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.title("Crowd Simulation Environment")
        plt.show()

# -------------------------------
# Prepare Pedestrian Data List from the Dataset
# -------------------------------
# Assume 'dataset' is your BIWIDataset instance (already created)
pedestrian_data = [dataset[i] for i in range(len(dataset))]

# -------------------------------
# Create the Environment Instance
# -------------------------------
env = CrowdSimEnv(model, pedestrian_data, goal=np.array([1.0, 1.0], dtype=np.float32), max_steps=50, max_agents=6)

# -------------------------------
# Create and Train a PPO Agent using Stable-Baselines3
# -------------------------------
ppo_agent = PPO("MlpPolicy", env, verbose=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Train the PPO agent
ppo_agent.learn(total_timesteps=10000)

# Save the trained PPO model
ppo_agent.save("ppo_crowd_sim_agent")

# -------------------------------
# Test the Trained PPO Agent
# -------------------------------
obs = env.reset()
done = False
while not done:
    action, _ = ppo_agent.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
