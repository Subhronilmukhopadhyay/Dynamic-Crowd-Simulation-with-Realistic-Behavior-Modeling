import gym
import numpy as np
import torch
from gym import spaces
import cv2
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# -------------------------------
# Environment with Destination and Obstacle Constraints
# -------------------------------
class CrowdSimEnvWithBoundaries(gym.Env):
    """
    This environment uses a pre-trained GNN+Transformer model to predict pedestrian positions.
    It then sets a random destination (from destinations.txt) as the goal and checks for collisions
    with obstacles derived from map.png using a homography matrix (H.txt). The agent's state (agent position
    and predicted pedestrian positions) and actions are in world coordinates.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, model, pedestrian_data, pos_min, pos_max, destinations_file, map_file, homography_file,
                 max_steps=50, max_agents=6):
        """
        Args:
          model: Pre-trained GNN+Transformer model (trained on normalized coordinates).
          pedestrian_data: List of samples from your BIWIDataset.
          pos_min: 2D vector [min_x, min_y] from the original MinMaxScaler.
          pos_max: 2D vector [max_x, max_y] from the original MinMaxScaler.
          destinations_file: Path to destinations.txt.
          map_file: Path to map.png.
          homography_file: Path to H.txt.
          max_steps: Maximum steps per episode.
          max_agents: Fixed number of pedestrian agents in the observation.
        """
        super(CrowdSimEnvWithBoundaries, self).__init__()
        self.model = model
        self.model.eval()
        self.pos_min = pos_min  # world coordinate minimum (e.g., from original data)
        self.pos_max = pos_max  # world coordinate maximum
        self.max_steps = max_steps
        self.max_agents = max_agents
        
        # Load destinations (each row: [x, y]) in world coordinates.
        self.destinations = np.loadtxt(destinations_file)  # shape (num_destinations, 2)
        self.goal = None  # to be selected in reset()
        
        # Load obstacle map image (grayscale) and homography matrix.
        self.map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise RuntimeError("Failed to load map image")
        self.H = np.loadtxt(homography_file)  # 3x3 matrix
        
        # pedestrian_data: list of samples as (x_dict_list, edge_index_dict_list, ground_truth)
        self.pedestrian_data = pedestrian_data
        
        # Observation: agent position (2D) plus predicted pedestrian positions (max_agents x 2).
        # Here, both agent and pedestrian positions are in world coordinates.
        self.obs_dim = 2 + self.max_agents * 2
        # Define observation space using known world bounds.
        low_obs = np.concatenate([self.pos_min, np.tile(self.pos_min, self.max_agents)])
        high_obs = np.concatenate([self.pos_max, np.tile(self.pos_max, self.max_agents)])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        
        # Action: 2D velocity command in world units.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Internal variables
        self.current_sample = None
        self.x_dict_list = None
        self.edge_index_dict_list = None
        self.ground_truth = None
        self.predicted_pedestrians = None  # in world coordinates
        self.agent_pos = None  # in world coordinates
        self.current_step = 0
        
        self.reset()
    
    def denormalize(self, norm_pos):
        """
        Convert normalized position (in [0,1]) to world coordinates using pos_min and pos_max.
        """
        return self.pos_min + norm_pos * (self.pos_max - self.pos_min)
    
    def is_in_obstacle(self, world_pos, threshold=100):
        """
        Check if the given world coordinate is inside an obstacle.
        The world coordinate is transformed to pixel coordinates using the homography matrix.
        """
        point = np.array([world_pos[0], world_pos[1], 1.0])
        p = self.H @ point
        p = p / p[2]
        px = int(round(p[0]))
        py = int(round(p[1]))
        h, w = self.map_img.shape
        if px < 0 or px >= w or py < 0 or py >= h:
            return False  # outside map: assume free space
        return self.map_img[py, px] < threshold  # obstacles assumed to be dark
    
    def reset(self):
        # Choose a random pedestrian sample from the dataset.
        self.current_sample = random.choice(self.pedestrian_data)
        self.x_dict_list, self.edge_index_dict_list, self.ground_truth = self.current_sample
        
        # Use the pre-trained model to predict pedestrian positions for the first time step.
        with torch.no_grad():
            output, all_agent_ids, src_key_padding_mask = self.model(self.x_dict_list, self.edge_index_dict_list)
            # output shape: (T, num_agents, output_dim); take time 0, first 2 dims (normalized positions).
            pred_norm = output[0, :, :2].cpu().numpy()  # shape: (n_agents, 2)
        
        # Denormalize pedestrian positions to world coordinates.
        pred_world = self.denormalize(pred_norm)
        n_agents = pred_world.shape[0]
        if n_agents < self.max_agents:
            pad_width = self.max_agents - n_agents
            padded_pred = np.concatenate([pred_world, np.zeros((pad_width, 2))], axis=0)
        else:
            padded_pred = pred_world[:self.max_agents]
        self.predicted_pedestrians = padded_pred  # shape: (max_agents, 2)
        
        # Set the agent's starting position. Here, we choose the center of the scene.
        self.agent_pos = (self.pos_min + self.pos_max) / 2.0
        
        # Randomly select a destination from the loaded destinations.
        self.goal = random.choice(self.destinations)
        
        self.current_step = 0
        
        obs = np.concatenate([self.agent_pos, self.predicted_pedestrians.flatten()])
        return obs.astype(np.float32)
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update the agent's position using simple Euler integration.
        self.agent_pos = self.agent_pos + action

        # Build observation.
        obs = np.concatenate([self.agent_pos, self.predicted_pedestrians.flatten()])

        # Initialize reward.
        reward = 0.0

        # 1. Per-step reward for staying inside the allowed area.
        # Here we give a small positive reward if the agent is within the boundaries.
        if (self.agent_pos[0] >= self.pos_min[0] and self.agent_pos[0] <= self.pos_max[0] and
            self.agent_pos[1] >= self.pos_min[1] and self.agent_pos[1] <= self.pos_max[1]):
            reward += 0.1  # Small reward for being inside
        else:
            reward -= 0.5  # Penalty for leaving the allowed area

        # 2. Collision penalty: subtract if the agent is too close to any pedestrian.
        collision_threshold = 0.5  # Adjust as needed (in world units)
        for ped in self.predicted_pedestrians:
            if np.linalg.norm(self.agent_pos - ped) < collision_threshold:
                reward -= 1.0  # Penalty for collision

        # 3. Goal bonus: if the agent is close enough to the goal, provide a large positive reward and end the episode.
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal)
        if dist_to_goal < 1.0:  # If within 1 world unit of the goal.
            reward += 20.0  # Large bonus for reaching the goal.
            done = True
        else:
            done = False

        # End the episode if maximum steps are reached.
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return obs.astype(np.float32), reward, done, {}

    
    def render(self, mode="human"):
        # Render by overlaying the agent, goal, and pedestrian predictions on the obstacle map.
        map_color = cv2.cvtColor(self.map_img, cv2.COLOR_GRAY2BGR)
        
        def world_to_pixel(world_pos):
            point = np.array([world_pos[0], world_pos[1], 1.0])
            p = self.H @ point
            p = p / p[2]
            return int(round(p[0])), int(round(p[1]))
        
        # Draw goal (green circle)
        goal_px, goal_py = world_to_pixel(self.goal)
        cv2.circle(map_color, (goal_px, goal_py), 8, (0, 255, 0), -1)
        # Draw agent (red circle)
        agent_px, agent_py = world_to_pixel(self.agent_pos)
        cv2.circle(map_color, (agent_px, agent_py), 8, (0, 0, 255), -1)
        # Draw predicted pedestrians (blue circles)
        for ped in self.predicted_pedestrians:
            ped_px, ped_py = world_to_pixel(ped)
            cv2.circle(map_color, (ped_px, ped_py), 6, (255, 0, 0), -1)
        
        cv2.imshow("Crowd Simulation with Boundaries", map_color)
        cv2.waitKey(30)

# -------------------------------
# Example Setup and PPO Integration
# -------------------------------

# Define world bounds (from the original scaler, adjust these as needed)
pos_min = np.array([-20.0, 0.0], dtype=np.float32)
pos_max = np.array([15.0, 12.0], dtype=np.float32)

# File paths for destinations, map, and homography matrix.
destinations_file = "ewap_dataset/seq_eth/destinations.txt"
map_file = "ewap_dataset/seq_eth/map.png"
homography_file = "ewap_dataset/seq_eth/H.txt"

# Assume 'dataset' is your BIWIDataset instance created earlier.
pedestrian_data = [dataset[i] for i in range(len(dataset))]

# Create the environment with boundaries.
env_bound = CrowdSimEnvWithBoundaries(model, pedestrian_data, pos_min, pos_max,
                                       destinations_file, map_file, homography_file,
                                       max_steps=50, max_agents=6)

# Create and train a PPO agent on the new environment.
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

# Wrap your environment in a DummyVecEnv
env = DummyVecEnv([lambda: env_bound])  # env_bound is your CrowdSimEnvWithBoundaries instance

# Normalize observations and rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Adjust PPO hyperparameters (for example, lowering vf_coef and using a lower learning rate)
ppo_agent_bound = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    vf_coef=0.3,            # Adjusted value function coefficient
    learning_rate=0.0002,   # Possibly lower learning rate for stability
)

# Train the PPO agent
ppo_agent_bound.learn(total_timesteps=20000)
ppo_agent_bound.save("ppo_crowd_sim_agent_bound_normalized")

# Test the trained agent.
obs = env_bound.reset()
done = False
while not done:
    action, _ = ppo_agent_bound.predict(obs)
    obs, reward, done, info = env_bound.step(action)
    env_bound.render()