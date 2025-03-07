# Dynamic Crowd Simulation with Realistic Behavior Modeling

This project presents a framework for simulating dynamic crowds with realistic behavior. The system uses a combination of a Graph Neural Network (GNN) + Transformer model to predict pedestrian trajectories and a Proximal Policy Optimization (PPO) agent to control an AI agent navigating through the crowd while respecting boundaries and obstacles.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Approach Overview](#approach-overview)
- [Model Architecture](#model-architecture)
  - [GNN + Transformer for Trajectory Prediction](#gnn--transformer-for-trajectory-prediction)
  - [PPO for Agent Navigation](#ppo-for-agent-navigation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Evaluation and Visualization](#evaluation-and-visualization)
- [Challenges and Solutions](#challenges-and-solutions)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Conclusion](#conclusion)

## Introduction

This project aims to create a realistic simulation of crowd behavior by modeling pedestrian trajectories and then using these predictions to inform the navigation of an autonomous agent. The system integrates two primary components:
1. **Trajectory Prediction:** A hybrid GNN+Transformer network forecasts the future positions of pedestrians.
2. **Reinforcement Learning:** A PPO agent is trained to navigate safely through the crowd, reaching designated destinations while avoiding obstacles.

## Dataset

The project uses the **ETH Walking Pedestrians (EWAP)** dataset. The dataset includes:
- **OBSMAT:** Contains pedestrian annotations with each line in the format:  
  `frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y`  
  *(Note: `pos_z` and `v_z` are not used.)*  
  Positions and velocities are in meters and are obtained by applying the provided homography matrix.
- **DESTINATIONS:** A file (`destinations.txt`) listing assumed destination points for pedestrians.
- **OBSTACLES:** An obstacle map (`map.png`) representing obstacles in the scene.  
  A homography matrix (`H.txt`) is used to convert between the map image and world coordinates.
- **GROUPS (Optional):** Contains IDs of pedestrians walking in groups (not used directly in this project).
- **LINK:** https://icu.ee.ethz.ch/research/datsets.html (BIWI Walking Pedestrians dataset)

## Problem Statement

The primary goal is to simulate a dynamic crowd and train an AI agent to navigate safely through it. The agent must:
- Predict and incorporate pedestrian trajectories,
- Avoid collisions with pedestrians and obstacles,
- Remain within the boundaries of the scene,
- Reach a specified destination.

## Approach Overview

The project is divided into the following main components:
1. **Data Preprocessing:** Extract and normalize pedestrian trajectories from OBSMAT, build spatial graphs, and generate sequence windows.
2. **Trajectory Prediction Model:** A hybrid model combining a GNN and a Transformer predicts future positions of pedestrians.
3. **Reinforcement Learning Agent:** A PPO agent uses the predicted pedestrian trajectories to learn a navigation policy.
4. **Integration and Visualization:** The trained model is integrated with a simulation environment and overlaid on video (e.g., `seq_eth.avi`) for visual validation.

## Model Architecture

### GNN + Transformer for Trajectory Prediction

- **GNN Module:**  
  Uses Graph Convolutional Networks (GCNConv layers) to capture spatial relationships between pedestrians. Each node represents a pedestrian, and edges represent proximity.
  
- **Transformer Module:**  
  Processes the sequence of spatial features (one per time step) using self-attention. This captures the temporal evolution of the crowd.

- **Output:**  
  The combined model outputs predicted pedestrian positions for a time window.

### PPO for Agent Navigation

- **State:**  
  The agent’s observation includes its own 2D position and a fixed number of predicted pedestrian positions (converted to world coordinates).
  
- **Action:**  
  A continuous 2D velocity command that updates the agent’s position.

- **Reward Function:**  
  The reward is structured as follows:
  - **Per-Step Reward:** A small positive reward for staying inside the allowed area (a defined rectangular boundary).
  - **Collision Penalty:** A penalty for colliding with or coming too close to a pedestrian.
  - **Goal Bonus:** A large bonus when the agent reaches the destination.
  
- **Training:**  
  The PPO agent is trained to maximize cumulative reward over episodes.

## Preprocessing

- **Data Loading:**  
  The OBSMAT file is read and relevant columns (frame, pedestrian_ID, pos_x, pos_y, etc.) are extracted and normalized.
  
- **Trajectory Extraction:**  
  Data is grouped by agent and segmented into fixed-length windows.
  
- **Graph Construction:**  
  For each frame, spatial graphs are built where nodes represent pedestrians and edges are formed based on distance thresholds.

- **Positional Encoding:**  
  Positional encodings are generated and concatenated with pedestrian features before being fed to the Transformer.

## Training

- **Trajectory Model Training:**  
  The GNN+Transformer model is trained to forecast pedestrian trajectories using metrics like Average Displacement Error (ADE) and Final Displacement Error (FDE).

- **PPO Agent Training:**  
  The environment is wrapped (with DummyVecEnv and VecNormalize), and the PPO agent is trained using the defined reward function.  
  Training metrics (e.g., mean reward, loss, value loss) are monitored via Stable-Baselines3 logging.

## Evaluation and Visualization

- **Quantitative Evaluation:**  
  The PPO agent is evaluated using average cumulative reward and episode length.
  
- **Graphical Visualization:**  
  Two methods are used:
  - **Matplotlib Visualization:** The simulation state (agent, goal, pedestrian predictions) is plotted on a graph.
  - **Video Overlay:** The simulation state is overlaid on the original video (`seq_eth.avi`) using a world-to-pixel transformation with the homography matrix.
  
- **Mapping Debugging:**  
  A separate test script was used to overlay destination points on the obstacle map to verify that the homography transformation is correct.

## Challenges and Solutions

1. **Mapping World Coordinates:**  
   - **Problem:** The homography matrix did not correctly map world coordinates to pixel coordinates, so dots did not appear on the video.  
   - **Solution:** Debugged by overlaying a test point on the map image. Printed out pixel values to verify alignment.

2. **Reward Function Design:**  
   - **Problem:** Early reward designs penalized distance to goal, resulting in high negative cumulative rewards.  
   - **Solution:** Modified reward logic to reward staying inside the allowed region, penalize collisions, and give a bonus for reaching the goal.

3. **Handling Variable Numbers of Pedestrians:**  
   - **Problem:** The number of agents varied between frames, complicating input for the network.  
   - **Solution:** Fixed the observation space by padding or truncating to a fixed number (`max_agents`).

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/dynamic-crowd-simulation.git
   cd dynamic-crowd-simulation
