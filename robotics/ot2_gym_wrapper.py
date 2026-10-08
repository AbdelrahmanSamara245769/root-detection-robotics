import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p

from project_paths import enter_simulation_dir

enter_simulation_dir()
from sim_class import Simulation  # noqa: E402

# OT-2 Workspace limits (approximate based on Task 10)
WORKSPACE_X = (-0.187, 0.253)
WORKSPACE_Y = (-0.171, 0.220)
WORKSPACE_Z = (0.169, 0.290)

class OT2ReachEnv(gym.Env):
    def __init__(self, render=False, max_steps=1000, tol=0.001, action_scale=0.08):
        super(OT2ReachEnv, self).__init__()
        self.render_sim = render
        self.max_steps = max_steps
        self.tol = tol
        self.action_scale = action_scale
        
        # ACTION SPACE:
        # Velocity control (x, y, z). We use -1 to 1 and scale it later.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # OBSERVATION SPACE:
        # [pipette_x, pipette_y, pipette_z, target_x, target_y, target_z]
        # Use -inf to inf to allow VecNormalize to work without bounds errors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.sim = Simulation(num_agents=1, render=self.render_sim)
        self.target = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # 1. Randomize Target
        self.target = np.array([
            self.np_random.uniform(WORKSPACE_X[0], WORKSPACE_X[1]),
            self.np_random.uniform(WORKSPACE_Y[0], WORKSPACE_Y[1]),
            self.np_random.uniform(WORKSPACE_Z[0], WORKSPACE_Z[1])
        ], dtype=np.float32)

        # 2. Randomize Robot Start Position (The "Teleport" Fix)
        # We manually set joints to random values within valid range to speed up exploration
        # Range is approx -0.1 to 0.1 for X/Y, 0 to 0.2 for Z (based on your joint limits)
        random_joints = [
            self.np_random.uniform(-0.1, 0.1), # Joint 0 (X/Y)
            self.np_random.uniform(-0.1, 0.1), # Joint 1 (Y/X)
            self.np_random.uniform(0.0, 0.15)  # Joint 2 (Z)
        ]
        
        # Access the robot ID from the sim class
        robot_id = self.sim.robotIds[0]
        # We need to ensure pybullet is available here since the user snippet imported it inside reset
        import pybullet as p
        p.resetJointState(robot_id, 0, random_joints[0])
        p.resetJointState(robot_id, 1, random_joints[1])
        p.resetJointState(robot_id, 2, random_joints[2])

        # 3. Get Observation
        # Pass num_steps=10 to let physics settle slightly
        state = self.sim.run([[0, 0, 0, 0]], num_steps=10) 
        self.pipette_pos = self._extract_pos(state)
        # Safe access to prev_dist
        self.prev_dist = np.linalg.norm(self.pipette_pos - self.target)
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        
        # 1. Action Scaling & Application
        # Clip action to be safe, then scale
        action = np.clip(action, -1.0, 1.0)
        vel = action * self.action_scale
        
        # CRITICAL FIX: Run 10 physics steps for every 1 RL step
        # This speeds up training by 10x and makes movement visible
        state = self.sim.run([[vel[0], vel[1], vel[2], 0]], num_steps=10)
        
        # 2. Update State
        self.pipette_pos = self._extract_pos(state)
        dist = np.linalg.norm(self.pipette_pos - self.target)
        
        # 3. Check Termination
        terminated = bool(dist < self.tol)
        truncated = bool(self.steps >= self.max_steps)

        # 4. Reward Calculation
        # Base: Negative distance (Standard dense reward)
        reward = -dist 
        
        # Bonus: Progress (encourages moving closer this specific step)
        reward += 2.0 * (self.prev_dist - dist)
        
        # Penalty: Energy/Magnitude (prevents violent shaking)
        reward -= 0.05 * np.mean(np.square(action)) 
        
        # Bonus: Success (The "Jackpot")
        if terminated:
            reward += 10.0

        # Update previous distance for next step
        self.prev_dist = dist
        
        # 5. Info Payload (Crucial for metrics)
        info = {
            "dist": dist,              # For logging
            "is_success": terminated   # For success rate calc
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        """Helper to construct the 6D observation vector"""
        return np.concatenate([self.pipette_pos, self.target]).astype(np.float32)

    def _extract_pos(self, state):
        """Helper to parse sim_class dictionary output"""
        # sim returns {robot_id: {...}}, we take the first value
        key = list(state.keys())[0]
        return np.array(state[key]['pipette_position'], dtype=np.float32)

    def close(self):
        self.sim.close()