import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


class DroneEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # ==========================================
        # ACTIONS
        # ==========================================
        # 0 = Forward
        # 1 = Left
        # 2 = Right
        # 3 = Up
        # 4 = Down

        self.action_space = spaces.Discrete(5)

        # ==========================================
        # OBSERVATION
        # ==========================================
        # Drone X, Y, Z
        # Target X, Y, Z
        # Static obstacle distance
        # Dynamic obstacle distance

        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(8,),
            dtype=np.float32
        )

        # ==========================================
        # INITIAL POSITION
        # ==========================================

        self.drone = np.array(
            [0.0, 0.0, 3.0],
            dtype=np.float32
        )

        # ==========================================
        # TARGET
        # ==========================================

        self.target = np.array(
            [10.0, 0.0, 3.0],
            dtype=np.float32
        )

        # ==========================================
        # STATIC OBSTACLE
        # ==========================================

        self.static_obstacle = np.array(
            [3.0, 0.0, 1.0],
            dtype=np.float32
        )

        # ==========================================
        # DYNAMIC OBSTACLE
        # ==========================================

        self.dynamic_obstacle = np.array(
            [6.0, 2.0, 1.5],
            dtype=np.float32
        )

        self.step_count = 0
        self.max_steps = 500

    # ==============================================
    # RESET
    # ==============================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.drone = np.array(
            [0.0, 0.0, 3.0],
            dtype=np.float32
        )

        self.dynamic_obstacle = np.array(
            [6.0, 2.0, 1.5],
            dtype=np.float32
        )

        self.step_count = 0

        observation = self._get_observation()

        info = {}

        return observation, info

    # ==============================================
    # OBSERVATION
    # ==============================================

    def _get_observation(self):

        static_distance = np.linalg.norm(
            self.drone - self.static_obstacle
        )

        dynamic_distance = np.linalg.norm(
            self.drone - self.dynamic_obstacle
        )

        observation = np.array([
            self.drone[0],
            self.drone[1],
            self.drone[2],

            self.target[0],
            self.target[1],
            self.target[2],

            static_distance,
            dynamic_distance

        ], dtype=np.float32)

        return observation

    # ==============================================
    # STEP
    # ==============================================

    def step(self, action):

        self.step_count += 1

        old_distance = np.linalg.norm(
            self.drone - self.target
        )

        # ==========================================
        # ACTION
        # ==========================================

        movement = 0.5

        if action == 0:
            # Forward
            self.drone[0] += movement

        elif action == 1:
            # Left
            self.drone[1] += movement

        elif action == 2:
            # Right
            self.drone[1] -= movement

        elif action == 3:
            # Up
            self.drone[2] += movement

        elif action == 4:
            # Down
            self.drone[2] -= movement

        # ==========================================
        # DYNAMIC OBSTACLE MOVEMENT
        # ==========================================

        self.dynamic_obstacle[1] = (
            2.0 +
            2.0 * math.sin(self.step_count * 0.1)
        )

        # ==========================================
        # NEW DISTANCE
        # ==========================================

        new_distance = np.linalg.norm(
            self.drone - self.target
        )

        # ==========================================
        # REWARD
        # ==========================================

        reward = 0.0

        # Getting closer
        if new_distance < old_distance:
            reward += 1.0
        else:
            reward -= 0.5

        # ==========================================
        # COLLISION CHECK
        # ==========================================

        static_distance = np.linalg.norm(
            self.drone - self.static_obstacle
        )

        dynamic_distance = np.linalg.norm(
            self.drone - self.dynamic_obstacle
        )

        terminated = False

        # Static collision
        if static_distance < 1.2:

            reward -= 100

            terminated = True

        # Dynamic collision
        elif dynamic_distance < 1.2:

            reward -= 100

            terminated = True

        # ==========================================
        # TARGET REACHED
        # ==========================================

        if new_distance < 0.5:

            reward += 100

            terminated = True

        # ==========================================
        # MAX STEPS
        # ==========================================

        truncated = False

        if self.step_count >= self.max_steps:

            truncated = True

        observation = self._get_observation()

        info = {}

        return observation, reward, terminated, truncated, info