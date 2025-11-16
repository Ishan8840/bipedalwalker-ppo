import numpy as np
import gymnasium as gym
import math

class GaitWrapper(gym.Wrapper):
    def __init__(self, env, T=20, d_lower=-0.3):
        """
        T = gait period (in environment steps)
        d_lower = stance/swing threshold (-1 to 1)
        """
        super().__init__(env)
        self.T = T
        self.t = 0
        self.d_lower = d_lower

    def reset(self, **kwargs):
        self.t = 0
        obs, info = self.env.reset(**kwargs)
        return np.concatenate([obs, [0.0, 0.0]]), info

    def step(self, action):
        self.t += 1

        # compute phases
        # left: phase offset = 0
        # right: phase offset = 0.5 (opposite)
        phase_left  = math.sin(2 * math.pi * (self.t / self.T + 0.0))
        phase_right = math.sin(2 * math.pi * (self.t / self.T + 1.0))

        # original step
        obs, reward, terminated, truncated, info = self.env.step(action)

        # append phases to observation
        obs = np.concatenate([obs, [phase_left, phase_right]])

        # make phases accessible in info for reward shaping
        info["phase_left"] = phase_left
        info["phase_right"] = phase_right

        return obs, reward, terminated, truncated, info
