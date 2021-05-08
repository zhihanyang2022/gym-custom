import numpy as np
import gym
from gym import spaces

class ContinuousHeavenHellOptLower(gym.Env):

    """No velocity; just position and direction bit."""

    def __init__(self):

        self.min_position = -1.2
        self.max_position = 1.2

        self.min_action = self.min_position
        self.max_action = self.max_position

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        num_obs_to_concatenate = 10
        single_obs_dim = 2
        concatenated_obs_dim = num_obs_to_concatenate * single_obs_dim
        self.concatenated_obs_dim = concatenated_obs_dim

        single_obs_min = [self.min_position, -1.0]  # -1 for direction bit
        single_obs_max = [self.max_position,  1.0]
        concatenated_obs_min = np.array(single_obs_min * num_obs_to_concatenate, dtype=np.float32)
        concatenated_obs_max = np.array(single_obs_max * num_obs_to_concatenate, dtype=np.float32)

        # for concatenated observations
        self.observation_space = spaces.Box(
            low=concatenated_obs_min,
            high=concatenated_obs_max,
            dtype=np.float32
        )

        self.priest_position = 0.5
        self.priest_delta = 0.1

    def reset(self):

        self.state = np.random.uniform(low=-0.2, high=0.2)
        observation = np.zeros((self.concatenated_obs_dim, ))
        observation[-2] = self.state  # -1 position is for position bit

        if (np.random.randint(2) == 0):
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0

        self.hell_position = -self.heaven_position

        return observation

    def _get_direction(self, position):
        direction = 0.0
        if position >= self.priest_position - self.priest_delta and position <= self.priest_position + self.priest_delta:
            if (self.heaven_position > self.hell_position):
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0
        return direction

    def step(self, action):

        prev_position = self.state
        position = action[0]
        self.state = position

        positions_along_the_way = np.linspace(prev_position, position, 10)  # imagining optimal lower level policy doing the work
        directions = np.array([self._get_direction(pos) for pos in positions_along_the_way])

        observation = np.vstack([positions_along_the_way, directions]).T.flatten()

        # Convert a possible numpy bool to a Python bool.
        max_position = max(self.heaven_position, self.hell_position)
        min_position = min(self.heaven_position, self.hell_position)

        done = bool(
            position >= max_position or position <= min_position
        )

        reward = 0.0
        if (self.heaven_position > self.hell_position):
            if (position >= self.heaven_position):
                reward = 1.0

            if (position <= self.hell_position):
                reward = -1.0

        if (self.heaven_position < self.hell_position):
            if (position >= self.hell_position):
                reward = -1.0

            if (position <= self.heaven_position):
                reward = 1.0

        return observation, reward, done, {}


