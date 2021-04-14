import gym
import gym_pomdps
import numpy as np
from gym import spaces

# A robot will be rewarded +1 for attaining heaven in one
# if it accidently reaches hell it will get -1
# Problem is attributed to Sebastian Thrun but first appeared in Geffner
# & Bonet: Solving Large POMDPs using Real Time DP 1998.
# A priest is available to tell it where heaven is (left or right)
#
#        Heaven  4  3  2  5  6  Hell
#                      1
#                      0
#                      7  8  9 Priest
#
#          Hell 14 13 12 15 16  Heaven
#                     11
#                     10
#                     17 18 19 Priest

"""
Location-signal version of Heaven Hell
- The observation consists of two parts: location and signal.
- There are 10 possible locations (0 to 9 inclusive) and 3 possible signals (0, 1, 2).
- These two pieces of information will be encoded as a one-hot vector such that
  the first eleven slots are for location and the last three slots are for signal.
  
Catches:
- Added attribute self.ready to make sure that the user always reset the environment before using.
- Added attribute self.terminated to make sure that the user always reset the environment per episode.
"""

class HeavenHellOneHotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.env = gym.make("POMDP-heavenhell-episodic-v0")
        self.discount = self.env.discount

        self.location_size = 10
        self.signal_size = 3
        self.memory_size = 1

        self.action_space = self.env.action_space  # for external use
        self.observation_space = spaces.Box(low=0.0, high=4.0, shape=((self.location_size + self.signal_size) * self.memory_size,),
                                            dtype=np.float32)  # for external use

        self.ready = False  # need to be reset before using at all
        self.info = {'episode' : None}  # for external use

    def set_memory(self, size:int) -> None:
        self.memory_size = size

    def push_to_memory(self, obs):
        self.memory.append(obs)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[1:]  # remove the oldest

    def close(self):
        pass

    def seed(self, seed):
        self.env.seed(seed)

    def get_state(self):
        return self.state

    def reset(self) -> np.array:
        self.state = self.env.reset_functional()
        self.ready = True
        self.memory = []
        for i in range(self.memory_size - 1):
            self.push_to_memory(np.ones((self.location_size + self.signal_size, )) * -1)
        self.push_to_memory(np.array(self._generate_obs(self.state)))
        return np.array(self.memory).flatten()

    def _generate_obs(self, state:int) -> list:
        return self._one_hot(_get_location(state), self.location_size) + \
               self._one_hot(self._get_signal(state), self.signal_size)

    def _get_location(self, state:int) -> int:
        if state >= 10:
            state -= 10
        return state

    def _get_signal(self, state:int) -> int:
        if state == 9:
            return 0  # heaven on the left
        elif state == 19:
            return 1  # heaven on the right
        else:
            return 2  # no information on the location of heaven

    def _one_hot(self, target_index:int, size:int) -> list:
        temp = [0] * size
        temp[target_index] = 1.0
        return temp

    def step(self, action:int) -> np.array:
        assert self.ready, "not ready yet / episode terminated, please reset"
        # self.state, o, r_extrinsic, done, info = self.env.step_functional(self.state, action)
        self.state, _, r_extrinsic, _, _ = self.env.step_functional(self.state, action)
        done = False
        if r_extrinsic in [-1, 1]:
            done = True
            self.ready = False
        # follows the gym template of state/obs, r, done, info
        obs = np.array(self._generate_obs(self.state))
        self.push_to_memory(obs)
        return np.array(self.memory).flatten(), r_extrinsic, done, self.info