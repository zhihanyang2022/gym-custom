# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class ContinuousMountainCarPomdpOptLowerEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_action = -5.0
        self.max_action = 5.0
        self.min_position = -1.2
        self.max_position = 1.2
        self.max_speed = 0.2
        self.heaven_position = 1.0 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.hell_position = -1.0 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.priest_position = 0.5
        self.power = 0.0015

        # When the cart is within this vicinity, it observes the direction given
        # by the priest
        self.priest_delta = 0.1

        self.low_state = np.array(
            [self.min_position, -self.max_speed, -1.0], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_speed, 1.0], dtype=np.float32
        )

        self.viewer = None

        self.action_space = spaces.Box(
            low=self.min_position,
            high=self.max_position,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_direction(self, position):
        direction = 0.0
        if position >= self.priest_position - self.priest_delta and position <= self.priest_position + self.priest_delta:
            if (self.heaven_position > self.hell_position):
                # Heaven on the right
                direction = 1.0
            else:
                # Heaven on the left
                direction = -1.0
        return direction

    def step(self, action: np.array):

        prev_position = self.state
        position = action[0]

        self.state = position

        positions_along_the_way = np.linspace(prev_position, position, 10)
        directions = np.array([self.get_direction(pos) for pos in positions_along_the_way])

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

    def reset(self):

        self.state = self.np_random.uniform(low=-0.2, high=0.2)
        observation = np.zeros((20, ))
        observation[-2] = self.state  # -1 position is for position bit

        # Randomize the heaven/hell location
        if (self.np_random.randint(2) == 0):
            self.heaven_position = 1.0
        else:
            self.heaven_position = -1.0
        
        self.hell_position = -self.heaven_position

        screen_width = 800
        screen_height = 600

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width

        if self.viewer is not None:
            self.draw_flags(scale)

        return observation

    def _height(self, xs):
        return .55 * np.ones_like(xs)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:

            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth / 4, clearance))
            )
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth / 4, clearance))
            )
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            self.draw_flags(scale)

            # Flag Priest (blue)
            flagx = (self.priest_position-self.min_position)*scale
            flagy1 = self._height(self.priest_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            )
            flag.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-self.min_position) * scale, self._height(pos) * scale
        )
        # self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def draw_flags(self, scale):
        # Flag Heaven
        from gym.envs.classic_control import rendering
        flagx = (abs(self.heaven_position)-self.min_position)*scale
        flagy1 = self._height(self.heaven_position)*scale
        flagy2 = flagy1 + 50
        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = rendering.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # RED for hell
        if self.heaven_position > self.hell_position:
            flag.set_color(0.0, 1.0, 0)
        else:
            flag.set_color(1.0, 0.0, 0)

        self.viewer.add_geom(flag)

        # Flag Hell
        flagx = (-abs(self.heaven_position)-self.min_position)*scale
        flagy1 = self._height(self.hell_position)*scale
        flagy2 = flagy1 + 50
        flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
        self.viewer.add_geom(flagpole)
        flag = rendering.FilledPolygon(
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        )

        # GREEN for heaven
        if self.heaven_position > self.hell_position:
            flag.set_color(1.0, 0.0, 0)
        else:
            flag.set_color(0.0, 1.0, 0)

        self.viewer.add_geom(flag)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
