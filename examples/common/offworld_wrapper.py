#!/usr/bin/env python

# Copyright 2019 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All rights reserved.
#
# Licensed under GNU General Public License v3.0 (the "License")
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law, any source code or other materials
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, express or implied.

from offworld_gym import version

__version__     = version.__version__


import time
import os
import re

import matplotlib.cm
import cv2
import numpy as np
from collections import deque
from math import sqrt

import gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType

def get_log_path(path=None, developerTestingFlag=True):
    """Setup a path where log files will be stored

    Path format .\[path]\YY-mm-dd\HH-MM-SS\
    """
    if path is None:
        path = os.path.abspath('rr_log')
    else:
        path = path

    logDir = path
    if developerTestingFlag:
        directory = path
    else:
        directory =  os.path.join(path, time.strftime("%Y-%m-%d", time.localtime()), time.strftime("%H-%M-%S", time.localtime()))
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Log dir: {}".format(directory))
    return directory, logDir


# Referrence: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work.

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        self.size = 100
        self.channel = env.observation_space.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=(self.size, self.size),  # default shape=(240, 320),
            dtype=env.observation_space.dtype
        )

    def observation(self, frame):
        """returns the current observation from a frame"""
        if self.channel == 1: #depth only
            depth = frame[0, :, :, -1]
            depth = (np.clip(depth / 5, 0, 1) * 255).astype('uint8')  # convert depth to uint8
            obs = depth
        elif self.channel == 3: #rgb only:
            rgb = cv2.cvtColor(frame[0, :, :, :3].astype('float32'), cv2.COLOR_RGB2GRAY).astype('uint8')
            obs = rgb
        else: # rgbd
            depth = frame[0, :, :, -1]
            depth = (np.clip(depth / 5, 0, 1) * 255).astype('uint8')  # convert depth to uint8
            rgb = cv2.cvtColor(frame[0, :, :, :3].astype('float32'), cv2.COLOR_RGB2GRAY).astype('uint8')
            obs = np.stack([rgb, depth], axis=-1) # 2-channel image 

        obs_resized = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        obs_resized = obs_resized[..., np.newaxis] if len(obs_resized.shape) == 2 else obs_resized
        return np.transpose(obs_resized, axes=[2, 0, 1])

class FrameStack(gym.Wrapper):
    """Stack n_frames last frames.

    :param gym.Env env: the environment to wrap.
    :param int n_frames: the number of frames to stack.
    """

    def __init__(self, env, n_frames):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        shape = (env.observation_space.shape[0] * n_frames, *env.observation_space.shape[1:])
        self.observation_space = gym.spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape,
            dtype=env.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        # the original wrapper use `LazyFrames` but since we use np buffer,
        # it has no effect
        return np.concatenate(self.frames, axis=0)

def wrap_offworld(
    env,
    frame_stack=1,
    warp_frame=True,
):
    """Configure environment for OffWorld-Gym. The observation is
    channel-first: (c, h, w) instead of (h, w, c).

    :param str env: the offworld-gym environment.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped OffWorld-Gym environment.
    """
    if warp_frame:
        env = WarpFrame(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env

