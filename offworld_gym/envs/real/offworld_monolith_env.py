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

#lib
from math import pi
import time
import numpy as np
from scipy.spatial import distance
from random import randint
from matplotlib import pyplot as plt

#gym
import gym
from gym import utils, spaces
from offworld_gym import logger
from offworld_gym.envs.real.real_env import RealEnv, AlgorithmMode, LearningType
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.common.offworld_gym_utils import ImageUtils
from offworld_gym.envs.real.core.request import SetUpRequest
from offworld_gym.envs.real.config import settings

DEBUG = settings.config["application"]["dev"]["debug"]

class OffWorldMonolithEnv(RealEnv):
    """Real Gym environment with a rosbot and a monolith on an uneven terrain.

    A RL agent learns to reach the goal(monolith) in shortest time.

    .. code:: python

        env = gym.make('OffWorldMonolithRealEnv-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.DEPTHONLY)
        env = gym.make('OffWorldMonolithRealEnv-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.SIM_2_REAL, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.RGB_ONLY)
        env = gym.make('OffWorldMonolithRealEnv-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.HUMAN_DEMOS, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.RGBD)

    Attributes:
        observation_space: Gym data structure that encapsulates an observation.
        action_space: Gym data structure that encapsulates an action.
        step_count: An integer count of step during an episode. 
    """
    
    def __init__(self, experiment_name, resume_experiment, learning_type, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.DEPTH_ONLY):
        super(OffWorldMonolithEnv, self).__init__(experiment_name, resume_experiment, learning_type, algorithm_mode)
        

        assert isinstance(channel_type, Channels), "Channel type is not of type Channels."
        logger.info("Environment has been initiated.")
        
        #environment
        self._channel_type = channel_type
        self.observation_space = spaces.Box(0, 255, shape = (1, ImageUtils.IMG_H, ImageUtils.IMG_W, channel_type.value))
        self.action_space = spaces.Discrete(4)
        self._initiate()
        self.step_count = 0
        self._last_state = None

        logger.info("Environment has been started.")

    def _initiate(self):
        """Initate communication with the real environment.
        """
        logger.info("Waiting to connect to the environment server.")
        wait_start = time.time()
        while True:
            heartbeat, registered, message = self.secured_bridge.perform_handshake(self.experiment_name, self.resume_experiment, self.learning_type, self.algorithm_mode)
            if heartbeat is None:
                continue
            elif heartbeat == SetUpRequest.STATUS_RUNNING and registered:
                logger.info(message)
                break
            elif heartbeat == SetUpRequest.STATUS_RUNNING and not registered:
                raise GymException(message)
            else:
                raise GymException("Gym server is not ready.")
            if time.time() - wait_start > 60:
                raise GymException("Connect to the environment server timed out.")
        
        logger.info("The environment server is running.")

    def step(self, action):
        """Take an action in the environment.

        Args:
            action: An action to be taken in the environment.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
            An integer with reward from the environment.
            A boolean flag which is true when an episode is complete.
            No info given for fair learning.
        """
        self.step_count += 1
        logger.info("Step count: {}".format(str(self.step_count)))
        
        assert action is not None, "Action cannot be None."
        assert isinstance(action, (FourDiscreteMotionActions, int, np.int32, np.int64)), "Action type is not recognized."

        if isinstance(action, (int, np.int32, np.int64)):
            assert action >= 0 and action < 4, "Unrecognized value for the action"
            action = FourDiscreteMotionActions(action)
        
        state, reward, done = self.secured_bridge.perform_action(action, self._channel_type)
        
        self._last_state = state

        if done:
            logger.debug('Environment episode is complete.')
            self.step_count = 0
        logger.info('Environment step is complete.')
        return state, reward, done, {}

    def reset(self):
        """Resets the state of the environment and returns an observation.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
        """
        state = self.secured_bridge.perform_reset(self._channel_type)
        logger.info("Environment reset complete")
        return state
    
    def render(self, mode='human'):
        """Renders the environment.
        
        Args:
            mode: The type of rendering to use.
                - 'human': Renders state to a graphical window.
        
        Returns:
            None as only human mode is implemented.
        """        
        if mode == 'human':
            self.plot(self._last_state)
        else:
            pass
        return None

    def plot(self, img, id=1, title="State"):
        """Plot an image in a non-blocking way.


        Args:
            img: A numpy array containing the observation.
            id: Numeric id which is assigned to the pyplot figure.
            title: String value which is used as the title.
        """
        if img is not None and isinstance(img, np.ndarray):
            plt.figure(id)     
            plt.ion()
            plt.clf() 
            plt.imshow(img.squeeze())         
            plt.title(title)  
            plt.show(block=False)
            plt.pause(0.05)

    def close(self):
        """Closes the environment.
        """
        self.secured_bridge.disconnect()
