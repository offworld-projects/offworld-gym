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
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.real.core.request import SetUpRequest
from offworld_gym.envs.real.config import settings
from offworld_gym.envs.real import RealEnv

DEBUG = settings.config["application"]["dev"]["debug"]

class OffWorldMonolithEnv(RealEnv):
    """Generic Real Gym environment with a rosbot and a monolith on an uneven terrain.

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
        self.observation_space = spaces.Box(0, 255, shape = (1, 240, 320, channel_type.value))
        self.step_count = 0
        self.environment_name = None

        self._channel_type = channel_type
        self._last_state = None
        self._closed = False
        logger.info("Environment has been started.")

    def _initiate(self):
        """Initate communication with the real environment.
        """
        logger.info("Waiting to connect to the environment server.")
        wait_start = time.time()
        while True:
            heartbeat, registered, message = self.secured_bridge.perform_handshake(self.experiment_name, self.resume_experiment, self.learning_type, self.algorithm_mode, self.environment_name)
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
        """Must be implemented in a child class
        """
        raise NotImplementedError("Must be implemented in a child class.")

    def reset(self, action):
        """Must be implemented in a child class
        """
        raise NotImplementedError("Must be implemented in a child class.")
    
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
        raise NotImplementedError("Must be implemented in a child class.")


class OffWorldMonolithDiscreteEnv(OffWorldMonolithEnv):
    """Real Gym environment with a rosbot and a monolith on an uneven terrain with discrete action space.

    A RL agent learns to reach the goal(monolith) in shortest time.

    .. code:: python

        env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.DEPTHONLY)
        env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.SIM_2_REAL, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.RGB_ONLY)
        env = gym.make('OffWorldMonolithDiscreteReal-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.HUMAN_DEMOS, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.RGBD)

    Attributes:
        observation_space: Gym data structure that encapsulates an observation.
        action_space: Gym space box type to represent that environment has 4 discrete actions.
        step_count: An integer count of step during an episode. 
    """ 

    def __init__(self, experiment_name, resume_experiment, learning_type, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.DEPTH_ONLY):
        super(OffWorldMonolithDiscreteEnv, self).__init__(experiment_name, resume_experiment, learning_type, algorithm_mode, channel_type)
        self.action_space = spaces.Discrete(4)
        self.environment_name = 'OffWorldMonolithDiscreteReal-v0'
        self._initiate()        

    def step(self, action):
        """Take a discrete action in the environment.

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

        # convert float if it's exactly an integer value, otherwise let it throw an error
        if isinstance(action, (float, np.float32, np.float64)) and float(action).is_integer():
            action = int(action)

        assert isinstance(action, (FourDiscreteMotionActions, int, np.int32, np.int64)), "Action type is not recognized."

        if self._closed:
            raise GymException("The environment has been closed.")

        if isinstance(action, (int, np.int32, np.int64)):
            assert action >= 0 and action < 4, "Unrecognized value for the action"
            action = FourDiscreteMotionActions(action)
        
        state, reward, done = self.secured_bridge.monolith_discrete_perform_action(action, self._channel_type, self.algorithm_mode)

        state = state[0] # removing the extra first dim
        
        self._last_state = state

        logger.info('Environment step is complete.')
        if done:
            logger.info('Environment episode is complete: {} steps, reward = {}'.format(self.step_count, reward))
            self.step_count = 0

        return state, reward, done, {}

    def reset(self):
        """Resets the state of the environment and returns an observation.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
        """

        if self._closed:
            raise GymException("The environment has been closed.")

        logger.info("Resetting the episode and moving to a random initial position...")
        state = self.secured_bridge.monolith_discrete_perform_reset(self._channel_type)

        state = state[0] # removing the extra first dim

        logger.info("Environment reset complete")
        return state

    def close(self):
        """Closes the environment.
        """
        self._closed = True
        self.secured_bridge.disconnect(self._channel_type, True)


class OffWorldMonolithContinuousEnv(OffWorldMonolithEnv):
    """Real Gym environment with a rosbot and a monolith on an uneven terrain with continous action space.

    A RL agent learns to reach the goal(monolith) in shortest time.

    .. code:: python

        env = gym.make('OffWorldMonolithContinousReal-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.END_TO_END, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.DEPTHONLY)
        env = gym.make('OffWorldMonolithContinousReal-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.SIM_2_REAL, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.RGB_ONLY)
        env = gym.make('OffWorldMonolithContinousReal-v0', experiment_name='first_experiment', resume_experiment=False, learning_type=LearningType.HUMAN_DEMOS, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.RGBD)

    Attributes:
        observation_space: Gym data structure that encapsulates an observation.
        action_space: Gym space box type to represent that environment has continouss actions.
        step_count: An integer count of step during an episode. 
    """ 
    
    def __init__(self, experiment_name, resume_experiment, learning_type, algorithm_mode=AlgorithmMode.TRAIN, channel_type=Channels.DEPTH_ONLY):
        super(OffWorldMonolithContinuousEnv, self).__init__(experiment_name, resume_experiment, learning_type, algorithm_mode, channel_type)
        self.action_space = spaces.Box(low=np.array([-0.7, -2.5]), high=np.array([0.7, 2.5]), dtype=np.float32)
        self.action_limit = np.array([[-0.5, -2.5], [0.5, 2.5]])
        self.environment_name = 'OffWorldMonolithContinousReal-v0'
        self._initiate()        

    def step(self, action):
        """Take a continous action in the environment.

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
        assert isinstance(action, (np.ndarray)), "Action type is not recognized."
        action = np.clip(action, self.action_limit[0], self.action_limit[1])

        if self._closed:
            raise GymException("The environment has been closed.")
        
        state, reward, done = self.secured_bridge.monolith_continous_perform_action(action, self._channel_type, self.algorithm_mode)

        state = state[0] # removing the extra first dim
        
        self._last_state = state

        logger.info('Environment step is complete.')
        if done:
            logger.info('Environment episode is complete: {} steps, reward = {}'.format(self.step_count, reward))
            self.step_count = 0

        return state, reward, done, {}

    def reset(self):
        """Resets the state of the environment and returns an observation.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
        """

        if self._closed:
            raise GymException("The environment has been closed.")

        logger.info("Resetting the episode and moving to a random initial position...")
        state = self.secured_bridge.monolith_continous_perform_reset(self._channel_type)

        state = state[0] # removing the extra first dim
        
        logger.info("Environment reset complete")
        return state

    def close(self):
        """Closes the environment.
        """
        self._closed = True
        self.secured_bridge.disconnect(self._channel_type, False)