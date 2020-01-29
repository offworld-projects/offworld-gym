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

from abc import abstractmethod
from abc import ABCMeta
from enum import Enum

import gym
from offworld_gym import logger
from offworld_gym.envs.real.core.secured_bridge import SecuredBridge
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType

class RealEnv(gym.Env, metaclass=ABCMeta):
    """Base class for the real environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, experiment_name, resume_experiment, learning_type, algorithm_mode=AlgorithmMode.TRAIN):
        if experiment_name is None:
            raise ValueError("Please provide a value for experiment name.")
        
        if resume_experiment is None:
            raise ValueError("Would you like to resume training if experiment already exists?")

        if learning_type is None:
            raise ValueError("Learning_type cannot be none")

        assert isinstance(learning_type, LearningType), "Learning_type is not of type LearningType."
        assert isinstance(algorithm_mode, AlgorithmMode), "Algorithm_mode is not of type AlgorithmMode."
        
        if not isinstance(resume_experiment, bool):
            raise ValueError("Not a valid value for resume_experiment.")

        self.experiment_name = experiment_name
        self.resume_experiment = resume_experiment
        self.secured_bridge = SecuredBridge()
        self.learning_type = learning_type
        self.algorithm_mode = algorithm_mode

    @abstractmethod
    def step(self, action):        
        """Abstract step method to be implemented in a child class.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):   
        """Abstract reset method to be implemented in a child class.
        """     
        raise NotImplementedError
    
    @abstractmethod
    def render(self, mode='human'):  
        """Abstract render method to be implemented in a child class.
        """      
        raise NotImplementedError
    
    @abstractmethod
    def close(self):      
        """Abstract close method to be implemented in a child class.
        """
        raise NotImplementedError
        
    def seed(self, seed=None):
        """No implementation for a real environment.
        """
        logger.warn("Can not seed a real environment")