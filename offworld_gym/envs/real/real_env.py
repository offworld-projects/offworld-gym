#!/usr/bin/env python
# Copyright offworld.ai 2018
from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

import gym

from offworld_gym import logger
from offworld_gym.envs.real.core.secured_bridge import SecuredBridge

class RealEnv(gym.Env):
    """Base class for the real environments
    """

    def __init__(self, experiment_name, resume_experiment):
        if experiment_name is None:
            raise ValueError("Please provide a value for experiment name.")
        elif resume_experiment is None:
            raise ValueError("Would you like to resume training if experiment already exists?")
        
        if not isinstance(resume_experiment, bool):
            raise ValueError("Not a valid value for resume_experiment.")

        self.experiment_name = experiment_name
        self.resume_experiment = resume_experiment
        self.secured_bridge = SecuredBridge()

    def step(self, action):        
        raise NotImplementedError

    def reset(self):        
        raise NotImplementedError

    def render(self, mode='human'):        
        raise NotImplementedError

    def close(self):      
        raise NotImplementedError
        
    def seed(self, seed=None):
        logger.warn("Can not seed a real environment")