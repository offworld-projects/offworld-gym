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
    def __init__(self):
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
        return