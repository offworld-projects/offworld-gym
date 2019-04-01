# Copyright offworld.ai 2019
import gym

from real_gym import logger
from real_gym.core import controller

class RealEnv(gym.Env):
    def __init__(self):
        pass

    def step(self, action):        
        raise NotImplementedError

    def reset(self):        
        raise NotImplementedError

    def render(self, mode='human'):        
        raise NotImplementedError

    def close(self):      
        raise NotImplementedError
        
    def seed(self, seed=None):
        logger.warn("Could not seed environment %s", self)
        return