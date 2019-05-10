# Copyright offworld.ai 2019
import gym

from offworld_gym import logger
from offworld_gym.core.controller import GymController

class RealEnv(gym.Env):
    def __init__(self):
        self.request_controller = GymController()

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