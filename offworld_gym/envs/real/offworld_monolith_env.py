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

# std

# gym
from offworld_gym.envs.real.real_env import RealEnv
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions

class OffWorldMonolithEnv(RealEnv):
    """Real Gym environment with a rosbot and a monolith on an uneven terrain

    A RL agent learns to reach the goal(monolith) in shortest time

    Usage:
        env = gym.make('OffWorldMonolithRealEnv-v0', channel_type=Channels.DEPTHONLY)
        env = gym.make('OffWorldMonolithRealEnv-v0', channel_type=Channels.RGB_ONLY)
        env = gym.make('OffWorldMonolithRealEnv-v0', channel_type=Channels.RGBD)
    """
    _PROXIMITY_THRESHOLD = 0.20

    def __init__(self, channel_type):
        assert isinstance(channel_type, Channels), "Channel type is not of Channels."
        
        