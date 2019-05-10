# Copyright offworld.ai 2019

from gym.envs.registration import register

# real environments
register(
    id='RosbotMonolithRealEnv-v0',
    entry_point='offworld_gym.envs.real:RosbotMonolithEnv',
    # More arguments here
)

# simulated environments
register(
    id='RosbotMonolithSimEnv-v0',
    entry_point='offworld_gym.envs.simulated:RosbotMonolithEnv',
    # More arguments here
)