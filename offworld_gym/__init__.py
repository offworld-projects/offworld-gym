# Copyright offworld.ai 2019

from gym.envs.registration import register

# real environments
register(
    id='OffWorldMonolithRealEnv-v0',
    entry_point='offworld_gym.envs.real:OffWorldMonolithEnv',
    # More arguments here
)

# simulated environments
register(
    id='OffWorldMonolithSimEnv-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithEnv',
    # More arguments here
)