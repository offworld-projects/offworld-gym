# Copyright offworld.ai 2019

from gym.envs.registration import register

# Real environments

# OffWorld Monolith Real 
register(
    id='OffWorldMonolithRealEnv-v0',
    entry_point='offworld_gym.envs.real:OffWorldMonolithEnv'
)

# simulated environments

# OffWorld Monolith Simulated replica 
register(
    id='OffWorldMonolithSimEnv-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithEnv'
)