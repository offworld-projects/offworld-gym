# Copyright offworld.ai 2019

from gym.envs.registration import register

# Real environments

# OffWorld Monolith Real with Discrete actions
register(
    id='OffWorldMonolithDiscreteReal-v0',
    entry_point='offworld_gym.envs.real:OffWorldMonolithDiscreteEnv'
)

# OffWorld Monolith Real with Discrete actions
register(
    id='OffWorldMonolithContinousReal-v0',
    entry_point='offworld_gym.envs.real:OffWorldMonolithContinousEnv'
)

# simulated environments

# OffWorld Monolith Simulated replica with discrete actions
register(
    id='OffWorldMonolithDiscreteSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithDiscreteEnv'
)

# OffWorld Monolith Simulated replica with continous actions
register(
    id='OffWorldMonolithContinousSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithContinousEnv'
)


# OffWorld Monolith Simulated replica with discrete actions
register(
    id='OffWorldMonolithObstacleDiscreteSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithObstacleDiscreteEnv'
)

# OffWorld Monolith Simulated replica with continous actions
register(
    id='OffWorldMonolithObstacleContinousSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithObstacleContinousEnv'
)