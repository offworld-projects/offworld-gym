"""
This module contains environments OffWorld Gym can interact with.  
Currently we support our own real environment for interaction with
the real robot and Gazebo environment to provide simulated replica
of the real environment.
"""

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
    id='OffWorldMonolithContinuousReal-v0',
    entry_point='offworld_gym.envs.real:OffWorldMonolithContinuousEnv'
)

# simulated environments

# OffWorld Monolith Simulated replica with discrete actions
register(
    id='OffWorldMonolithDiscreteSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithDiscreteEnv'
)

# OffWorld Monolith Simulated replica with continous actions
register(
    id='OffWorldMonolithContinuousSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithContinuousEnv'
)


# OffWorld Monolith Simulated replica with discrete actions
register(
    id='OffWorldMonolithObstacleDiscreteSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithObstacleDiscreteEnv'
)

# OffWorld Monolith Simulated replica with continous actions
register(
    id='OffWorldMonolithObstacleContinuousSim-v0',
    entry_point='offworld_gym.envs.gazebo:OffWorldMonolithObstacleContinuousEnv'
)

# Dockerized Simulated Environments

# OffWorld Monolith Simulated replica with discrete actions
register(
    id='OffWorldDockerMonolithDiscreteSim-v0',
    entry_point='offworld_gym.envs.gazebo_docker:OffWorldDockerMonolithDiscreteEnv'
)

# OffWorld Monolith Simulated replica with continous actions
register(
    id='OffWorldDockerMonolithContinuousSim-v0',
    entry_point='offworld_gym.envs.gazebo_docker:OffWorldDockerMonolithContinuousEnv'
)


# OffWorld Monolith Simulated replica with discrete actions
register(
    id='OffWorldDockerMonolithObstacleDiscreteSim-v0',
    entry_point='offworld_gym.envs.gazebo_docker:OffWorldDockerMonolithObstacleDiscreteEnv'
)

# OffWorld Monolith Simulated replica with continous actions
register(
    id='OffWorldDockerMonolithObstacleContinuousSim-v0',
    entry_point='offworld_gym.envs.gazebo_docker:OffWorldDockerMonolithObstacleContinuousEnv'
)