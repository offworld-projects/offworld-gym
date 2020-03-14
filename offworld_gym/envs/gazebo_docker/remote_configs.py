from offworld_gym.envs.gazebo_docker.remote_env import OffWorldDockerizedEnv, EnvVersions, with_base_config
from offworld_gym.envs.common.channels import Channels


class OffWorldDockerMonolithDiscreteEnv(OffWorldDockerizedEnv):

    def __init__(self, channel_type=Channels.DEPTH_ONLY, random_init=True):
        super(OffWorldDockerMonolithDiscreteEnv, self).__init__(
            config={"version": EnvVersions.MONOLITH_DISCRETE,
                    "channel_type": channel_type,
                    "random_init": random_init})


class OffWorldDockerMonolithContinuousEnv(OffWorldDockerizedEnv):

    def __init__(self, channel_type=Channels.DEPTH_ONLY, random_init=True):
        super(OffWorldDockerMonolithContinuousEnv, self).__init__(
            config={"version": EnvVersions.MONOLITH_CONTINUOUS,
                    "channel_type": channel_type,
                    "random_init": random_init})


class OffWorldDockerMonolithObstacleDiscreteEnv(OffWorldDockerizedEnv):

    def __init__(self, channel_type=Channels.DEPTH_ONLY, random_init=True):
        super(OffWorldDockerMonolithObstacleDiscreteEnv, self).__init__(
            config={"version": EnvVersions.OBSTACLE_DISCRETE,
                    "channel_type": channel_type,
                    "random_init": random_init})


class OffWorldDockerMonolithObstacleContinuousEnv(OffWorldDockerizedEnv):

    def __init__(self, channel_type=Channels.DEPTH_ONLY, random_init=True):
        super(OffWorldDockerMonolithObstacleContinuousEnv, self).__init__(
            config={"version": EnvVersions.OBSTACLE_CONTINUOUS,
                    "channel_type": channel_type,
                    "random_init": random_init})
