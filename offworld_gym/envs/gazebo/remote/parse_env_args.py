import os

from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.gazebo import OffWorldMonolithContinuousEnv, OffWorldMonolithDiscreteEnv, \
    OffWorldMonolithObstacleContinuousEnv, OffWorldMonolithObstacleDiscreteEnv

USABLE_ENV_CLASSES = [OffWorldMonolithContinuousEnv, OffWorldMonolithDiscreteEnv,
                      OffWorldMonolithObstacleContinuousEnv, OffWorldMonolithObstacleDiscreteEnv]

USABLE_ENV_CLASSES_BY_NAME = {env_cls.__name__: env_cls for env_cls in USABLE_ENV_CLASSES}

USABLE_CHANNEL_TYPES = {
    elem.name: elem for elem in Channels
}

USABLE_RANDOM_INIT_VALS = {
    "TRUE": True,
    "FALSE": False,
}


def parse_env_class_from_environ():
    try:
        env_type = os.environ["OFFWORLD_ENV_TYPE"]
    except KeyError:
        raise EnvironmentError("The env variable OFFWORLD_ENV_TYPE isn't specified, and it needs to be. "
                               "It should be the name of the gym environment "
                               "that this server should provide an interface for"
                               f"\nAcceptable values are {list(USABLE_ENV_CLASSES_BY_NAME.keys())}")
    try:
        env_class = USABLE_ENV_CLASSES_BY_NAME[env_type]
    except KeyError:
        raise EnvironmentError(f"The env variable OFFWORLD_ENV_TYPE is set to an unrecognized value: {env_type}"
                               f"\nAcceptable values are {list(USABLE_ENV_CLASSES_BY_NAME.keys())}")
    return env_class


def parse_channel_type_from_environ():
    try:
        env_type = os.environ["OFFWORLD_ENV_CHANNEL_TYPE"]
    except KeyError:
        raise EnvironmentError("The env variable OFFWORLD_ENV_CHANNEL_TYPE needs to be specified to an enum value."
                               f"\nAcceptable values are {list(USABLE_CHANNEL_TYPES.keys())}")
    try:
        channel_type = USABLE_CHANNEL_TYPES[env_type]
    except KeyError:
        raise EnvironmentError(f"The env variable OFFWORLD_ENV_TYPE is set to an unrecognized value: {env_type}"
                               f"\nAcceptable values are {list(USABLE_CHANNEL_TYPES.keys())}")
    return channel_type


def parse_random_init_from_environ():
    try:
        env_type: str = os.environ["OFFWORLD_ENV_RANDOM_INIT"]
    except KeyError:
        raise EnvironmentError("The env variable OFFWORLD_ENV_RANDOM_INIT needs to be specified to "
                               "\'TRUE\' or \'FALSE\'.")
    try:
        random_init = USABLE_RANDOM_INIT_VALS[env_type.upper()]
    except KeyError:
        raise EnvironmentError(f"The env variable OFFWORLD_ENV_RANDOM_INIT is set to an unrecognized value: {env_type}"
                               f"\nAcceptable values are {list(USABLE_RANDOM_INIT_VALS.keys())}")
    return random_init