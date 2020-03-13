
# import the generated GRPC classes
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2 import Observation, ObservationRewardDone, Action, Image
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2_grpc import RemoteEnvServicer, add_RemoteEnvServicer_to_server

from offworld_gym.envs.gazebo.gazebo_env import GazeboGymEnv
from offworld_gym.envs.gazebo.offworld_monolith_env import OffWorldMonolithContinousEnv, OffWorldMonolithDiscreteEnv
from offworld_gym.envs.gazebo.offworld_monolith_obstacle_env import OffWorldMonolithObstacleContinousEnv, OffWorldMonolithObstacleDiscreteEnv
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.gazebo.remote.ndarray_proto import ndarray_to_proto, proto_to_ndarray

import os
import grpc
import threading
import numpy as np
from concurrent import futures
from google.protobuf.empty_pb2 import Empty

OFFWORLD_GYM_GRPC_SERVER_PORT = int(os.environ.get("OFFWORLD_GYM_GRPC_SERVER_PORT", 50051))


def parse_env_class_from_environ():
    usable_env_classes = {
        "OffWorldMonolithContinuousEnv": OffWorldMonolithContinousEnv,
        "OffWorldMonolithDiscreteEnv": OffWorldMonolithDiscreteEnv,
        "OffWorldMonolithObstacleContinuousEnv": OffWorldMonolithObstacleContinousEnv,
        "OffWorldMonolithObstacleDiscreteEnv": OffWorldMonolithObstacleDiscreteEnv
    }

    env_type = os.environ.get("OFFWORLD_ENV_TYPE", None)
    if env_type is None:
        raise EnvironmentError("The env variable OFFWORLD_ENV_TYPE isn't specified, and it needs to be. "
                               "It should be the name of the gym environment "
                               "that this server should provide an interface for"
                               f"\nAcceptable values are {list(usable_env_classes.keys())}")
    try:
        env_class = usable_env_classes[env_type]
    except KeyError:
        raise EnvironmentError(f"The env variable OFFWORLD_ENV_TYPE is set to an unrecognized value: {env_type}"
                               f"\nAcceptable values are {list(usable_env_classes.keys())}")
    return env_class


def parse_channel_type_from_environ():
    usable_channel_types = {
        "DEPTH_ONLY": Channels.DEPTH_ONLY,
        "RGB_ONLY": Channels.RGB_ONLY,
        "RGBD": Channels.RGBD,
    }

    env_type = os.environ.get("OFFWORLD_ENV_CHANNEL_TYPE", None)
    if env_type is None:
        raise EnvironmentError("The env variable OFFWORLD_ENV_CHANNEL_TYPE needs to be specified to an enum value."
                               f"\nAcceptable values are {list(usable_channel_types.keys())}")

    try:
        channel_type = usable_channel_types[env_type]
    except KeyError:
        raise EnvironmentError(f"The env variable OFFWORLD_ENV_TYPE is set to an unrecognized value: {env_type}"
                               f"\nAcceptable values are {list(usable_channel_types.keys())}")
    return channel_type


def parse_random_init_from_environ():
    usable_values = {
        "TRUE": True,
        "FALSE": False,
    }

    env_type: str = os.environ.get("OFFWORLD_ENV_RANDOM_INIT", None)
    if env_type is None:
        raise EnvironmentError("The env variable OFFWORLD_ENV_RANDOM_INIT needs to be specified to "
                               "\'TRUE\' or \'FALSE\'.")

    try:
        random_init = usable_values[env_type.upper()]
    except KeyError:
        raise EnvironmentError(f"The env variable OFFWORLD_ENV_RANDOM_INIT is set to an unrecognized value: {env_type}"
                               f"\nAcceptable values are {list(usable_values.keys())}")
    return random_init


if __name__ == '__main__':

    env_class = parse_env_class_from_environ()
    channel_type = parse_channel_type_from_environ()
    random_init = parse_random_init_from_environ()

    print("Starting env...")
    env: GazeboGymEnv = env_class(channel_type=channel_type, random_init=random_init)
    print("Env fully initialized")

    # GRPC Server
    class RemoteEnvServicerImpl(RemoteEnvServicer):

        def __init__(self, stop_event):
            self._stop_event = stop_event

        def Reset(self, request, context):
            response = Observation()
            response.observation.ndarray = ndarray_to_proto(env.reset())
            # response.observation.ndarray = ndarray_to_proto(np.ones(shape=(5,)))
            return response

        def Step(self, request: Action, context):
            action = np.squeeze(proto_to_ndarray(request.action))
            if action.shape == ():
                action = float(action)

            obs, rew, done, info = env.step(action=action)
            response = ObservationRewardDone()
            response.observation = ndarray_to_proto(obs)
            response.reward = float(rew)
            response.done = bool(done)

            return response

        def Render(self, request, context):
            image = env.render(mode='array')
            response = Image()
            response.image.ndarray = ndarray_to_proto(image)
            return response

        def Shutdown(self, request, context):
            self._stop_event.set()
            return Empty()

    # create a GRPC server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    # Stop event for server handler threads to signal this thread that it's time to shutdown
    stop_event = threading.Event()

    add_RemoteEnvServicer_to_server(RemoteEnvServicerImpl(stop_event=stop_event), grpc_server)

    print(f'Starting server. Listening on port {OFFWORLD_GYM_GRPC_SERVER_PORT}.')
    grpc_server.add_insecure_port(f'[::]:{OFFWORLD_GYM_GRPC_SERVER_PORT}')
    grpc_server.start()  # does not block

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        grpc_server.stop(0)
        env.close()
        exit(0)

    grpc_server.stop(0)
    env.close()
    exit(0)


