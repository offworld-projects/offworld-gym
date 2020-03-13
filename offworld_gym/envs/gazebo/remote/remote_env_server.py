from offworld_gym.envs.gazebo.remote.parse_env_args import parse_env_class_from_environ, \
    parse_channel_type_from_environ, parse_random_init_from_environ, parse_clip_depth_value_from_environ, \
    parse_image_out_size_from_environ
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2 import Observation, ObservationRewardDone, Action, Image, Spaces
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2_grpc import RemoteEnvServicer, add_RemoteEnvServicer_to_server

from offworld_gym.envs.gazebo.gazebo_env import GazeboGymEnv

import os
import grpc
import threading
import numpy as np
import traceback
import cloudpickle
from concurrent import futures
from google.protobuf.empty_pb2 import Empty

OFFWORLD_GYM_GRPC_SERVER_PORT = int(os.environ.get("OFFWORLD_GYM_GRPC_SERVER_PORT", 50051))

if __name__ == '__main__':

    env_class = parse_env_class_from_environ()
    channel_type = parse_channel_type_from_environ()
    random_init = parse_random_init_from_environ()
    clip_depth_value = parse_clip_depth_value_from_environ()
    image_out_size = parse_image_out_size_from_environ()

    print(f"Starting env with "
          f"class: {env_class}, "
          f" type: {channel_type}, "
          f"random_init: {random_init}, "
          f"clip_depth_value: {clip_depth_value}, "
          f"image_out_size: {image_out_size} "
          f"...\n")

    env: GazeboGymEnv = env_class(channel_type=channel_type, random_init=random_init, clip_depth_value=clip_depth_value,
                                  image_out_size=image_out_size)

    print("OffWorld Gym env initialized")

    # GRPC Server
    class RemoteEnvServicerImpl(RemoteEnvServicer):

        def __init__(self, stop_event):
            self._stop_event = stop_event

        def GetSpaces(self, request, context):
            response = Spaces()
            response.observation_space = cloudpickle.dumps(env.observation_space)
            response.action_space = cloudpickle.dumps(env.action_space)
            return response

        def Reset(self, request, context):
            response = Observation()
            response.observation = cloudpickle.dumps(np.asarray(env.reset()))
            return response

        def Step(self, request: Action, context):
            try:
                action = np.squeeze(cloudpickle.loads(request.action))
                if action.shape == ():
                    action = float(action)
                obs, rew, done, info = env.step(action=action)
                response = ObservationRewardDone()
                response.observation = cloudpickle.dumps(np.asarray(obs))
                response.reward = float(rew)
                response.done = bool(done)
                return response
            except Exception as err:
                context.set_code(grpc.StatusCode.INTERNAL)
                details = f"{err}\n{traceback.format_exc()}"
                context.set_details(details)
            return context

        def Render(self, request, context):
            image = env.render(mode='array')
            response = Image()
            response.image = cloudpickle.dumps(np.asarray(image))
            return response

        def Shutdown(self, request, context):
            self._stop_event.set()
            return Empty()

    # create a GRPC server
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))

    # Stop event for server handler threads to signal this thread that it's time to shutdown
    stop_event = threading.Event()

    add_RemoteEnvServicer_to_server(RemoteEnvServicerImpl(stop_event=stop_event), grpc_server)

    print(f'Starting GRPC server. Listening on port {OFFWORLD_GYM_GRPC_SERVER_PORT}.')
    grpc_server.add_insecure_port(f'[::]:{OFFWORLD_GYM_GRPC_SERVER_PORT}')
    grpc_server.start()  # does not block

    try:
        stop_event.wait()
    except KeyboardInterrupt:
        # TODO (JB) Something in the GazeboGymEnv hangs upon closing and doesn't return control to the shell.
        # TODO (JB) It's not a huge issue since we're generally launching this in Docker and killing the container.
        grpc_server.stop(0)
        env.close()
        exit(0)

    grpc_server.stop(0)
    env.close()
    exit(0)


