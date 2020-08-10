import os
import time
import threading
import traceback
from concurrent import futures

import cloudpickle
import grpc
import numpy as np
from google.protobuf.empty_pb2 import Empty

from offworld_gym.envs.gazebo.gazebo_env import GazeboGymEnv
from offworld_gym.envs.gazebo_docker.parse_env_args import parse_env_class_from_environ, \
    parse_channel_type_from_environ, parse_random_init_from_environ
from offworld_gym.envs.gazebo_docker.protobuf.remote_env_pb2 import Observation, ObservationRewardDoneInfo, Action, Image, \
    Spaces
from offworld_gym.envs.gazebo_docker.protobuf.remote_env_pb2_grpc import RemoteEnvServicer, \
    add_RemoteEnvServicer_to_server

OFFWORLD_GYM_GRPC_SERVER_PORT = int(os.environ.get("OFFWORLD_GYM_GRPC_SERVER_PORT", 50051))
MAX_TOLERABLE_GAZEBO_START_UP_HANG_TIME_SECONDS = 20
HEART_BEAT_WAIT_TOLERANCE_SECONDS = 90


if __name__ == '__main__':

    env_class = parse_env_class_from_environ()
    channel_type = parse_channel_type_from_environ()
    random_init = parse_random_init_from_environ()

    print(f"Starting env with "
          f"class: {env_class}, "
          f" type: {channel_type}, "
          f"random_init: {random_init}, "
          f"...\n")

    env: GazeboGymEnv = env_class(channel_type=channel_type, random_init=random_init)

    # There seems to be a race condition with the Gazebo simulator
    # where rendering isn't functioning by the time the env init function returns.
    # Wait until we get an image or timeout.
    simulation_wait_state_time = time.time()
    while env.reset().max() == 0:
        if time.time() - simulation_wait_state_time > MAX_TOLERABLE_GAZEBO_START_UP_HANG_TIME_SECONDS:
            raise TimeoutError("Simulation seems to have never started, "
                               "observations returned were always blank (max pixel value of 0).")
        time.sleep(0.1)

    print("OffWorld Gym env initialized")

    # GRPC Server
    class RemoteEnvServicerImpl(RemoteEnvServicer):

        def __init__(self, stop_event):
            self._stop_event = stop_event
            self.time_of_last_heart_beat = time.time()

            def _heart_beat_check_worker():
                # exits if a heartbeat from the launching env stops being received
                global HEART_BEAT_WAIT_TOLERANCE_SECONDS
                while True:
                    time.sleep(HEART_BEAT_WAIT_TOLERANCE_SECONDS)
                    HEART_BEAT_WAIT_TOLERANCE_SECONDS = 10  # TODO: Figure out why on some machine the very first heartbeat takes 75 seconds to arrive
                    time_since_last_heat_beat = time.time() - self.time_of_last_heart_beat

                    if time_since_last_heat_beat > HEART_BEAT_WAIT_TOLERANCE_SECONDS:
                        print(f"Time since last heartbeat ({time_since_last_heat_beat}) seconds has surpassed the max"
                              f" heart beat wait time ({HEART_BEAT_WAIT_TOLERANCE_SECONDS}) seconds. Killing server...")
                        os._exit(1)

            heart_beat_thread = threading.Thread(target=_heart_beat_check_worker)
            heart_beat_thread.daemon = True
            heart_beat_thread.start()

        def HeartBeat(self, request, context):
            self._has_ever_received_heartbeat = True
            self.time_of_last_heart_beat = time.time()
            return Empty()

        def GetSpaces(self, request, context):
            try:
                response = Spaces()
                response.observation_space = cloudpickle.dumps(env.observation_space)
                response.action_space = cloudpickle.dumps(env.action_space)
                return response
            except Exception as err:
                context.set_code(grpc.StatusCode.INTERNAL)
                details = f"{err}\n{traceback.format_exc()}"
                context.set_details(details)
            return context

        def SetSeed(self, request, context):
            try:
                env.seed(request.seed)
                return Empty()
            except Exception as err:
                context.set_code(grpc.StatusCode.INTERNAL)
                details = f"{err}\n{traceback.format_exc()}"
                context.set_details(details)
            return context

        def Reset(self, request, context):
            try:
                response = Observation()
                response.observation = cloudpickle.dumps(np.asarray(env.reset()))
                return response
            except Exception as err:
                context.set_code(grpc.StatusCode.INTERNAL)
                details = f"{err}\n{traceback.format_exc()}"
                context.set_details(details)
            return context

        def Step(self, request: Action, context):
            try:
                action = np.squeeze(cloudpickle.loads(request.action))
                if action.shape == ():
                    action = float(action)
                obs, rew, done, info = env.step(action=action)
                response = ObservationRewardDoneInfo()
                response.observation = cloudpickle.dumps(np.asarray(obs))
                response.reward = float(rew)
                response.done = bool(done)
                response.info = cloudpickle.dumps(info)
                return response
            except Exception as err:
                context.set_code(grpc.StatusCode.INTERNAL)
                details = f"{err}\n{traceback.format_exc()}"
                context.set_details(details)
            return context

        def Render(self, request, context):
            try:
                image = env.render(mode='array')
                response = Image()
                response.image = cloudpickle.dumps(np.asarray(image))
                return response
            except Exception as err:
                context.set_code(grpc.StatusCode.INTERNAL)
                details = f"{err}\n{traceback.format_exc()}"
                context.set_details(details)
            return context

        def Shutdown(self, request, context):
            self._stop_event.set()
            return Empty()


    # create a GRPC server, one thread worker for env commands, one for heartbeat
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))

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
