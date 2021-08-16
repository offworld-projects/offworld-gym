import copy
import logging
import os
import subprocess
import time
import uuid
from enum import Enum

import cloudpickle
import cv2
import docker
import grpc
import gym
import atexit
import threading
import weakref
import socket
import numpy as np
from termcolor import colored
from gym.utils import seeding
from google.protobuf.empty_pb2 import Empty
from offworld_gym.envs.common.channels import Channels

from offworld_gym.envs.gazebo_docker.protobuf.remote_env_pb2 import Action, Observation, ObservationRewardDoneInfo, \
    Spaces, Image, Seed
from offworld_gym.envs.gazebo_docker.protobuf.remote_env_pb2_grpc import RemoteEnvStub

logger = logging.getLogger(__name__)
logger.setLevel(level)
level = logging.DEBUG

OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "offworldai/offworld-gym")

CONTAINER_INTERNAL_GRPC_PORT = 7676
CONTAINER_INTERNAL_GRPC_PORT_BINDING = f'{CONTAINER_INTERNAL_GRPC_PORT}/tcp'

GAZEBO_SERVER_INTERNAL_PORT = 11345
CONTAINER_INTERNAL_GAZEBO_PORT_BINDING = f'{GAZEBO_SERVER_INTERNAL_PORT}/tcp'

GAZEBO_WEB_SERVER_INTERNAL_PORT = 8080
CONTAINER_INTERNAL_GAZEBO_WEB_PORT_BINDING = f'{GAZEBO_WEB_SERVER_INTERNAL_PORT}/tcp'

MAX_TOLERABLE_HANG_TIME_SECONDS = 90
HEART_BEAT_TO_CONTAINER_INTERVAL_SECONDS = 2


class EnvVersions(Enum):
    MONOLITH_CONTINUOUS = "OffWorldMonolithContinuousEnv"
    MONOLITH_DISCRETE = "OffWorldMonolithDiscreteEnv"
    OBSTACLE_CONTINUOUS = "OffWorldMonolithObstacleContinuousEnv"
    OBSTACLE_DISCRETE = "OffWorldMonolithObstacleDiscreteEnv"


OFFWORLD_GYM_CONFIG_DEFAULTS = {
    "version": EnvVersions.MONOLITH_CONTINUOUS,
    "channel_type": Channels.RGB_ONLY,  # options are DEPTH_ONLY, RGB_ONLY, RGBD
    "random_init": True,
}


def validate_env_config(env_config):
    assert isinstance(env_config['version'], EnvVersions)
    assert isinstance(env_config['channel_type'], Channels)
    assert isinstance(env_config['random_init'], bool)


def with_base_config(base_config, overrides_config):
    config = copy.deepcopy(base_config)
    if overrides_config is not None:
        config.update(overrides_config)
    return config


def _heart_beat_to_container_worker(grpc_port, weak_ref_to_parent_env):
    channel = grpc.insecure_channel(f'localhost:{grpc_port}')
    grpc_stub = RemoteEnvStub(channel)
    ever_made_successful_hearbeat = False
    hb_loop_start_time = time.time()
    attempts = 0 
    while attempts < 3:
        logger.debug(f"\n Attempts to call client heartbeat service {attempts} times.")
        if weak_ref_to_parent_env() is None:
            # exit if parent env is garbage collected or otherwise deleted
            logger.debug("heartbeat thread exiting after parent env was destroyed.")
            return
        try:
            before = time.time() 
            grpc_stub.HeartBeat(Empty(), timeout=1.0) # if interupted, change timeout from 1.0 to 10.0
            logger.debug(f"heartbeat interval : {time.time() - before}")
            ever_made_successful_hearbeat = True
            if ever_made_successful_hearbeat: attempts = 0
        except grpc.RpcError as rpc_error:
            attempts += 1
            time_in_hb_loop = time.time() - hb_loop_start_time
            if ever_made_successful_hearbeat and attempts > 2:
                logger.debug(f"heartbeat thread exiting after catching grpc error:\n{rpc_error}")
                # print client side message here, in case next clause does not meet
                print(f"No heartbeat from the client in {time.time() - before} seconds, killing the server.")
                return
            elif time_in_hb_loop > MAX_TOLERABLE_HANG_TIME_SECONDS and attempts > 2:
                logger.debug(f"heartbeat thread exiting after taking too long ({time_in_hb_loop} seconds) without successfully sending a single first hearbeat. Latest heartbeat grpc error: {rpc_error}")
                return
        time.sleep(HEART_BEAT_TO_CONTAINER_INTERVAL_SECONDS)


class OffWorldDockerizedEnv(gym.Env):

    def __init__(self, config=None):
        config = with_base_config(base_config=OFFWORLD_GYM_CONFIG_DEFAULTS,
                                  overrides_config=config)
        validate_env_config(env_config=config)
        self._config = config
        self._docker_client = docker.from_env()
        self._container_instance = None
        self._cv2_windows_need_destroy_on_close = False

        # Run "xhost local:" command to enable XServer to be written from localhost over network
        xhost_command = "xhost local:"
        try:
            subprocess.check_output(['bash', '-c', xhost_command])
        except subprocess.CalledProcessError:
            logger.warning(f"The bash command \"{xhost_command}\" failed. "
                        f"Installing \'xhost\' may be required for OffWorldDockerizedEnv to render properly. "
                        f"Further issues may be caused by this.")

        # Set up 'docker run' command to launch gazebo env in a container
        self._launch_docker_instance()

    def _launch_docker_instance(self):

        container_env = {
            # "DISPLAY": os.environ['DISPLAY'],
            "OFFWORLD_GYM_GRPC_SERVER_PORT": CONTAINER_INTERNAL_GRPC_PORT,
            "OFFWORLD_ENV_TYPE": self._config['version'].value,
            "OFFWORLD_ENV_CHANNEL_TYPE": self._config['channel_type'].name.upper(),
            "OFFWORLD_ENV_RANDOM_INIT": str(self._config['random_init']).upper(),
        }
        container_env_str = ""
        for k, v in container_env.items():
            container_env_str += f" -e {k}={v}"

        # A dictionary to configure volumes mounted inside the container.
        # The key is either the host path or a volume name, and the value is a dictionary with the keys:
        # "bind" The path to mount the volume inside the container
        # "mode" Either rw to mount the volume read/write, or ro to mount it read-only.
        container_volumes = {
            # "/tmp/.X11-unix": {
            #     "bind": "/tmp/.X11-unix",
            #     "mode": "rw"
            # }
        }
        container_volumes_str = ""
        for k, v in container_volumes.items():
            container_volumes_str += f" -v {k}:{v['bind']}:{v['mode']}"

        container_ports = {
            CONTAINER_INTERNAL_GRPC_PORT_BINDING: None,  # will be published to a random available host port
            CONTAINER_INTERNAL_GAZEBO_PORT_BINDING: None,
            CONTAINER_INTERNAL_GAZEBO_WEB_PORT_BINDING: None,
        }
        container_ports_str = ""
        for k, v in container_ports.items():
            if v is None:
                container_ports_str += f" -p {k}"
            else:
                container_ports_str += f" -p {k}:{v}"

        container_name = f"offworld-gym{uuid.uuid4().hex[:10]}"

        container_entrypoint = "/offworld-gym/offworld_gym/envs/gazebo_docker/docker_entrypoint.sh"
        docker_run_command = f"docker run --name \'{container_name}\' -it -d --rm" \
                             f"{container_env_str}{container_volumes_str}{container_ports_str} " \
                             f"{OFFWORLD_GYM_DOCKER_IMAGE} {container_entrypoint}"
        logger.debug(f"Docker run command is:\n{docker_run_command}\n")
        container_id = subprocess.check_output(["/bin/bash", "-c", docker_run_command]).decode("utf-8").strip()

        # ensure cleanup at exit
        def kill_container_if_it_still_exists():
            try:
                # bash command kills the container if it exists, otherwise return error code 1 without printing an error
                kill_command = f"docker ps -q --filter \"id={container_id}\" | grep -q . && docker kill {container_id}"
                removed_container = subprocess.check_output(['bash', '-c', kill_command]).decode("utf-8").strip()
                print(f"Cleaned up container {removed_container}")
            except subprocess.CalledProcessError:
                pass

        atexit.register(kill_container_if_it_still_exists)

        logger.info(f"container_id is {container_id}")
        self._container_instance = self._docker_client.containers.get(container_id=container_id)
        logger.debug(f"{self._container_instance.name} launched")
        host_published_grpc_port = self._container_instance.ports[CONTAINER_INTERNAL_GRPC_PORT_BINDING][0]['HostPort']
        host_published_gazebo_port = self._container_instance.ports[CONTAINER_INTERNAL_GAZEBO_PORT_BINDING][0]['HostPort']
        host_published_gazebo_web_port = self._container_instance.ports[CONTAINER_INTERNAL_GAZEBO_WEB_PORT_BINDING][0]['HostPort']
        logger.debug(f"host gazebo port is {host_published_gazebo_port}")
        logger.info(colored(f"For visualization of simulation, visit gzweb server at http://{socket.gethostbyname(socket.gethostname())}:{host_published_gazebo_web_port}", "green"))
        logger.debug(f"Connecting on GRPC port: {host_published_grpc_port}")

        # open a gRPC channel
        channel = grpc.insecure_channel(f'localhost:{host_published_grpc_port}')
        self._grpc_stub = RemoteEnvStub(channel)

        # Assure that the time this method returns, the docker env is fully initialized.
        # Verify this by successfully querying the container's gym env observation and action spaces
        connected = False
        spaces_response: Spaces = None
        connection_attempt_start_time = time.time()
        while not connected:
            time.sleep(0.1)
            try:
                spaces_response: Spaces = self._grpc_stub.GetSpaces(Empty(), timeout=MAX_TOLERABLE_HANG_TIME_SECONDS)
                connected = True
            except grpc.RpcError as rpc_error:
                if rpc_error.code() != grpc.StatusCode.UNAVAILABLE or \
                        time.time() - connection_attempt_start_time > MAX_TOLERABLE_HANG_TIME_SECONDS:
                    print("The docker instance launched but the GRPC server couldn't be connected to.")
                    raise

        self.observation_space = cloudpickle.loads(spaces_response.observation_space)
        self.action_space = cloudpickle.loads(spaces_response.action_space)
        
        #_np_random is None when it is dumped using cloudpickle in the container. Cloudpickle ignores the None variables(maybe) when dumping.
        # after loading from the dump, here we define the variable _np_random again.
        self.observation_space._np_random = None
        self.action_space._np_random = None

        self._heart_beat_thread = threading.Thread(target=_heart_beat_to_container_worker,
                                             args=(host_published_grpc_port, weakref.ref(self)))
        self._heart_beat_thread.daemon = True
        self._heart_beat_thread.start()

        self.reset()

    def _clean_up_docker_instance(self):
        if self._container_instance is not None:
            logger.debug(f"Removing container {self._container_instance.name}")
            self._container_instance.remove(force=True)
            logger.debug(f"Container {self._container_instance.name} removed")
            self._container_instance = None

    def reset(self):

        reset_response: Observation = self._grpc_stub.Reset(Empty(), timeout=MAX_TOLERABLE_HANG_TIME_SECONDS)
        observation = np.asarray(cloudpickle.loads(reset_response.observation))
        return observation

    def step(self, action):
        request = Action()
        request.action = cloudpickle.dumps(np.asarray(action))
        step_response: ObservationRewardDoneInfo = self._grpc_stub.Step(request, timeout=MAX_TOLERABLE_HANG_TIME_SECONDS)
        observation = np.asarray(cloudpickle.loads(step_response.observation))
        reward = float(step_response.reward)
        done = bool(step_response.done)
        info = cloudpickle.loads(step_response.info)
        return observation, reward, done, info

    def render(self, mode='human'):
        render_response: Image = self._grpc_stub.Render(Empty(), timeout=MAX_TOLERABLE_HANG_TIME_SECONDS)
        env_image = np.asarray(cloudpickle.loads(render_response.image))[0]
        if mode == 'human':
            if not self._cv2_windows_need_destroy_on_close:
                self._cv2_windows_need_destroy_on_close = True

            images_to_show = []
            if self._config['channel_type'] == Channels.RGB_ONLY:
                images_to_show.append(('rgb', env_image))
            elif self._config['channel_type'] == Channels.DEPTH_ONLY:
                images_to_show.append(('depth', env_image))
            elif self._config['channel_type'] == Channels.RGBD:
                rgb_image = env_image[:, :, :3]/255
                depth_image = env_image[:, :, 3]
                images_to_show.append(('rgb', rgb_image))
                images_to_show.append(('depth', depth_image))
            else:
                raise NotImplementedError(f"Unknown Channel type for rendering: {self._config['channel_type']}")

            for image_name, image in images_to_show:
                cv2.imshow(f"{self._container_instance.name} {image_name}", image)
            cv2.waitKey(1)
        elif mode == 'array':
            return env_image
        else:
            raise NotImplementedError(mode)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        seed_request = Seed()
        seed_request.seed = seed
        self._grpc_stub.SetSeed(seed_request)
        return [seed]

    def close(self):
        self._clean_up_docker_instance()
        if self._cv2_windows_need_destroy_on_close:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    env = gym.make("OffWorldDockerMonolithDiscreteSim-v0", channel_type=Channels.RGBD)
    logger.info(f"action space: {env.action_space} observation_space: {env.observation_space}")
    while True:
        obs = env.reset()
        done = False
        while not done:
            sampled_action = env.action_space.sample()
            #env.render()
            print(sampled_action)
            obs, rew, done, info = env.step(sampled_action)
