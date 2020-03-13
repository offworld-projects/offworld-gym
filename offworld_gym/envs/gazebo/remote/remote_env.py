import docker
import os
import cv2
import gym
import uuid
import subprocess
import time
import grpc
import copy
import cloudpickle
import numpy as np
from enum import Enum
from google.protobuf.empty_pb2 import Empty
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2_grpc import RemoteEnvStub
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2 import Action, Observation, ObservationRewardDone, Spaces, Image

OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "offworld-gym")
CONTAINER_INTERNAL_GRPC_PORT = 7676
CONTAINER_INTERNAL_GRPC_PORT_BINDING = f'{CONTAINER_INTERNAL_GRPC_PORT}/tcp'
MAX_TOLERABLE_HANG_TIME_SECONDS = 10000000


class EnvVersions(Enum):
    MONOLITH_CONTINUOUS = "OffWorldMonolithContinuousEnv"
    MONOLITH_DISCRETE = "OffWorldMonolithDiscreteEnv"
    OBSTACLE_CONTINUOUS = "OffWorldMonolithObstacleContinuousEnv"
    OBSTACLE_DISCRETE = "OffWorldMonolithObstacleDiscreteEnv"


class Channels(Enum):
    """Channel Types of the camera
    """
    DEPTH_ONLY = 1
    RGB_ONLY = 3
    RGBD = 4


OFFWORLD_GYM_CONFIG_DEFAULTS = {
    "version": EnvVersions.MONOLITH_CONTINUOUS,
    "channel_type": Channels.RGB_ONLY,  # options are DEPTH_ONLY, RGB_ONLY, RGBD
    "random_init": True,
    "clip_depth_value": 3.0,  # positive int or None, if not None, clips to this value and normalizes to [0, 1.0]
    "image_out_size": (240, 240)  # observation images resized to this
}


def _parse_int_tuple_from_string(tuple_str):
    tuple_str = ''.join(ch for ch in tuple_str if ch.isdigit() or ch == ',')
    return tuple(map(int, tuple_str.split(',')))


def validate_env_config(env_config):
    assert isinstance(env_config['version'], EnvVersions)
    assert isinstance(env_config['channel_type'], Channels)
    assert isinstance(env_config['random_init'], bool)
    assert np.isscalar(env_config['clip_depth_value']) and env_config['clip_depth_value'] > 0
    assert np.shape(env_config['image_out_size']) == (2,) and all(elem >= 1 for elem in env_config['image_out_size'])
    try:
        _parse_int_tuple_from_string(str(env_config['image_out_size']))
    except ValueError:
        print(f"image_out_size needs to be parsable as a tuple after being converted to a string.")
        raise


def with_base_config(base_config, overrides_config):
    config = copy.deepcopy(base_config)
    config.update(overrides_config)
    return config


class OffWorldDockerizedGym(gym.Env):

    def __init__(self, config=None):
        config = with_base_config(base_config=OFFWORLD_GYM_CONFIG_DEFAULTS,
                                  overrides_config=config if config is not None else {})
        validate_env_config(env_config=config)
        self._config = config
        self._docker_client = docker.from_env()
        self._container_instance = None
        self._cv2_windows_need_destroy = False

        # Run "xhost local:" command to enable XServer to be written from localhost over network
        xhost_command = "xhost local:"
        try:
            subprocess.check_call(['bash', '-c', xhost_command])
        except subprocess.CalledProcessError:
            print(f"The bash command \"{xhost_command}\" returned an error. Exiting.")
            exit(1)

        # Set up 'docker run' command to launch gazebo env in a container

        container_env = {
            "DISPLAY": os.environ['DISPLAY'],
            "OFFWORLD_GYM_GRPC_SERVER_PORT": CONTAINER_INTERNAL_GRPC_PORT,
            "OFFWORLD_ENV_TYPE": self._config['version'].value,
            "OFFWORLD_ENV_CHANNEL_TYPE": self._config['channel_type'].name.upper(),
            "OFFWORLD_ENV_RANDOM_INIT": str(self._config['random_init']).upper(),
            "OFFWORLD_ENV_CLIP_DEPTH_VALUE": str(float(self._config['clip_depth_value'])),
            "OFFWORLD_ENV_IMAGE_OUT_SIZE": f"\"{self._config['image_out_size']}\""
        }
        container_env_str = ""
        for k, v in container_env.items():
            container_env_str += f" -e {k}={v}"

        # A dictionary to configure volumes mounted inside the container.
        # The key is either the host path or a volume name, and the value is a dictionary with the keys:
        # "bind" The path to mount the volume inside the container
        # "mode" Either rw to mount the volume read/write, or ro to mount it read-only.
        container_volumes = {
            "/tmp/.X11-unix": {
                "bind": "/tmp/.X11-unix",
                "mode": "rw"
            }
        }
        container_volumes_str = ""
        for k, v in container_volumes.items():
            container_volumes_str += f" -v {k}:{v['bind']}:{v['mode']}"

        container_ports = {
            CONTAINER_INTERNAL_GRPC_PORT_BINDING: None  # will be published to a random available host port
        }
        container_ports_str = ""
        for k, v in container_ports.items():
            if v is None:
                container_ports_str += f" -p {k}"
            else:
                container_ports_str += f" -p {k}:{v}"

        container_name = f"offworld-gym{uuid.uuid4().hex[:10]}"

        container_entrypoint = "/offworld-gym/offworld_gym/envs/gazebo/remote/docker_entrypoint.sh"
        debug_entrypoint = "python3 -m http.server"
        docker_run_command = f"docker run --name \'{container_name}\' -it -d --rm --gpus all" \
                  f"{container_env_str}{container_volumes_str}{container_ports_str} {OFFWORLD_GYM_DOCKER_IMAGE} {container_entrypoint}"
        print(f"Docker run command is:\n{docker_run_command}\n")
        container_id = subprocess.check_output(["/bin/bash", "-c", docker_run_command]).decode("utf-8").strip()
        print(f"container_id is {container_id}")
        self._container_instance = self._docker_client.containers.get(container_id=container_id)
        print(f"{self._container_instance.name} launched")
        host_published_grpc_port = self._container_instance.ports[CONTAINER_INTERNAL_GRPC_PORT_BINDING][0]['HostPort']
        print(f"Connecting on GRPC port: {host_published_grpc_port}")
        # open a gRPC channel
        channel = grpc.insecure_channel(f'localhost:{host_published_grpc_port}')

        # create a stub (client)
        self._grpc_stub = RemoteEnvStub(channel)

        # Assure that the time this method returns, the docker env is fully initialized.
        # Verify this by successfully querying the container's gym env observation and action spaces
        connected = False
        spaces_response: Spaces = None
        connection_attempt_start_time = time.time()
        while not connected:
            time.sleep(0.1)
            try:
                spaces_response: Spaces = self._grpc_stub.GetSpaces(Empty())
                connected = True
            except grpc.RpcError as rpc_error:
                if rpc_error.code() != grpc.StatusCode.UNAVAILABLE or \
                        time.time() - connection_attempt_start_time > MAX_TOLERABLE_HANG_TIME_SECONDS:
                    print("The docker instance launched but the GRPC server couldn't be connected to.")
                    raise

        self.observation_space = cloudpickle.loads(spaces_response.observation_space)
        self.action_space = cloudpickle.loads(spaces_response.action_space)
        self.reset()

    def reset(self):
        reset_response: Observation = self._grpc_stub.Reset(Empty())
        observation = np.asarray(cloudpickle.loads(reset_response.observation))
        return observation

    def step(self, action):
        request = Action()
        request.action = cloudpickle.dumps(np.asarray(action))
        step_response: ObservationRewardDone = self._grpc_stub.Step(request)
        observation = np.asarray(cloudpickle.loads(step_response.observation))
        reward = float(step_response.reward)
        done = bool(step_response.done)
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        render_response: Image = self._grpc_stub.Render(Empty())
        env_image = np.asarray(cloudpickle.loads(render_response.image))
        if mode == 'human':
            if not self._cv2_windows_need_destroy:
                self._cv2_windows_need_destroy = True

            cv2.imshow(self._config['version'].name, env_image)
            cv2.waitKey(1)
        elif mode == 'array':
            return env_image
        else:
            raise NotImplementedError(mode)

    def close(self):
        if self._cv2_windows_need_destroy:
            cv2.destroyAllWindows()

        if self._container_instance is not None:
            print("Cleaning up container...")
            print(f"Removing container {self._container_instance.name}")
            self._container_instance.remove(force=True)
            print(f"Container {self._container_instance.name} removed")
            self._container_instance = None

    def __del__(self):
        self.close()


if __name__ == '__main__':
    env = OffWorldDockerizedGym()
    print(f"action space: {env.action_space} observation_space: {env.observation_space}")
    while True:
        if env.reset().max() > 0:
            sampled_action = env.action_space.sample()
            print(sampled_action)
            env.render()
            env.step(sampled_action)
            time.sleep(0.5)

    time.sleep(10)
    env.close()
