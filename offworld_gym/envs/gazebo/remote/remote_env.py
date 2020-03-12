import docker
import os
import uuid
import subprocess
import time
import grpc
import threading
from google.protobuf.empty_pb2 import Empty
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2_grpc import RemoteEnvStub
from offworld_gym.envs.gazebo.remote.protobuf.remote_env_pb2 import Action
from grpc._channel import _InactiveRpcError
from offworld_gym.envs.gazebo.remote.ndarray_proto import ndarray_to_proto, proto_to_ndarray

OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "ros-gym")
CONTAINER_GRPC_PORT = 7676
CONTAINER_GRPC_PORT_BINDING = f'{CONTAINER_GRPC_PORT}/tcp'

if __name__ == '__main__':
    client = docker.from_env()

    xhost_command = "xhost local:"
    try:
        subprocess.check_call(['bash', '-c', xhost_command])
    except subprocess.CalledProcessError:
        print(f"The bash command \"{xhost_command}\" returned an error. Exiting.")
        exit()

    container_env = {
        "DISPLAY": os.environ['DISPLAY'],
        "OFFWORLD_ENV_TYPE": "OffWorldMonolithContinuousEnv",
        "OFFWORLD_ENV_CHANNEL_TYPE": "DEPTH_ONLY",
        "OFFWORLD_ENV_RANDOM_INIT": "TRUE",
        "OFFWORLD_GYM_GRPC_SERVER_PORT": CONTAINER_GRPC_PORT,
    }

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

    container_ports = {
        CONTAINER_GRPC_PORT_BINDING: None  # will be published to a random available host port
    }

    container_name = f"offworld-gym{uuid.uuid4().hex[:10]}"

    container_entrypoint = "/offworld-gym/offworld_gym/envs/gazebo/remote/docker_entrypoint.sh"
    stub_command = "python3.6 -m http.server"
    container_instance = None
    try:
        container_instance = client.containers.run(image=OFFWORLD_GYM_DOCKER_IMAGE,
                              command=container_entrypoint,
                              name=container_name,
                              # hostname=container_name,
                              detach=True,
                              auto_remove=False,
                              environment=container_env,
                              volumes=container_volumes,
                              ports=container_ports,
                              stdout=True,
                              stderr=True)
        print(f"{container_instance.name} launched")
        container_instance.reload()  # required to get auto-assigned ports, not needed if it was an already running container
        host_published_grpc_port = container_instance.ports[CONTAINER_GRPC_PORT_BINDING][0]['HostPort']
        print(f"grpc port: {host_published_grpc_port}")

        # open a gRPC channel
        channel = grpc.insecure_channel(f'localhost:{host_published_grpc_port}')


        # create a stub (client)
        stub = RemoteEnvStub(channel)
        print("making call")
        # make the call
        connected = False
        response = None
        while not connected:
            print("sleeping")
            time.sleep(1)
            print("done sleeping")
            try:
                response = stub.Reset(Empty())
                connected = True
            except grpc.RpcError as rpc_error:
                if rpc_error.code() != grpc.StatusCode.UNAVAILABLE:
                    raise rpc_error


        print(f"observation: {proto_to_ndarray(response.observation)})")









        # block until container exits
        exit_code = container_instance.wait()
        print(f"{container_instance.name} exited {exit_code}")

    except KeyboardInterrupt:
        if container_instance is not None:
            print("Cleaning up container...")
            print(f"Removing container {container_instance.name}")
            container_instance.remove(force=True)
            print(f"Container {container_instance.name} removed")

