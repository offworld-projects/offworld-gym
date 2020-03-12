import docker
import os
import uuid
import subprocess
# from offworld_gym import logger


OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "ros-gym")
CONTAINER_GRPC_PORT_BINDING = '7676/tcp'

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
        "OFFWORLD_GYM_GRPC_SERVER_PORT": CONTAINER_GRPC_PORT_BINDING,
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

    container_instance = None
    try:
        container_instance = client.containers.run(image=OFFWORLD_GYM_DOCKER_IMAGE,
                              entrypoint=container_entrypoint,
                              name=container_name,
                              hostname=container_name,
                              detach=True,
                              auto_remove=False,
                              environment=container_env,
                              volumes=container_volumes,
                              ports=container_ports)
        print(f"{container_instance.name} launched")
        container_instance.reload()  # required to get auto-assigned ports, not needed if it was an already running container
        host_published_grpc_port = container_instance.ports[CONTAINER_GRPC_PORT_BINDING][0]['HostPort']
        print(f"grpc port: {host_published_grpc_port}")
        # block until container exits
        exit_code = container_instance.wait()
        print(f"{container_instance.name} exited {exit_code}")

    except KeyboardInterrupt:
        if container_instance is not None:
            print("Cleaning up container...")
            print(f"Removing container {container_instance.name}")
            container_instance.remove(force=True)
            print(f"Container {container_instance.name} removed")

