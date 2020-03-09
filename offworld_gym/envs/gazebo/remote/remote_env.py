import docker
import os
import uuid
# from offworld_gym import logger


OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "ros-gym")

if __name__ == '__main__':
    client = docker.from_env()

    container_env = {
        "DISPLAY": os.environ['DISPLAY']
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

    }

    container_name = f"offworld-gym{uuid.uuid4().hex[:10]}"

    container_entrypoint = "/offworld_gym/envs/gazebo/remote/docker_entrypoint.sh"

    container_instance = None
    try:
        container_instance = client.containers.run(image=OFFWORLD_GYM_DOCKER_IMAGE,
                              entrypoint=container_entrypoint,
                              name=container_name,
                              hostname=container_name,
                              detach=True,
                              auto_remove=True,
                              environment=container_env,
                              volumes=container_volumes,
                              ports=container_ports)
        print(f"{container_instance.name} launched")
        # block until container exits
        exit_code = container_instance.wait()
        print(f"{container_instance.name} exited {exit_code}")

    except KeyboardInterrupt:
        if container_instance is not None:
            print("Cleaning up containers...")
            print(f"Removing container {container_instance.name}")
            container_instance.remove(force=True)
            print(f"Container {container_instance.name} removed")

