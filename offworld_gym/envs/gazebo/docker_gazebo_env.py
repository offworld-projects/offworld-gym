#!/usr/bin/env python

# Copyright 2022 OffWorld Inc.
# Doing business as Off-World AI, Inc. in California.
# All rights reserved.
#
# Licensed under GNU General Public License v3.0 (the "License")
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law, any source code or other materials
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, express or implied.
from typing import NamedTuple

from offworld_gym import version

__version__ = version.__version__

# std
import os
import time
import uuid
import subprocess
import atexit
from abc import abstractmethod
from abc import ABCMeta

# other dependencies
import gym
import logging
import json
import ast
import requests
import roslibpy
import socket

logger = logging.getLogger(__name__)
level = logging.DEBUG
logger.setLevel(level)

OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "offworldai/offworld-gym")
GAZEBO_WEB_SERVER_INTERNAL_PORT = 8080
HTTP_COMMAND_SERVER_PORT = 8008
XSERVER_VOLUME = "/tmp/.X11-unix"


class DockerizedGazeboEnv(gym.Env, metaclass=ABCMeta):
    """Base class for Dockerized Gazebo based Gym environments

    Attributes:
        package_name: String containing the ROS package name
        launch_file: String containing the name of the environment's launch file.
        node_name: String with a ROS node name for the environment's node
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, package_name, launch_file, node_name='gym_offworld_env'):

        assert package_name is not None and package_name != '', "Must provide a valid package name."
        assert launch_file is not None and launch_file != '', "Must provide a valid launch file name."

        self.package_name = package_name
        self.launch_file = launch_file
        self.node_name = node_name

        self._latest_odom_message = None
        self._latest_clock_message = None
        self._latest_depth_img = None
        self._latest_rgb_img = None

        self._start_container()

        # Run "xhost local:" command to enable XServer to be written from localhost over network
        xhost_command = "xhost +local:docker"
        try:
            subprocess.check_output(['bash', '-c', xhost_command])
        except subprocess.CalledProcessError:
            logger.warning(f"The bash command \"{xhost_command}\" failed. "
                           f"Installing \'xhost\' may be required for OffWorldDockerizedEnv to render properly. "
                           f"Further issues may be caused by this.")

        try:
            self.launch_node()
        except:
            import traceback
            traceback.print_exc()

        # initialize a ros bridge client
        self._rosbridge_client = roslibpy.Ros(host=str('localhost'), port=int(self._ports.ros_port))
        self._rosbridge_client.run()

    def _start_container(self):
        """ Start a container, get id and ip
        """
        container_name = f"offworld-gym-{uuid.uuid4().hex[:10]}"

        # retracted to "docker run" for automatic ip and container name assignment
        container_entrypoint = "/offworld-gym/offworld_gym/envs/gazebo/docker_entrypoint.sh"
        container_env_str = " -e DISPLAY"
        container_volumes_str = f" -v {XSERVER_VOLUME}:{XSERVER_VOLUME}"

        self._ports = _find_open_ports()
        container_ports_str = f" -p {self._ports.ros_port}:{self._ports.ros_port}" \
                              f" -p {self._ports.gazebo_port}:{GAZEBO_WEB_SERVER_INTERNAL_PORT}" \
                              f" -p {self._ports.server_port}:{HTTP_COMMAND_SERVER_PORT}"
        docker_run_command = f"docker run --name \'{container_name}\' -it -d --rm " \
                             f"{container_env_str}{container_volumes_str}{container_ports_str} " \
                             f"offworldai/offworld-gym:latest {container_entrypoint}"
        _ = subprocess.Popen(["/bin/bash", "-c", docker_run_command])
        logger.debug(f"Docker run command is:\n{docker_run_command}\n")
        time.sleep(10.0)
        # get the container id for corresponding container
        get_container_id_command = f'docker ps -aqf "name={container_name}"'
        container_id = subprocess.check_output(["/bin/bash", "-c", get_container_id_command]).decode("utf-8").strip()
        logger.debug(f"Docker container id is:\n{container_id}\n")
        # get the ip address for corresponding container
        filter_string = "'{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}'"
        get_container_ip_command = f"docker inspect -f {filter_string} {container_name}"
        self._container_ip = subprocess.check_output(["/bin/bash", "-c", get_container_ip_command]).decode(
            "utf-8").strip()
        logger.debug(f"Docker container ip is:\n{self._container_ip}\n")

        # ensure cleanup at exit
        def kill_container_if_it_still_exists():
            try:
                # bash command kills the container if it exists, otherwise return error code 1 without printing an error
                kill_command = f"docker ps -q --filter \"id={container_id}\" | grep -q . && docker kill {container_id}"
                removed_container = subprocess.check_output(['bash', '-c', kill_command]).decode("utf-8").strip()
                # remove_service = subprocess.Popen(["/bin/bash", "-c", "docker-compose down"])
                print(f"Cleaned up container {removed_container}")
            except subprocess.CalledProcessError:
                pass

        atexit.register(kill_container_if_it_still_exists)

    def launch_node(self):
        """Launches the gazebo world in the docker.
        Localhost: sending launch command to http server inside container.
        Container: Launches a ROS node containing a gazebo world, spawns a robot in the world.
        """
        try:
            headers = {'Content-type': 'application/json'}
            json_data = f'{{"command_name": "launch_node", "package_name": "gym_offworld_monolith", "launch_file_name":"env_bringup.launch", "ros_port": {self._ports.ros_port}}}'
            json_data = ast.literal_eval(json_data)
            _ = requests.post(f"http://127.0.0.1:{self._ports.server_port}/", data=json.dumps(json_data), headers=headers)
            logger.info("The environment has been started.")
            print(f"The simulation can be viewed at 'http://localhost:{self._ports.gazebo_port}'")
        except Exception:
            logger.error("Environment cannont be launched in the docker.")
            import traceback
            traceback.print_exc()

    def register_ros_service(self, service_name, service_type):
        """Register service 
        """
        service = roslibpy.Service(self._rosbridge_client, service_name, service_type)
        logger.debug(f'Registering service {service_name}')
        return service

    def call_ros_service(self, service_proxy, service_name, service_type, data=None):
        """Call service 
        """
        if not data:
            request = roslibpy.ServiceRequest()
        else:
            request = roslibpy.ServiceRequest(data)

        respond = service_proxy.call(request)
        logger.debug(f'Calling service {service_name}')
        return respond

    def register_publisher(self, topic_name, message_type, queue_size=1, throttle_rate=300):
        """Register publisher
        """
        publisher = roslibpy.Topic(ros=self._rosbridge_client, name=topic_name, message_type=message_type,
                                   queue_size=queue_size, throttle_rate=throttle_rate)
        return publisher

    def register_subscriber(self, topic_name, message_type, placeholder, queue_size=1, throttle_rate=200):
        """Register subscriber and consistently listening
        """

        def update_odom(msg):
            self._latest_odom_message = msg

        def update_sim_time_from_clock(msg):
            self._latest_clock_message = msg

        def update_depth_img(msg):
            self._latest_depth_img = msg

        def update_rgb_img(msg):
            self._latest_rgb_img = msg

        subscriber = roslibpy.Topic(ros=self._rosbridge_client, name=topic_name, message_type=message_type,
                                    queue_size=queue_size, throttle_rate=throttle_rate)

        if self._rosbridge_client.is_connected:
            if str(topic_name) == '/odom':
                subscriber.subscribe(update_odom)
            elif str(topic_name) == '/clock':
                subscriber.subscribe(update_sim_time_from_clock)
            elif str(topic_name) == '/camera/depth/image_raw':
                subscriber.subscribe(update_depth_img)
            elif str(topic_name) == '/camera/rgb/image_raw':
                subscriber.subscribe(update_rgb_img)
            else:
                logger.error(f"Message type not found {message_type}")
        return subscriber

    def pause_physics_remotely(self):
        """Pause physics remotely from localhost, send command to docker command server
        """
        headers = {'Content-type': 'application/json'}
        json_data = '{"command_name": "pause"}'
        json_data = ast.literal_eval(json_data)
        _ = requests.post(f"http://127.0.0.1:{self._ports.server_port}/", data=json.dumps(json_data), headers=headers)

    def unpause_physics_remotely(self):
        """Unpause physics remotely from localhost, send command to docker command server
        """
        # import pdb; pdb.set_trace()
        headers = {'Content-type': 'application/json'}
        json_data = '{"command_name": "unpause"}'
        json_data = ast.literal_eval(json_data)
        _ = requests.post(f"http://127.0.0.1:{self._ports.server_port}/", data=json.dumps(json_data), headers=headers)

    def publish_cmd_vel_remotely(self, lin_x_speed, ang_z_speed):
        """send cmd_vel from localhost to python server inside docker, then publish cmd_vel in as bash command inside 
        """
        headers = {'Content-type': 'application/json'}
        json_data = '{"command_name": "cmd_vel", "lin_x_speed":' + f'"{lin_x_speed}"' + ', "lin_y_speed":"0.0","lin_z_speed":"0.0", "ang_z_speed":' + f'"{ang_z_speed}"' + ',"ang_x_speed":"0.0", "ang_y_speed":"0.0"}'
        json_data = ast.literal_eval(json_data)
        _ = requests.post(f"http://127.0.0.1:{self._ports.server_port}/", data=json.dumps(json_data), headers=headers)

    @abstractmethod
    def step(self, action):
        """Abstract step method to be implemented in a child class
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Abstract reset method to be implemented in a child class
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, mode='human'):
        """Abstract render method to be implemented in a child class
        """
        raise NotImplementedError

    def close(self):
        """Closes environment and all processes created for the environment
        """
        # ask container to run these
        os.system("killall -9 -u `whoami` gzclient")
        os.system("killall -9 -u `whoami` gzserver")
        os.system("killall -9 -u `whoami` rosmaster")
        os.system("killall -9 -u `whoami` roscore")
        # kill the container


class DockerPorts(NamedTuple):
    ros_port: str
    gazebo_port: str
    server_port: str

    def __str__(self):
        return f'rosbridge port: {self.ros_port}\n' \
               f'gazebo port: {self.gazebo_port}\n' \
               f'server port: {self.server_port}'


def _find_open_ports():
    """Finds 3 random open ports to assign to the host for communication with the Docker container
    This allows multiple containers to run simultaneously without conflicting
    """
    ros_sock = socket.socket()
    ros_sock.bind(('localhost', 0))
    _, ros_port = ros_sock.getsockname()

    gazebo_sock = socket.socket()
    gazebo_sock.bind(('localhost', 0))
    _, gazebo_port = gazebo_sock.getsockname()

    server_sock = socket.socket()
    server_sock.bind(('localhost', 0))
    _, server_port = server_sock.getsockname()

    ros_sock.close()
    gazebo_sock.close()
    server_sock.close()

    return DockerPorts(ros_port, gazebo_port, server_port)
