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

from offworld_gym import version

__version__     = version.__version__

#std
import os
import sys
import time
import uuid
import subprocess
import signal
import atexit
import threading
from abc import abstractmethod
from abc import ABCMeta

# gym and ros
import gym

# other dependencies
import base64
import logging
from json import dumps, loads
import roslibpy
import numpy as np

logger = logging.getLogger(__name__)
level = logging.INFO
logger.setLevel(level)


OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "offworldai/offworld-gym")
ROS_BRIDGE_PORT = 9090
GAZEBO_SERVER_INTERNAL_PORT = 11345
GAZEBO_WEB_SERVER_INTERNAL_PORT = 8080
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
            
            # initialize a ros bridge client
            self._rosbridge_client = roslibpy.Ros(host=self._container_ip, port=ROS_BRIDGE_PORT)
            self._rosbridge_client.run()

            # try:
            #     self.launch_node()
            # except:
            #     import traceback
            #     traceback.print_exc()
    
    def _start_container(self):

        # container_name = f"offworld-gym{uuid.uuid4().hex[:10]}"
        container_name = "gym-test"
        # container_entrypoint = "/offworld-gym/offworld_gym/envs/gazebo_docker/docker_entrypoint.sh"
        # container_env_str = "DISPLAY"
        # container_volumes_str = f"{XSERVER_VOLUME}:{XSERVER_VOLUME}"
        # container_ports_str = f"-p {ROS_BRIDGE_PORT}:{ROS_BRIDGE_PORT} -p {GAZEBO_WEB_SERVER_INTERNAL_PORT}:{GAZEBO_WEB_SERVER_INTERNAL_PORT}"
        # docker_run_command = f"docker run --name \'{container_name}\' -it -d --rm" \
        #                      f"{container_env_str}{container_volumes_str}{container_ports_str} " \
        #                      f"offworldai/offworld-gym:latest {container_entrypoint}"
        # logger.debug(f"Docker run command is:\n{docker_run_command}\n")
        # container_id = subprocess.check_output(["/bin/bash", "-c", docker_run_command]).decode("utf-8").strip()
        # import pdb; pdb.set_trace()
        filter_string = "'{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}'"
        check_ip_command = f"docker inspect -f {filter_string} {container_name}"
        self._container_ip = subprocess.check_output(["/bin/bash", "-c", check_ip_command]).decode("utf-8").strip()
        logger.debug(f"Docker container ip is:\n{self._container_ip}\n")

        # # ensure cleanup at exit
        # def kill_container_if_it_still_exists():
        #     try:
        #         # bash command kills the container if it exists, otherwise return error code 1 without printing an error
        #         kill_command = f"docker ps -q --filter \"id={container_id}\" | grep -q . && docker kill {container_id}"
        #         removed_container = subprocess.check_output(['bash', '-c', kill_command]).decode("utf-8").strip()
        #         print(f"Cleaned up container {removed_container}")
        #     except subprocess.CalledProcessError:
        #         pass

        # atexit.register(kill_container_if_it_still_exists)

    def launch_node(self):
        """Launches the gazebo world in the docker

        Launches a ROS node containing a gazebo world, spawns a robot in the world
        remotely by sending launch command over ssh
        """
        try:
            roslaunch_command = 'curl --data "{\"command_name\": \"launch_node\", \"package_name\": \"gym_offworld_monolith\", \"launch_file_name\":\"env_bringup.launch\"}" \
                                --header "Content-Type: application/json" \
                                --request POST \
                                http://127.0.0.1:8008/'

            subprocess.check_output(['bash', '-c', roslaunch_command])

            logger.info("The environment has been started.")

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
        
    def register_publisher(self, topic_name, message_type, queue_size=10, throttle_rate=300):
        """Register publisher
        """
        publisher= roslibpy.Topic(ros=self._rosbridge_client, name=topic_name, message_type=message_type,
                    queue_size=queue_size, throttle_rate=throttle_rate)
        return publisher

    def register_subscriber(self, topic_name, message_type, placeholder, queue_size=10, throttle_rate=300):
        """Register subscriber and consistently listening
        """
        def update_odom(msg):
            self._latest_odom_message = msg

        def update_sim_time_from_clock(msg):
            self._latest_clock_message = msg

        def update_depth_img(msg):
            # print("Encoding of depth img", msg['encoding'])
            base64_bytes = msg['data'].encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            np_image = np.frombuffer(image_bytes, dtype=np.float32)
            self._latest_depth_img = np_image.reshape((msg['height'],msg['width'],-1))

        def update_rgb_img(msg):
            # print("Encoding of rgb img", msg['encoding'])
            base64_bytes = msg['data'].encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            np_image = np.frombuffer(image_bytes, dtype=np.uint8)
            self._latest_rgb_img = np_image.reshape((msg['height'],msg['width'],-1))

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