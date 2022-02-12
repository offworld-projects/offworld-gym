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
import subprocess
import signal
import psutil
import threading
from abc import abstractmethod
from abc import ABCMeta

# gym and ros
import gym

# other dependencies
import paramiko
import logging
from json import dumps, loads
import roslibpy
import numpy as np

logger = logging.getLogger(__name__)
# level = logging.DEBUG
# logger.setLevel(level)


OFFWORLD_GYM_DOCKER_IMAGE = os.environ.get("OFFWORLD_GYM_DOCKER_IMAGE", "offworldai/offworld-gym")
CONTAINER_IP = "172.20.0.10"
GAZEBO_SERVER_INTERNAL_PORT = 11345
CONTAINER_INTERNAL_GAZEBO_PORT_BINDING = f'{GAZEBO_SERVER_INTERNAL_PORT}/tcp'

GAZEBO_WEB_SERVER_INTERNAL_PORT = 8080
CONTAINER_INTERNAL_GAZEBO_WEB_PORT_BINDING = f'{GAZEBO_WEB_SERVER_INTERNAL_PORT}/tcp'


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
            
            # initialize a ros bridge client
            self._rosbridge_client = roslibpy.Ros(host=CONTAINER_IP, port=9090)
            self._rosbridge_client.run()

            try:
                self.launch_node()
            except:
                import traceback
                traceback.print_exc()
    
    def _start_container(self):

        docker_run_command = f"docker-compose up"
        logger.debug(f"Docker run command is:\n{docker_run_command}\n")
        # container_id = subprocess.check_output(["/bin/bash", "-c", container_start_command]).decode("utf-8").strip()
        
        # check_ip_command = "docker inspect f"{container_name}" | grep 'IPAddress' | head -n 1"
        # self._container_ip = subprocess.check_output(["/bin/bash", "-c", check_ip_command]).decode("utf-8").strip()
        # logger.debug(f"Docker container ip is:\n{self._container_ip}\n")

    # def _send_command(self, shell, data):
    #     # send command throgh a specific shell session
    #     while not shell.send_ready(): time.sleep(1)
    #     shell.send(str(data))

    # def _recv_data(self, shell, size):
    #     # recieve command's respoinse throgh a specific shell session
    #     while not shell.recv_ready(): time.sleep(1)
    #     data = shell.recv(size)
    #     return data

    def launch_node(self):
        """Launches the gazebo world in the docker

        Launches a ROS node containing a gazebo world, spawns a robot in the world
        remotely by sending launch command over ssh
        """
        try:
            # start ssh connection
            # self._ssh_client = paramiko.SSHClient()
            # self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # self._ssh_client.connect(hostname=CONTAINER_IP,username=’root’,password=’offworld_gym’,look_for_keys=False, allow_agent=False)
            # self._ros_shell = self._ssh_client.invoke_shell() # open a shell seesion for roslaunch

            # launch environment remotely using ssh
            # ros_env = os.environ.copy()
            # if ros_env.get("ROSLAUNCH_PYTHONPATH_OVERRIDE", None) is not None:
            #     ros_env["PYTHONPATH"] = ros_env["ROSLAUNCH_PYTHONPATH_OVERRIDE"]

            # # pass launch arguments and launch gazebo
            # self._send_command(self._ros_shell, f"export "{ros_env["PYTHONPATH"]}")
            # rospack_path = os.path.join(rospack.get_path(self.package_name))
            
            # roslaunch_command = f"roslaunch {self.package_name}  {self.launch_file}"
            # self._send_command(self._ros_shell, roslaunch_command)

            # roslaunch_command = 'curl --data "{\"package_name\": \"gym_offworld_monolith\", \"launch_file_name\":\"env_bringup.launch\"}" \
            #                     --header "Content-Type: application/json" \
            #                     --request POST \
            #                     http://127.0.0.1:8008/'

            logger.info("The environment has been started.")

        except Exception:
            logger.error("Environment cannont be launched in the docker.")
            import traceback
            traceback.print_exc() 

    def call_ros_service(self, service_name, service_type, data=None):
        service = roslibpy.Service(self._rosbridge_client, service_name, service_type)
        if not data:
            request = roslibpy.ServiceRequest()
        else:
            request = roslibpy.ServiceRequest(data)

        logger.info(f'Calling service {service_name}')
        result = service.call(request)
        return result
        
    def register_publisher(self, topic_name, message_type):
        publisher= roslibpy.Topic(ros=self._rosbridge_client, name=topic_name, message_type=message_type)
        return publisher

    def register_subscriber(self, topic_name, message_type, placeholder, queue_size):
        def update_odom(msg):
            self._latest_odom_message = msg

        def update_sim_time_from_clock(msg):
            self._latest_clock_message = msg

        def update_depth_img(msg):
            self._latest_depth_img = msg

        def update_rgb_img(msg):
            self._latest_rgb_img = msg

        # def decode_image(msg):
        #     base64_bytes = msg['data'].encode('ascii')
        #     image_bytes = base64.b64decode(base64_bytes)
        #     msg = np.array(image_bytes)

        subscriber = roslibpy.Topic(ros=self._rosbridge_client, name=topic_name, message_type=message_type, queue_size=queue_size)  
        
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