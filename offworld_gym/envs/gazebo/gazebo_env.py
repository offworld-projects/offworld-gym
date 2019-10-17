#!/usr/bin/env python

# Copyright 2019 OffWorld Inc.
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

#gym
from offworld_gym import logger
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from offworld_gym.envs.common.exception.gym_exception import GymException

#ros
import rospy
import rospkg


class GazeboGymEnv(gym.Env):
    """Abstract Gazebo based Gym environments
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, package_name, launch_file, node_name='gym_offworld_env'):

        assert package_name is not None and package_name != '', "Must provide a valid package name."
        assert launch_file is not None and launch_file != '', "Must provide a valid launch file name."
        
        self.package_name = package_name
        self.launch_file = launch_file
        self.node_name = node_name

        try:
            self.launch_node()
        except:
            import traceback
            traceback.print_exc()

    def launch_node(self):
        """Launches the gazebo world 

        Launches a ros node containing a gazebo world, spawns a robot in the world
        """
        try:
            rospack = rospkg.RosPack()
            if rospack.get_path(self.package_name) is None:
                raise GymException("The ROS package does not exist.")
        
            self.roslaunch_process = subprocess.Popen(['roslaunch', os.path.join(rospack.get_path(self.package_name), "launch", self.launch_file)])
            self.roslaunch_wait_thread = threading.Thread(target=self._process_waiter, args=(1,))
            self.roslaunch_wait_thread.start()
        except OSError:
            print("Ros node could not be started.")
            import traceback
            traceback.print_exc() 

        rospy.loginfo("The environment has been started.")
        rospy.init_node(self.node_name, anonymous=True)
        rospy.on_shutdown(self.close)

    def _process_waiter(self, popen):
        try: 
            self.roslaunch_process.wait()
            logger.info("Roslaunch has finished.")
        except: 
            logger.error("An error occured while waiting for roslaunch to finish.")

    def step(self, action):
        """Gym step function

        Must be implemented in a child class
        """
        raise NotImplementedError

    def reset(self):
        """Gym reset function

        Must be implemented in a child class
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """Gym render function

        Must be implemented in a child class
        """
        raise NotImplementedError

    def close(self):
        """Closes environment and all processes created for the environment
        """
        rospy.loginfo("Closing the environment and all processes. ")
        try:
            launch_process = psutil.Process(self.roslaunch_process.pid)
            launch_children = launch_process.children(recursive=True)
            for process in launch_children:
                process.send_signal(signal.SIGTERM)
        except:
            import traceback
            traceback.print_exc()
        
        #force any lingering processes to shutdown
        os.system("killall -9 -u `whoami` gzclient")
        os.system("killall -9 -u `whoami` gzserver")
        os.system("killall -9 -u `whoami` rosmaster")
        os.system("killall -9 -u `whoami` roscore")


