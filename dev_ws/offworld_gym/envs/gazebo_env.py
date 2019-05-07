
#!/usr/bin/env python
# Copyright offworld.ai 2018
from offworld_gym import version

import os
import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import rospy
import roslaunch
import rospkg

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

class GazeboGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, package_name, launch_file, node_name='gym_rosbot_env'):

        assert(package_name is not None and package_name != '', "Must provide a valid package name.")
        assert(launch_file is not None and launch_file != '', "Must provide a valid launch file name.")
        
        rospy.init_node(node_name, anonymous=True)

        rospack = rospkg.RosPack()
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        
        self.env_launch = roslaunch.parent.ROSLaunchParent(uuid, [os.path.join(rospack.get_path(package_name), "launch", launch_file)])
        self.env_launch.start()
        rospy.loginfo("started")

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        self.env_launch.shutdown()

        #clean up
        os.system("killall -9 -u `whoami` gzclient")
        os.system("killall -9 -u `whoami` gzserver")
        os.system("killall -9 -u `whoami` rosmaster")
        os.system("killall -9 -u `whoami` roscore")


