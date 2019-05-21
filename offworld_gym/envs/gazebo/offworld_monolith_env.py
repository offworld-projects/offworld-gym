#!/usr/bin/env python
# Copyright offworld.ai 2019
from offworld_gym import version

__author__      = "Ashish Kumar"
__copyright__   = "Copyright offworld.ai 2019"
__license__     = "None"
__version__     = version.__version__
__maintainer__  = "Ashish Kumar"
__email__       = "ashish.kumar@offworld.ai"
__status__      = "Development"

#lib
from math import pi
import time
import numpy as np
from scipy.spatial import distance

#gym
import gym
from gym import utils, spaces
from offworld_gym import version
from offworld_gym.envs.gazebo.gazebo_env import GazeboGymEnv
from gym.utils import seeding
from offworld_gym.envs.common.offworld_gym_utils import ImageUtils
from offworld_gym.envs.common.offworld_gym_utils import GazeboUtils
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions

#ros
import rospy
from std_srvs.srv import Empty as Empty_srv
from geometry_msgs.msg import Twist
import tf
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image

class OffWorldMonolithEnv(GazeboGymEnv):
    """Simulated Gym environment with a offworld and a monolith on an uneven terrain

    A RL agent learns to reach the goal(monolith) in shortest time

    Usage:
        env = gym.make('OffWorldMonolithSimEnv-v0', channel_type=Channels.DEPTHONLY)
        env = gym.make('OffWorldMonolithSimEnv-v0', channel_type=Channels.RGB_ONLY)
        env = gym.make('OffWorldMonolithSimEnv-v0', channel_type=Channels.RGBD)
    """
    _PROXIMITY_THRESHOLD = 0.20

    def __init__(self, channel_type=Channels.DEPTH_ONLY, random_init=False):
        super(OffWorldMonolithEnv, self).__init__(package_name='gym_offworld_monolith', launch_file='env_bringup.launch')

        assert isinstance(channel_type, Channels), "Channel type is not of Channels."
        rospy.loginfo("Environment has been initiated.")

        #gazebo
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty_srv)
        rospy.loginfo("Service proxies have been initiated.")

        #rosbot
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)

        # environment
        self.seed()
        self.channel_type = channel_type
        self.random_init = random_init

        self.observation_space = spaces.Box(0, 255, shape = (1, ImageUtils.IMG_H, ImageUtils.IMG_W, channel_type.value))
        self.action_space = spaces.Discrete(4)
        self._monolith_space = self._get_state_vector('monolith')
        rospy.logdebug("----------------Monolith----------------")
        rospy.logdebug("Pose x: {}".format(str(self._monolith_space[0])))
        rospy.logdebug("Pose y: {}".format(str(self._monolith_space[1])))
        rospy.logdebug("Pose z: {}".format(str(self._monolith_space[2])))
        rospy.logdebug("----------------------------------------")
        rospy.loginfo("Environment has been started.")

    def seed(self, seed=None):
        """Seed the environment
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_state(self):
        """Encapsulates state of the environment as captured by an rgbd sensor in a numpy array
        Returns:
            state: The state of the environment as captured by the robot's
                rgbd sensor
        """
        rgb_data = None
        img = None
        while rgb_data is None:
            try:
                rgb_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                img = ImageUtils.process_img_msg(rgb_data)
            except:
                rospy.logerr("Error: '/camera/rgb/image_raw'timeout.")
        
        depth_data = None
        depth_img = None
        while depth_data is None:
            try:
                depth_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                depth_img = ImageUtils.process_depth_msg(depth_data)
            except:
                rospy.logerr("Error: '/camera/depth/image_raw'timeout.")
        
        if self.channel_type == Channels.DEPTH_ONLY:
            state = depth_img
        elif self.channel_type == Channels.RGB_ONLY:
            state = img
        elif self.channel_type == Channels.RGBD:
            state =  np.concatenate((img, depth_img)) 
        rospy.loginfo("State of the environment captured.")
        return state

    def _move_rosbot(self, lin_x_speed, ang_z_speed, sleep_time=2):
        """Moves the ROSBot 

        Accepts linear x speed and angular z speed and moves the
        ROSBot by issuing the velocity commands to the ROSBot.
        
        Args:
            lin_x_speed: linear speed in x-direction
            ang_z_speed: angular speed about z-direction
        """
        vel_cmd = Twist()
        vel_cmd.linear.x = lin_x_speed
        vel_cmd.angular.z = ang_z_speed
        self.vel_pub.publish(vel_cmd)
        
        time.sleep(sleep_time)
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
        
    def _send_action_commands(self, action_type):
        """Sends an action command to the robot
        Args:
            action_type: FourDiscreteMotionActions instance 
        """
        if action_type == FourDiscreteMotionActions.LEFT:
            self._move_rosbot(0.0, -0.25) 
        elif action_type == FourDiscreteMotionActions.RIGHT:
            self._move_rosbot(0.0, 0.25) 
        elif action_type == FourDiscreteMotionActions.FORWARD:
            self._move_rosbot(0.15, 0.0)
        elif action_type == FourDiscreteMotionActions.BACKWARD:
            self._move_rosbot(-0.15, 0.0)
            
    def _get_model_state(self, name):
        """Get model state object

        Args:
            name: name of the model
        Returns:
            state: state of the model contained in a GetModelState service message
        """
        try:
            g_getStateVector = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            rospy.wait_for_service("/gazebo/get_model_state")
            state = g_getStateVector(model_name=name)
            return state
        except:
            rospy.logerr("An error occured while invoking the get_model_state service.")
            return None

    def _get_state_vector(self, name):
        """Get state vector

        Args:
            name: name of the model
        Returns:
            state: A vector representing 4-DOF of a model (x, y, z, w)
        """
        try:
            ms = self._get_model_state(name)
            if ms is not None:
                quaternion = (
                    ms.pose.orientation.x,
                    ms.pose.orientation.y,
                    ms.pose.orientation.z,
                    ms.pose.orientation.w
                )

                euler = tf.transformations.euler_from_quaternion(quaternion)
                state = (ms.pose.position.x, ms.pose.position.y, ms.pose.position.z, euler[2])
                return state
            else:
                return None
        except:
            rospy.logerr("An error occured while creating the state vector.")
            return None

    def _calculate_reward(self):
        """Calculates the reward at every step
        Returns:
            reward: Reward from the environment
            done: A flag which is true when an episode is complete
        """
        rosbot_state = self._get_state_vector('rosbot')
        dst = distance.euclidean(rosbot_state[0:3], self._monolith_space[0:3])

        if dst < OffWorldMonolithEnv._PROXIMITY_THRESHOLD:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        return reward, done

    def step(self, action):
        """Take an action in the environment

        Args:
            action: An action to be taken in the environment
        Returns:
            state: The state of the environment as captured by the robot's
                rgbd sensor
            reward: Reward from the environment
            done: A flag which is true when an episode is complete
            info: No info given for fair learning :)
        """
        # unpause physics before interaction with the environment
        GazeboUtils.unpause_physics()

        assert action is not None, "Action cannot be None."
        assert isinstance(action, FourDiscreteMotionActions) or isinstance(action, int), "Action type is not recognized."

        if isinstance(action, int):
            assert action >= 0 and action < 4, "Unrecognized value for the action"
            action = FourDiscreteMotionActions(action)

        self._send_action_commands(action)

        state = self._get_state()

        reward, done = self._calculate_reward()

        # pause physics now, speeds up simulation
        GazeboUtils.pause_physics()
        
        return state, reward, done, {}

    def _move_to_random_position(self, model_name, orient=True):
        """re-position a model to a random position

        Args:
            name - name of the model
        Raises:
            GymException: Exception occured while resetting the environment.
        """
        try:
            assert model_name is not None or model_name != ''
            goal_state = ModelState()
            goal_state.model_name = model_name
            goal_state.pose.position.x, goal_state.pose.position.y = np.random.uniform(-2.0, 2.0, size=(2,))
            goal_state.pose.position.z = -0.1
                
            if orient:
                goal_state.pose.orientation.z, goal_state.pose.orientation.w = np.random.uniform(-3.14, 3.14, size=(2,))
            GazeboUtils.move_to_position(goal_state)
        except:
            rospy.logerr("An error occured while resetting the environment.")
            raise GymException("An error occured while resetting the environment.")
    
    def _move_to_original_position(self, model_name):
        """re-position a model to original position

        Args:
            name: name of the model.
        Raises:
            GymException: Exception occured while resetting the environment.
        """
        try:
            assert model_name is not None or model_name != ''

            goal_state = ModelState()
            goal_state.model_name = model_name
            goal_state.pose.position.x = 0.0
            goal_state.pose.position.y = 0.0
            goal_state.pose.position.z = -0.1

            GazeboUtils.move_to_position(goal_state)
        except:
            rospy.logerr("An error occured while resetting the environment.")
            raise GymException("An error occured while resetting the environment.")

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state as seen by
            the robot
        """
        GazeboUtils.pause_physics()

        if self.random_init:
            self._move_to_random_position('rosbot')
        else:
            self._move_to_original_position('rosbot')            

        GazeboUtils.unpause_physics()

        state = self._get_state()
        return state

    def render(self, mode='human'):
        """
        """
        #TODO
        raise NotImplementedError