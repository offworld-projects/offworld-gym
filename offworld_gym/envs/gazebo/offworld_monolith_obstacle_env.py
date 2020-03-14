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

#lib
from math import pi
import time
import numpy as np
from scipy.spatial import distance
import pdb
from pyquaternion import Quaternion
from matplotlib import pyplot as plt

#gym
import gym
from gym import utils, spaces
from offworld_gym.envs.gazebo.gazebo_env import GazeboGymEnv
from gym.utils import seeding
from offworld_gym.envs.gazebo.utils import ImageUtils, GazeboUtils
from offworld_gym.envs.common.exception.gym_exception import GymException
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.actions import FourDiscreteMotionActions
from offworld_gym.envs.gazebo.offworld_monolith_env import DEFAULT_MAX_DEPTH_VALUE, DEFAULT_IMAGE_OUT_SIZE

#ros
import rospy
from std_srvs.srv import Empty as Empty_srv
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Image


class OffWorldMonolithObstacleEnv(GazeboGymEnv):
    """Generic Simulated gym environment that replicates the real OffWorld Monolith Obstacle environment in Gazebo.  

    Agent receives RGB or Depth camera feed as input and needs to learn to approch the visual
    beacon in the center of the environment. Agent receives a sparse reward of `+1` upon
    approaching the monolith within ``_PROXIMITY_THRESHOLD`` radius.

    Attributes:
        channel_type: Channels type value indicating channel type to use for observation.
        random_init: Boolean value indicating whether robot spawns at a random location after reset.
        step_count: An integer count of step during an episode.
        observation_space: Gym data structure that encapsulates an observation.
        action_space: Gym data structure that encapsulates an action.
    """
    _PROXIMITY_THRESHOLD = 0.50
    _EPISODE_LENGTH = 100
    _TIME_DILATION = 10.0 # Has to match `<real_time_factor>` in `offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/worlds/offworld_monolith_environment.world`

    def __init__(self,
                 channel_type=Channels.DEPTH_ONLY,
                 random_init=True,
                 clip_depth_value=DEFAULT_MAX_DEPTH_VALUE,
                 image_out_size=DEFAULT_IMAGE_OUT_SIZE):

        super(OffWorldMonolithObstacleEnv, self).__init__(package_name='gym_offworld_monolith', launch_file='env_obstacle_bringup.launch')

        assert isinstance(channel_type, Channels), "Channel type is not of Channels."
        rospy.loginfo("Environment has been initiated.")

        # gazebo
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty_srv)
        rospy.loginfo("Service proxies have been initiated.")

        # rosbot
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=2)

        # environment
        self.seed()
        self.channel_type = channel_type
        self.random_init = random_init
        self.step_count = 0
        self._current_state = None
        self._clip_depth_value = clip_depth_value
        self._image_out_w = image_out_size[0]
        self._image_out_h = image_out_size[1]

        self.observation_space = spaces.Box(0.0, 1.0, shape=(self._image_out_w, self._image_out_h, channel_type.value))
        self.action_space = None
        self._monolith_space = self._get_state_vector('monolith')
        self._boulder_poses = [self._get_state_vector('boulder_' + str(i)) for i in range(0,14)]
        rospy.logdebug("----------------Monolith----------------")
        rospy.logdebug("Pose x: {}".format(str(self._monolith_space[0])))
        rospy.logdebug("Pose y: {}".format(str(self._monolith_space[1])))
        rospy.logdebug("Pose z: {}".format(str(self._monolith_space[2])))
        rospy.logdebug("----------------------------------------")
        rospy.loginfo("Environment has been started.")

    def seed(self, seed=None):
        """Calls ``gym`` strong random seed generator.

        Args:
            seed: None seeds from an operating system specific randomness source.
        """        
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_state(self):
        """Encapsulates state of the environment as captured by an rgbd sensor in a numpy array.
        
        Returns:
            Numpy array with the state of the environment as captured by the robot's rgbd sensor.
        """
        rgb_data = None
        rgb_img = None
        while rgb_data is None and not rospy.is_shutdown():
            try:
                rgb_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                rgb_img = ImageUtils.process_img_msg(rgb_data,
                                                     resized_width=self._image_out_w,
                                                     resized_height=self._image_out_h,
                                                     max_value_for_clip_and_normalize=255.0)
            except rospy.ROSException:
                rospy.sleep(0.1)
            
        depth_data = None
        depth_img = None
        while depth_data is None and not rospy.is_shutdown():
            try:
                depth_data = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=5)
                depth_img = ImageUtils.process_depth_msg(depth_data,
                                                         resized_width=self._image_out_w,
                                                         resized_height=self._image_out_h,
                                                         max_value_for_clip_and_normalize=self._clip_depth_value)
            except rospy.ROSException:
                rospy.sleep(0.1)

        if self.channel_type == Channels.DEPTH_ONLY:
            state = depth_img
        elif self.channel_type == Channels.RGB_ONLY:
            state = rgb_img
        elif self.channel_type == Channels.RGBD:
            state = np.concatenate((rgb_img, depth_img))
        rospy.logdebug("State of the environment captured.")
        return np.reshape(state, self.observation_space.shape)

    def _move_rosbot(self, lin_x_speed, ang_z_speed, sleep_time=2.):
        """Moves the ROSBot.

        Accepts linear x speed and angular z speed and moves the
        ROSBot by issuing the velocity commands to the ROSBot.
        
        Args:
            lin_x_speed: Float value indicating linear speed in x-direction.
            ang_z_speed: Float value indicating angular speed about z-direction.
            sleep_time: Float value indicating sleep time between velocity command and stopping.
        """
        vel_cmd = Twist()
        vel_cmd.linear.x = lin_x_speed
        vel_cmd.angular.z = ang_z_speed
        self.vel_pub.publish(vel_cmd)
        
        time.sleep(sleep_time/OffWorldMonolithObstacleEnv._TIME_DILATION)
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.vel_pub.publish(vel_cmd)
            
    def _get_model_state(self, name):
        """Get model state object.

        Args:
            name: String name of the model.

        Returns:
            A vector containing state of the model contained in a GetModelState service message.
        """
        try:
            g_getStateVector = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
            rospy.wait_for_service("/gazebo/get_model_state")
            state = g_getStateVector(model_name=name)
            return state
        except rospy.ServiceException:
            rospy.logerr("An error occured while invoking the get_model_state service.")
            return None

    def _get_state_vector(self, name):
        """Get state vector.

        Args:
            name: String name of the model.

        Returns:
            A vector representing 4-DOF of a model (x, y, z, w).
        """
        try:
            ms = self._get_model_state(name)
            if ms is not None:
                quaternion = Quaternion(ms.pose.orientation.w,  ms.pose.orientation.x, ms.pose.orientation.y, ms.pose.orientation.z,)
                yaw, _, _ = quaternion.yaw_pitch_roll
                state = (ms.pose.position.x, ms.pose.position.y, ms.pose.position.z, yaw)
                return state
            else:
                return None
        except rospy.ServiceException:
            rospy.logerr("An error occured while creating the state vector.")
            return None

    def _calculate_reward(self):
        """Calculates the reward at every step.
       
        Returns:
            A number indicating reward from the environment.
            A boolean flag which is true when an episode is complete.
        """
        rosbot_state = self._get_state_vector('rosbot')
        dst = distance.euclidean(rosbot_state[0:3], self._monolith_space[0:3])
        rospy.logdebug("Distance between the rosbot and the monolith is : {}".format(str(dst)))
        
        # check distance to the monolith
        if dst < OffWorldMonolithObstacleEnv._PROXIMITY_THRESHOLD:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        
        # check boundaries
        if rosbot_state[0] < -1.65 or rosbot_state[0] > 1.80 or rosbot_state[1] < -1.50 or rosbot_state[1] > 1.00:
            reward = 0.0
            done = True

        if self.step_count == OffWorldMonolithObstacleEnv._EPISODE_LENGTH:
            done = True
        return reward, done

    def step(self, action):
        raise NotImplementedError("Must be implemented in the child class.")

    def _move_to_random_position(self, model_name, orient=True):
        """re-position a model to a random position.

        Args:
            name: String name of the model.
            orient: Boolean value to indicate if the robot is randomly oriented.

        Raises:
            GymException: Exception occured while resetting the environment.
        """
        try:
            assert model_name is not None or model_name != ''
            goal_state = ModelState()
            goal_state.model_name = model_name

            # random spawn location exlucing no-spawn 30cm radius around the monolith
            goal_state.pose.position.x = self._monolith_space[0]
            goal_state.pose.position.y = self._monolith_space[1]
            while True:
                goal_state.pose.position.x = np.random.uniform(-1.75, 1.90)
                goal_state.pose.position.y = np.random.uniform(-1.60, 1.10)
                flag = distance.euclidean((goal_state.pose.position.x, goal_state.pose.position.y), self._monolith_space[0:2]) > 0.50
                for pos in self._boulder_poses:
                    flag = flag and distance.euclidean((goal_state.pose.position.x, goal_state.pose.position.y), (pos[0], pos[1])) > 0.30
                if flag:
                    break
                
            rospy.loginfo("Spawning at (%.2f, %.2f)" % (goal_state.pose.position.x, goal_state.pose.position.y))
            goal_state.pose.position.z = 0.20
                
            if orient:
                goal_state.pose.orientation.z, goal_state.pose.orientation.w = np.random.uniform(-3.14, 3.14, size=(2,))
            GazeboUtils.move_to_position(goal_state)
        except rospy.ROSException:
            rospy.logerr("An error occured while resetting the environment.")
            raise GymException("An error occured while resetting the environment.")
    
    def _move_to_original_position(self, model_name):
        """re-position a model to original position

        Args:
            name: String name of the model.

        Raises:
            GymException: Exception occured while resetting the environment.
        """
        try:
            assert model_name is not None or model_name != ''

            goal_state = ModelState()
            goal_state.model_name = model_name
            goal_state.pose.position.x = 0.0
            goal_state.pose.position.y = 0.0
            goal_state.pose.position.z = 0.20

            GazeboUtils.move_to_position(goal_state)
        except rospy.ServiceException:
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
        """Renders the environment.
        
        Args:
            mode: The type of rendering to use.
                - 'human': Renders state to a graphical window.
        
        Returns:
            None as only human mode is implemented.
        """
        if mode == 'human':
            self.plot(self._current_state)
        elif mode == 'array':
            if self._current_state is not None:
                return self._current_state
            return self._get_state()
        else:
            raise NotImplementedError(mode)
        
    def plot(self, img, id=1, title="State"):
        """Plot an image in a non-blocking way.


        Args:
            img: A numpy array containing the observation.
            id: Numeric id which is assigned to the pyplot figure.
            title: String value which is used as the title.
        """
        if img is not None and isinstance(img, np.ndarray):
            plt.figure(id)     
            plt.ion()
            plt.clf() 
            plt.imshow(img.squeeze())         
            plt.title(title)  
            plt.show(block=False)
            plt.pause(0.05)


class OffWorldMonolithObstacleDiscreteEnv(OffWorldMonolithObstacleEnv):
    """Discrete version of the simulated gym environment that replicates the real OffWorld Monolith environment in Gazebo.      
    
    .. code:: python
    
        env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.DEPTHONLY, random_init=True)
        env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.RGB_ONLY, random_init=True)
        env = gym.make('OffWorldMonolithDiscreteSim-v0', channel_type=Channels.RGBD, random_init=True)
    """

    def __init__(self,
                 channel_type=Channels.DEPTH_ONLY,
                 random_init=True,
                 clip_depth_value=DEFAULT_MAX_DEPTH_VALUE,
                 image_out_size=DEFAULT_IMAGE_OUT_SIZE):
        super(OffWorldMonolithObstacleDiscreteEnv, self).__init__(channel_type=channel_type, random_init=random_init,
                                                                  clip_depth_value=clip_depth_value, image_out_size=image_out_size)
        self.action_space = spaces.Discrete(4)
        
    def _send_action_commands(self, action_type):
        """Sends an action command to the robot.
        
        Args:
            action_type: FourDiscreteMotionActions instance.
        """
        if action_type == FourDiscreteMotionActions.LEFT:
            self._move_rosbot(0.07, 1.25, 4) 
        elif action_type == FourDiscreteMotionActions.RIGHT:
            self._move_rosbot(0.07, -1.25, 4) 
        elif action_type == FourDiscreteMotionActions.FORWARD:
            self._move_rosbot(0.1, 0.0)
        elif action_type == FourDiscreteMotionActions.BACKWARD:
            self._move_rosbot(-0.1, 0.0)

    def step(self, action):
        """Take an action in the environment.

        Args:
            action: An action to be taken in the environment.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
            An integer with reward from the environment.
            A boolean flag which is true when an episode is complete.
            No info given for fair learning.
        """        
        # unpause physics before interaction with the environment
        GazeboUtils.unpause_physics()
        self.step_count += 1

        assert action is not None, "Action cannot be None."

        # convert float if it's exactly an integer value, otherwise let it throw an error
        if isinstance(action, (float, np.float32, np.float64)) and float(action).is_integer():
            action = int(action)

        assert isinstance(action, (FourDiscreteMotionActions, int, np.int32, np.int64)), "Action type is not recognized."

        if isinstance(action, (int, np.int32, np.int64)):
            assert action >= 0 and action < 4, "Unrecognized value for the action"
            action = FourDiscreteMotionActions(action)

        rospy.loginfo("Step: %d" % self.step_count)
        rospy.loginfo(action)
        self._send_action_commands(action)
        
        self._current_state = self._get_state()
        reward, done = self._calculate_reward()
        
        if done:
            self.step_count = 0
            
        # pause physics now, speeds up simulation
        GazeboUtils.pause_physics()
        
        return self._current_state, reward, done, {}


class OffWorldMonolithObstacleContinuousEnv(OffWorldMonolithObstacleEnv):
    """Continous version of the simulated gym environment that replicates the real OffWorld Monolith environment in Gazebo.      
    
    .. code:: python
    
        env = gym.make('OffWorldMonolithContinousSim-v0', channel_type=Channels.DEPTHONLY, random_init=True)
        env = gym.make('OffWorldMonolithContinousSim-v0', channel_type=Channels.RGB_ONLY, random_init=True)
        env = gym.make('OffWorldMonolithContinousSim-v0', channel_type=Channels.RGBD, random_init=True)
    """

    def __init__(self,
                 channel_type=Channels.DEPTH_ONLY,
                 random_init=True,
                 clip_depth_value=DEFAULT_MAX_DEPTH_VALUE,
                 image_out_size=DEFAULT_IMAGE_OUT_SIZE):
        super(OffWorldMonolithObstacleContinuousEnv, self).__init__(channel_type=channel_type, random_init=random_init,
                                                                    clip_depth_value=clip_depth_value, image_out_size=image_out_size)
        self.action_space = spaces.Box(low=np.array([-0.7, -2.5]), high=np.array([0.7, 2.5]), dtype=np.float32)
        self.action_limit = np.array([[-0.7, -2.5], [0.7, 2.5]])

    def step(self, action):
        """Take an action in the environment.

        Args:
            action: An action to be taken in the environment.

        Returns:
            A numpy array with rgb/depth/rgbd encoding of the state observation.
            An integer with reward from the environment.
            A boolean flag which is true when an episode is complete.
            No info given for fair learning.
        """        
        # unpause physics before interaction with the environment
        GazeboUtils.unpause_physics()
        self.step_count += 1

        assert action is not None, "Action cannot be None."
        assert isinstance(action, (np.ndarray)), "Action type is not recognized."
        action = np.clip(action, self.action_limit[0], self.action_limit[1])
        rospy.loginfo("Step: %d" % self.step_count)
        rospy.loginfo(action)
        self._move_rosbot(action[0], action[1], 1.0) 
        
        self._current_state = self._get_state()
        reward, done = self._calculate_reward()
        
        if done:
            self.step_count = 0
            
        # pause physics now, speeds up simulation
        GazeboUtils.pause_physics()
        
        return self._current_state, reward, done, {}
