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

import numpy as np
import cv2
import io
from imageio import imread
import base64
import time
import rospy
from std_srvs.srv import Empty as Empty_srv
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState

class GazeboUtils:
    """Gazebo utility functions used by the OffWorld Gym environments
    """
    unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty_srv)
    pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty_srv)

    @staticmethod
    def unpause_physics(): 
        """Unpause physics of the Gazebo simulator
        """       
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            GazeboUtils.unpause()
        except rospy.ServiceException:
            rospy.logerr("/gazebo/pause_physics service call failed")

    @staticmethod
    def pause_physics():
        """Pause physics of the Gazebo simulator
        """
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            GazeboUtils.pause()
        except rospy.ServiceException:
            rospy.logerr("/gazebo/unpause_physics service call failed")
    
    @staticmethod
    def move_to_position(model_state):
        """Set a model's state
        Args:
            model_state: A ModelState object with the model's state details
        Raises:
             Exception: Exception occured during the /gazebo/set_model_state 
             service call
        """
        try:
            assert isinstance(model_state, ModelState)
            rospy.wait_for_service('/gazebo/set_model_state')
            set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_model_state(model_state)
        except rospy.ServiceException as e:
            rospy.logerr('The robot cannot be reset.')
            raise e


