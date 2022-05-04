#! /usr/bin/env python3

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
import rospy
from geometry_msgs.msg import Twist
from gym_offworld_monolith.msg import EnvTwist

def talker():
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.init_node('velocity_middleware', anonymous=True)
    
    def _vel_cb(msg):
        vel_pub.publish(msg.twist)
    
    rospy.Subscriber("/cmd_vel_env", EnvTwist, _vel_cb)
    rospy.spin()

if __name__ == '__main__':

    try:
        talker()
    except rospy.ROSInterruptException:
        pass