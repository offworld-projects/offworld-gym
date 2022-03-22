#! /usr/bin/env python3
# license removed for brevity
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