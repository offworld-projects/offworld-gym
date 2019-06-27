import rospy
import time
import cv2
import sys,tty,termios

# start ros node
rospy.init_node('test_tele_op')
from geometry_msgs.msg import Twist
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def move_rosbot(lin_vel, ang_vel, sleep_time=2):
    """Moves the ROSBot 

    Accepts linear x speed and angular z speed and moves the
    ROSBot by issuing the velocity commands to the ROSBot.
    """
    vel_cmd = Twist()
    vel_cmd.linear.x = lin_vel
    vel_cmd.angular.z = ang_vel
    vel_pub.publish(vel_cmd)

    time.sleep(sleep_time)
    vel_cmd = Twist()
    vel_cmd.angular.z = 0.0
    vel_cmd.linear.x = 0.0
    vel_pub.publish(vel_cmd) 

def stop():
    move_rosbot(0.0, 0.0)

def forward():
    move_rosbot(0.21, 0.0)

def backward():
    move_rosbot(-0.21, 0.0)

def left():
    move_rosbot(0.09, 1.5, 1)

def right():
    move_rosbot(0.09, -1.5, 1)

def capture_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(3)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def perform_teleop():
    key = None
    while True:
        while key != '': key = capture_key()
        if key =='\x1b[A': forward()
        elif key =='\x1b[B': backward()
        elif key =='\x1b[C': right()
        elif key =='\x1b[D': left()
        else: quit()

if __name__ == "__main__":
    perform_teleop()
