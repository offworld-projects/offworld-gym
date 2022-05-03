#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
from offworld_gym.envs.gazebo.gazebo_utils import GazeboUtils
import rospy
from geometry_msgs.msg import Twist
import logging
import subprocess
from sys import argv

logger = logging.getLogger(__name__)
level = logging.INFO
logger.setLevel(level)

# Starts a new node when open the HTTP server
rospy.init_node('move_robot', anonymous=True)
velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


def move_robot(lin_x_speed, ang_z_speed):
    # Create a new cmd_vel message
    vel_msg = Twist()
    vel_msg.linear.x = float(lin_x_speed)
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0.0
    vel_msg.angular.x = 0.0
    vel_msg.angular.y = 0.0
    vel_msg.angular.z = float(ang_z_speed)
    # wait till the connection is available, then pub once
    while not rospy.is_shutdown() and velocity_publisher.get_num_connections() < 1:
        pass
    velocity_publisher.publish(vel_msg)


class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_HEAD(self):
        self._set_headers()

    # GET sends back a Hello world message
    def do_GET(self):
        self._set_headers()
        self.wfile.write(json.dumps({'hello': 'world', 'received': 'ok'}).encode())

    # POST echoes the message adding a JSON field
    def do_POST(self):
        logger.debug("do post")
        # ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        ctype, pdict = cgi.parse_header(self.headers.get_content_type())

        # refuse to receive non-json content
        if ctype != 'application/json':
            self.send_response(100)
            self.end_headers()
            return

        # read the message and convert it into a python dictionary
        length = int(self.headers.get('content-length'))
        message = json.loads(self.rfile.read(length))
        logger.debug(" json message {} ".format(json.dumps(message)))

        # run command
        command_name = message['command_name']

        if command_name == "launch_node":
            package_name = message['package_name']
            launch_file_name = message['launch_file_name']
            ros_port = message["ros_port"]

            ros_launch_cmd = f'roslaunch {package_name} {launch_file_name} ros_port:={ros_port}'
            subprocess.Popen(["bash", "-c", ros_launch_cmd])
            logger.debug("launch node {} {}".format(package_name, launch_file_name))
            # add a property to the object, just to mess with data
            message = {'success': 'launch node ok'}

        elif command_name == "pause":
            GazeboUtils.pause_physics()
            # add a property to the object, just to mess with data
            message = {'success': 'pause sim ok'}

        elif command_name == "unpause":
            GazeboUtils.unpause_physics()
            # add a property to the object, just to mess with data
            message = {'success': 'unpause sim ok'}

        elif command_name == "cmd_vel":
            # add a property to the object, just to mess with data
            lin_x_speed = message['lin_x_speed']
            lin_y_speed = message['lin_y_speed']
            lin_z_speed = message['lin_z_speed']
            ang_x_speed = message['ang_x_speed']
            ang_y_speed = message['ang_y_speed']
            ang_z_speed = message['ang_z_speed']

            move_robot(lin_x_speed, ang_z_speed)

            message = {}
            message['success'] = 'cmd_vel publish ok'
        
        # send the message back
        self._set_headers()
        self.wfile.write(json.dumps(message).encode())


def run(server_class=HTTPServer, handler_class=Server, port=8008):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()

    print("The command server is running and accepting commands.")
