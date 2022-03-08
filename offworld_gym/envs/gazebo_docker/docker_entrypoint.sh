#!/bin/bash

Xvfb :1 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:1.0

export OFFWORLD_GYM_ROOT='/offworld-gym'
source /opt/ros/noetic/setup.bash
source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend


# start python server
# python3 /offworld-gym/offworld_gym/envs/gazebo_docker/command_server.py 
# start node.js and gazebo
# npm start --prefix /gzweb &

while true
do
   sleep 2;
done