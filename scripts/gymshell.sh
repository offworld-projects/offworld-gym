#!/usr/bin/env bash
source ~/ve/py35gym/bin/activate
unset PYTHONPATH
source /opt/ros/kinetic/setup.bash
source /home/ilya/offworld/Code/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend
export GAZEBO_MODEL_PATH=/home/ilya/offworld/Code/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:
export PYTHONPATH=~/ve/py35gym/lib/python3.5/site-packages:$PYTHONPATH


