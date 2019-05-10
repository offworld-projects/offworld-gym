#!/usr/bin/env bash

# Install python3.6
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt update
sudo apt-get install python3.6
curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6

# Create virtual env
pip3.6 install virtualenv
virtualenv -p `python3.6` ~/ve/py36
echo "source ~/ve/py36/bin/activate" >> ~/.bashrc
source ~/.bashrc

# Python packages
pip install numpy
pip install tensorflow-gpu
pip install keras

# python ros 
pip install catkin_pkg
pip install empy
pip install defusedxml

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6-dev
pip install regex
pip install psutil

# ros workspac
mkdir -p ~/catkin_ws/src
cd ~/catkin_src
git clone https://github.com/ros/xacro.git -b kinetic-devel
git clone https://github.com/ros/ros.git -b kinetic-devel
git clone https://github.com/ros/catkin.git -b kinetic-devel
git clone https://github.com/ros/ros_comm_msgs.git -b indigo-devel
git clone https://github.com/ros/gencpp.git -b indigo-devel
git clone https://github.com/jsk-ros-pkg/geneus.git -b master
git clone https://github.com/ros/genlisp.git -b groovy-devel
git clone https://github.com/ros/genmsg.git -b indigo-devel
git clone https://github.com/ros/genpy.git -b kinetic-devel
git clone https://github.com/RethinkRobotics-opensource/gennodejs.git -b kinetic-devel
git clone https://github.com/ros/std_msgs.git -b groovy-devel
git clone https://github.com/ros/geometry.git -b indigo-devel
git clone https://github.com/ros/geometry2.git -b indigo-devel
git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b kinetic-devel
git clone https://github.com/ros-controls/ros_control.git -b kinetic-devel