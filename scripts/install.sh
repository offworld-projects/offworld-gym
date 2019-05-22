#!/usr/bin/env bash

# Install python3.6
cd /opt
sudo wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz
sudo tar -xvf Python-3.6.3.tgz
cd Python-3.6.3
sudo ./configure
sudo make 
sudo make install 

# Create virtual env
sudo pip3.6 install virtualenv
virtualenv -p python3.6 ~/ve/py36
echo "source ~/ve/py36/bin/activate" >> ~/.bashrc
source ~/.bashrc

# Python packages
pip install numpy
pip install tensorflow-gpu
pip install keras
pip install opencv-python

# python ros 
pip install catkin_pkg
pip install empy
pip install defusedxml
pip install rospkg
pip install matplotlib
pip install netifaces

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

./install_environment.sh
