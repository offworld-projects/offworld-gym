#!/usr/bin/env bash

echo "export GAZEBO_MODEL_PATH=`pwd`/../models:"'$GAZEBO_MODEL_PATH' >> ~/.bashrc

cd ../..
git clone https://github.com/husarion/rosbot_description.git -b devel
sudo apt install ros-kinetic-grid-map ros-kinetic-frontier-exploration ros-kinetic-ros-controllers -y

pip install defusedxml
pip install netifaces

cd ..
catkin_make
echo "source `pwd`/devel/setup.bash" >> ~/.bashrc

source ~/.bashrc
exec bash 
