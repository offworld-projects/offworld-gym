#!/usr/bin/env bash

# this assumes the script is executed from under `scripts` directory
# (should we move `install.sh` to the repository root? will be cleaner)
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

# make sure we have Python 3.5
sudo apt install python3.5 python3.5-dev

# create virtual environment
sudo apt install virtualenv
mkdir ~/ve
virtualenv -p python3.5 ~/ve/py35gym
source ~/ve/py35gym/bin/activate
pip install --upgrade pip

echo "Virtual environment set up done."
# (we will drop this)
#echo "source ~/ve/py36/bin/activate" >> ~/.bashrc
#source ~/.bashrc

# intall Python packages
pip install numpy
pip install tensorflow-gpu
pip install keras
pip install opencv-python
pip install catkin_pkg
pip install empy
pip install defusedxml
pip install rospkg
pip install matplotlib
pip install netifaces
pip install regex
pip install psutil
echo "Python packages installed."

# install additional ROS packages
sudo apt install ros-kinetic-grid-map ros-kinetic-frontier-exploration ros-kinetic-ros-controllers -y

# build Python 3.5 version of catkin *without* installing it system-wide
mkdir $OFFWORLD_GYM_ROOT/assets
cd $OFFWORLD_GYM_ROOT/assets
echo "Building catkin here: `pwd`."
git clone https://github.com/ros/catkin.git -b kinetic-devel
cd $OFFWORLD_GYM_ROOT/assets/catkin
mkdir build && cd build && cmake .. && make
echo "Catkin build for Python 3.5 complete."

# build ROS workspace
cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
$OFFWORLD_GYM_ROOT/assets/catkin/bin/catkin_init_workspace

git clone https://github.com/ros/xacro.git -b kinetic-devel
git clone https://github.com/ros/ros.git -b kinetic-devel
git clone https://github.com/ros/ros_comm.git -b kinetic-devel
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
git clone https://github.com/ros/dynamic_reconfigure.git -b master
git clone https://github.com/husarion/rosbot_description.git -b devel

cd ..
$OFFWORLD_GYM_ROOT/assets/catkin/bin/catkin_make -j1

echo "ROS dependencies build complete."

# integrate the new environment into the system
echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend" >> ~/.bashrc
echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> ~/.bashrc
export 
unset PYTHONPATH
source ~/.bashrc

echo "Installation complete!"
echo "To test it run 'roslaunch gym_offworld_monolith env_bringup.launch'."
