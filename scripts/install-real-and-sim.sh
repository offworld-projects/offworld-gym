#!/usr/bin/env bash

# Copyright 2019 OffWorld Inc.
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


# this assumes the script is executed from under `scripts` directory
# (should we move `install.sh` to the repository root? will be cleaner)
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

sudo apt update
sudo apt install -y libbullet-dev python-pip git curl wget

curl https://bootstrap.pypa.io/pip/2.7/get-pip.py | sudo -H python2.7
pip install --user --upgrade pip 
pip install --user --upgrade testresources
pip install --user --upgrade setuptools
pip install --user --upgrade numpy==1.16.5
pip install --user --upgrade scipy==1.2.2
pip install --user --upgrade futures==3.1.1
pip install --user --upgrade tensorflow-gpu==1.14.0
pip install --user --upgrade keras==2.2.4
# pip install --user --upgrade opencv-python
pip install --user --upgrade catkin_pkg
pip install --user --upgrade empy
pip install --user --upgrade requests
pip install --user --upgrade defusedxml
pip install --user --upgrade rospkg
pip install --user --upgrade matplotlib
pip install --user --upgrade netifaces
# pip install --user --upgrade regex
pip install --user --upgrade psutil
pip install --user --upgrade gym
pip install --user --upgrade python-socketio
pip install --user --upgrade scikit-image
pip install --user --upgrade pyquaternion
pip install --user --upgrade imageio
cd $OFFWORLD_GYM_ROOT
pip install --user -e .

# Installing Python3.6
# cd /tmp
# wget http://10.0.3.12:20500/offworld/common-libraries/python-3.6/Python-3.6.3.tgz
# tar -xvf Python-3.6.3.tgz
# cd /tmp/Python-3.6.3
# ./configure --enable-shared
# make
# make install 

# sudo add-apt-repository ppa:jonathonf/python-3.6
# sudo apt update
# sudo apt install python3.6
# sudo apt install python3.6-dev
# sudo apt install python3.6-venv
# wget https://bootstrap.pypa.io/get-pip.py
# sudo python3.6 get-pip.py
# sudo ln -s /usr/bin/python3.6 /usr/local/bin/python3
# sudo ln -s /usr/local/bin/pip /usr/local/bin/pip3

# pip3.6 install --user --upgrade setuptools      
# pip3.6 install --user --upgrade pip
# pip3.6 install --user --upgrade numpy
# pip3.6 install --user --upgrade scipy
# pip3.6 install --user --upgrade tensorflow-gpu==1.14.0
# pip3.6 install --user --upgrade keras==2.2.4
# pip3.6 install --user --upgrade opencv-python
# pip3.6 install --user --upgrade catkin_pkg
# pip3.6 install --user --upgrade empy
# pip3.6 install --user --upgrade requests
# pip3.6 install --user --upgrade defusedxml 
# pip3.6 install --user --upgrade rospkg
# pip3.6 install --user --upgrade matplotlib
# pip3.6 install --user --upgrade netifaces
# pip3.6 install --user --upgrade regex
# pip3.6 install --user --upgrade psutil
# pip3.6 install --user --upgrade gym
# pip3.6 install --user --upgrade python-socketio
# pip3.6 install --user --upgrade scikit-image
# pip3.6 install --user --upgrade pyquaternion
# pip3.6 install --user --upgrade imageio

# cd $OFFWORLD_GYM_ROOT
# pip3.6 install --user -e .
		
# # put modified keras-rl under ./assets
# # gymshell.sh will update the $PYTHONPATH to point here
# mkdir -p $OFFWORLD_GYM_ROOT/assets
# cd $OFFWORLD_GYM_ROOT/assets
# git clone https://github.com/offworld-projects/keras-rl.git -b offworld-gym

# source /opt/ros/kinetic/setup.bash


# # install additional ROS packages
# sudo apt install --allow-unauthenticated -y ros-kinetic-grid-map ros-kinetic-frontier-exploration \
#                     ros-kinetic-ros-controllers ros-kinetic-rospack \
#                     libignition-math2 libignition-math2-dev python3-tk libeigen3-dev \
#                     ros-kinetic-roslint


# # Milestone 1: Python and system packages
# if [ $? -eq 0 ]
# then
#   printf "\nOK: Python and system packages were installed successfully.\n\n"
# else
#   printf "\nFAIL: Errors detected installing system packages, please resolve them and restart the installation script.\n\n" >&2
#   exit 1
# fi

# cd /usr/lib/x86_64-linux-gnu
# sudo ln -s libboost_python-py35.so libboost_python3.so
# sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
# wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install -y libignition-math4-dev
# cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src

# git clone https://github.com/ros-perception/vision_opencv.git -b kinetic
# git clone https://github.com/offworld-projects/rosbot_description.git -b offworld-gym

# cd ..
# catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.6 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so

# # Milestone 1: Python and system packages
# if [ $? -eq 0 ]
# then
#   printf "\nOK: ROS workspace built successfully\n\n"
# else
#   printf "\nFAIL: Errors detected while building ROS workspace. Please resolve the issues and finish installation manually, line-by-line, do no restart this script.\n\n" >&2
#   exit 1
# fi

# echo "ROS dependencies build complete."

# # build the Gym Shell script
# echo '#!/usr/bin/env bash' > $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# echo "source /opt/ros/kinetic/setup.bash --extend" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# echo "export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# echo 'export PYTHONPATH=$OFFWORLD_GYM_ROOT/assets/keras-rl:~/.local/lib/python3.6/site-packages/:$PYTHONPATH' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# echo 'export OFFWORLD_GYM_ACCESS_TOKEN="COPY IT HERE"' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
# chmod +x $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

# # update to gazebo 7.13
# # http://answers.gazebosim.org/question/18934/kinect-in-gazebo-not-publishing-topics/
# sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
# wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install -y gazebo7 libgazebo7-dev

# printf "\n\nInstallation complete\n---------------------\n\n"
# printf "To setup a shell for OffWorld Gym run\n\n\tsource $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\nin each new terminal to activate Gym Shell.\n"
# printf "Or add to your ~/.bashrc by running\n\n\techo \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\" >> ~/.bashrc\n\n---------------------\n\n"
# printf "To test the Real environment:\n\n\t* Book your time slot at https://gym.offworld.ai/book\n\t* Copy the access token from https://gym.offworld.ai/account to $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\t* Activate Gym Shell by calling \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\"\n\t* Open https://gym.offworld.ai/cameras in your browser to see the real-time camera stream\n\t* Run the minimal example \"python3.6 examples/minimal_example_real.py\"\n\n"
# printf "To test Sim environment: open two terminals, activate Gym Shell in each one, and run:\n\n\tTerminal 1: roslaunch gym_offworld_monolith env_bringup.launch\n\tTerminal 2: gzclient\n\n"
