#!/usr/bin/env bash

# this assumes the script is executed from under `scripts` directory
# (should we move `install.sh` to the repository root? will be cleaner)
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

sudo apt update
sudo apt install -y libbullet-dev python-pip git curl wget

pip install --user --upgrade pip
pip install --user --upgrade setuptools
pip install --user numpy==1.16.5
pip install --user scipy==1.2.2
pip install --user tensorflow-gpu==1.14.0
pip install --user keras==2.2.4
pip install --user opencv-python
pip install --user catkin_pkg
pip install --user empy
pip install --user requests
pip install --user defusedxml
pip install --user rospkg
pip install --user matplotlib
pip install --user netifaces
pip install --user regex
pip install --user psutil
pip install --user gym
pip install --user python-socketio
pip install --user scikit-image
pip install --user pyquaternion
cd $OFFWORLD_GYM_ROOT
pip install --user -e .

# Python3.6
sudo add-apt-repository ppa:jonathonf/python-3.6 -y
sudo apt-get update
sudo apt-get install -y python3.6 python3.6-dev
curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6

pip3.6 install --upgrade --user setuptools
pip3.6 install --user numpy
pip3.6 install --user scipy
pip3.6 install --user tensorflow==1.14.0
pip3.6 install --user keras==2.2.4
pip3.6 install --user opencv-python
pip3.6 install --user catkin_pkg
pip3.6 install --user empy
pip3.6 install --user requests
pip3.6 install --user defusedxml 
pip3.6 install --user matplotlib
pip3.6 install --user netifaces
pip3.6 install --user regex
pip3.6 install --user psutil
pip3.6 install --user gym
pip3.6 install --user python-socketio
pip3.6 install --user scikit-image
pip3.6 install --user pyquaternion

cd $OFFWORLD_GYM_ROOT
pip3.6 install --user -e .
		
source /opt/ros/kinetic/setup.bash


# install additional ROS packages
sudo apt install --allow-unauthenticated -y ros-kinetic-grid-map ros-kinetic-frontier-exploration \
                    ros-kinetic-ros-controllers ros-kinetic-rospack \
                    libignition-math2 libignition-math2-dev python3-tk libeigen3-dev \
                    ros-kinetic-roslint


# Milestone 1: Python and system packages
if [ $? -eq 0 ]
then
  printf "\nOK: Python and system packages were installed successfully.\n\n"
else
  printf "\nFAIL: Errors detected installing system packages, please resolve them and restart the installation script.\n\n" >&2
  exit 1
fi

cd /usr/lib/x86_64-linux-gnu
sudo ln -s libboost_python-py35.so libboost_python3.so
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y libignition-math4-dev
sudo rm -f /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so
cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src

#git clone https://github.com/ros/geometry2.git -b indigo-devel
#git clone https://github.com/ros-simulation/gazebo_ros_pkgs.git -b kinetic-devel
git clone https://github.com/ros-perception/vision_opencv.git -b kinetic

cd ..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.6 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so

# Milestone 1: Python and system packages
if [ $? -eq 0 ]
then
  printf "\nOK: ROS workspace built successfully\n\n"
else
  printf "\nFAIL: Errors detected while building ROS workspace. Please resolve the issues and finish installation manually, line-by-line, do no restart this script.\n\n" >&2
  exit 1
fi

echo "ROS dependencies build complete."

# build the Gym Shell script
echo '#!/usr/bin/env bash' > $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source /opt/ros/kinetic/setup.bash" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo 'export PYTHONPATH=$OFFWORLD_GYM_ROOT/assets/keras-rl:$PYTHONPATH' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
chmod +x $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

# update to gazebo 7.13
# http://answers.gazebosim.org/question/18934/kinect-in-gazebo-not-publishing-topics/
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gazebo7 libgazebo7-dev

printf "\n\nInstallation complete\n---------------------\n\n"
printf "To setup a shell for OffWorld Gym run\n\n\tsource $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\nin each new terminal to activate Gym Shell.\n"
printf "Or add to your ~/.bashrc by running\n\n\techo \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\" >> ~/.bashrc\n\n---------------------\n\n"
printf "To test Real environment:\n\t(add instructions here)\n\n"
printf "To test Sim environment: open two terminals, activate Gym Shell in each one, and run:\n\t1. roslaunch gym_offworld_monolith env_bringup.launch\n\t2. gzclient\n\n"
