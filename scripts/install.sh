#!/usr/bin/env bash

# this assumes the script is executed from under `scripts` directory
# (should we move `install.sh` to the repository root? will be cleaner)
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

# make sure we have Python 3.5
sudo apt install -y python3.5 python3.5-dev

# create virtual environment
sudo apt install -y virtualenv
mkdir ~/ve
virtualenv -p python3.5 ~/ve/py35gym
source ~/ve/py35gym/bin/activate
pip install --upgrade pip
source /opt/ros/kinetic/setup.bash

echo "Virtual environment set up done."
# (we will drop this)
#echo "source ~/ve/py36/bin/activate" >> ~/.bashrc
#source ~/.bashrc

# intall Python packages
pip install numpy
pip install tensorflow-gpu
pip install keras==2.2.4
pip install opencv-python
pip install catkin_pkg
pip install empy
pip install requests
pip install defusedxml
pip install rospkg
pip install matplotlib
pip install netifaces
pip install regex
pip install psutil
pip install gym
pip install python-socketio
pip install scikit-image
cd $OFFWORLD_GYM_ROOT
pip install -e .

# install customized version of keras-rl
mkdir $OFFWORLD_GYM_ROOT/assets
cd $OFFWORLD_GYM_ROOT/assets
git clone https://github.com/offworld-projects/keras-rl.git -b offworld-gym
cd keras-rl
pip install -e .

# install additional ROS packages
sudo apt install -y ros-kinetic-grid-map ros-kinetic-frontier-exploration \
                    ros-kinetic-ros-controllers ros-kinetic-rospack \
                    libignition-math2 libignition-math2-dev python3-tk libeigen3-dev \
                    ros-kinetic-roslint ros-kinetic-tf2-bullet


# Milestone 1: Python and system packages
if [ $? -eq 0 ]
then
  printf "\nOK: Python and system packages were installed successfully.\n\n"
else
  printf "\nFAIL: Errors detected installing system packages, please resolve them and restart the installation script.\n\n" >&2
  exit 1
fi

# build Python 3.5 version of catkin *without* installing it system-wide
cd $OFFWORLD_GYM_ROOT/assets
echo "Building catkin here: `pwd`."
git clone https://github.com/ros/catkin.git -b kinetic-devel
cd $OFFWORLD_GYM_ROOT/assets/catkin
mkdir build && cd build && cmake .. && make
echo "Catkin build for Python 3.5 complete."

# prepare for building the workspace
cd /usr/lib/x86_64-linux-gnu
sudo ln -s libboost_python-py35.so libboost_python3.so

# build ROS workspace
cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
$OFFWORLD_GYM_ROOT/assets/catkin/bin/catkin_init_workspace

git clone https://github.com/ros/xacro.git -b kinetic-devel
git clone https://github.com/ros/ros.git -b kinetic-devel
git clone https://github.com/ros/ros_comm.git -b kinetic-devel
git clone https://github.com/ros/common_msgs.git -b indigo-devel
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
git clone https://github.com/offworld-projects/offworld_rosbot_description.git -b kinetic-devel
git clone https://github.com/ros-perception/vision_opencv.git -b kinetic

cd ..
$OFFWORLD_GYM_ROOT/assets/catkin/bin/catkin_make -j1

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
echo "source ~/ve/py35gym/bin/activate" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "unset PYTHONPATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source /opt/ros/kinetic/setup.bash" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash --extend" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo 'export PYTHONPATH=~/ve/py35gym/lib/python3.5/site-packages:$PYTHONPATH' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
chmod +x $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

# update to gazebo 7.13
# http://answers.gazebosim.org/question/18934/kinect-in-gazebo-not-publishing-topics/
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
sudo apt install wget
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gazebo7 libgazebo7-dev

printf "\n\nInstallation complete\n---------------------\n\n"
printf "To setup a shell for OffWorld Gym run\n\n\tsource $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\nin each new terminal to activate Gym Shell.\n"
printf "Or add to your ~/.bashrc by running\n\n\techo \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\" >> ~/.bashrc\n\n---------------------\n\n"
printf "To test Real environment:\n\t(add instructions here)\n\n"
printf "To test Sim environment: open two terminals, activate Gym Shell, and run:\n\t1. roslaunch gym_offworld_monolith env_bringup.launch\n\t2. gzclient\n\n"
