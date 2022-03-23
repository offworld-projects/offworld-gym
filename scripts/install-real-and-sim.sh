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

export OFFWORLD_GYM_ROOT='/offworld-gym'

install_ros_dep_lib() {
    # Install Ubuntu system level dependecies
    apt-get update -y
    apt-get install -y -q gnupg2 apt-utils lsb-core lsb-release software-properties-common dialog 
    apt-get clean all 
    apt-get install -y -q nano net-tools xvfb glew-utils mesa-utils

    # Install ROS and other Python dependencies
    export DEBIAN_FRONTEND=noninteractive
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' 
    apt install -y curl 
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
    apt update -y
    apt  --fix-broken install -y
    apt install -y ros-noetic-desktop-full git
    apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
    rosdep init
    rosdep fix-permissions
    rosdep update
    source /opt/ros/noetic/setup.bash

    # install additional ROS packages
    apt-get update -y
    apt install --allow-unauthenticated -y ros-noetic-rosbridge-suite ros-noetic-multirobot-map-merge ros-noetic-explore-lite \
                        ros-noetic-ros-controllers ros-noetic-rospack \
                        python3-tk libeigen3-dev \
                        ros-noetic-roslint ros-noetic-catkin python3-catkin-tools  ros-noetic-robot-state-publisher

}

install_python_dep_lib() {
    # Install Python3.8
    add-apt-repository ppa:deadsnakes/ppa \
        && apt-get update \
        && apt-get install -y python3.8 python3.8-dev 

    # Install Offworld Client Library and Python dependecies
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3/bin
    export PATH=$PATH:/root/.local/bin
    apt install -y libbullet-dev git curl wget python3-distutils
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3.8 get-pip.py
    python3.8 -m pip install --upgrade pip

    pip3.8 install --user --upgrade testresources==2.0.1
    pip3.8 install --user --upgrade setuptools==60.9.3
    pip3.8 install --user --upgrade numpy==1.22.3
    pip3.8 install --user --upgrade scipy==1.8.0
    pip3.8 install --user --upgrade opencv-python==4.5.5.64
    pip3.8 install --user --upgrade catkin_pkg==0.4.24
    pip3.8 install --user --upgrade empy==3.3.4
    pip3.8 install --user --upgrade requests==2.27.1
    pip3.8 install --user --upgrade defusedxml==0.7.1 
    pip3.8 install --user --upgrade rospkg==1.4.0
    pip3.8 install --user --upgrade matplotlib==3.5.1
    pip3.8 install --user --upgrade netifaces==0.11.0
    pip3.8 install --user --upgrade regex==2022.3.2
    pip3.8 install --user --upgrade psutil==5.9.0
    pip3.8 install --user --upgrade gym==0.23.0
    pip3.8 install --user --upgrade python-socketio==5.5.2
    pip3.8 install --user --upgrade scikit-image==0.19.2
    pip3.8 install --user --upgrade pyquaternion==0.9.5
    pip3.8 install --user --upgrade imageio==2.16.1
    pip3.8 install --user --upgrade importlib-metadata==4.11.2
    }

install_gazebo_nvm_dep_lib() {
    # Gazebo and nvm
    # Replace shell with bash so we can source files
    sudo rm /bin/sh && ln -s /bin/bash /bin/sh

    curl -sSL http://get.gazebosim.org | sh
    apt-get install -y libjansson-dev npm libboost-dev imagemagick libtinyxml-dev mercurial cmake build-essential 
    apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control

    export NVM_DIR=/usr/local/nvm
    mkdir -p $NVM_DIR
    export NODE_VERSION=9.11.2

    # Install nvm 
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash \
        && source $NVM_DIR/nvm.sh \
        && nvm install $NODE_VERSION \
        && nvm alias default $NODE_VERSION \
        && nvm use default 

    export NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
    export PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

    # # Copy and Initialize workspace
    # mkdir -p /offworld-gym
    # export OFFWORLD_GYM_ROOT='/offworld-gym'
    # cp . /offworld-gym

    # Install gzweb 
    cd /
    git clone https://github.com/offworld-projects/offworld-gzweb.git -b fix-camera-angle/gzweb_1.4.1
    mv offworld-gzweb gzweb
    cd /gzweb 
    source /usr/share/gazebo/setup.sh 
    mkdir -p /gzweb/http/client/assets/ 
    npm run deploy --- -m local 
    npm run update

    cd $OFFWORLD_GYM_ROOT
    pip3.8 install --user -e .

    mkdir -p $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
    cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
    git clone https://github.com/offworld-projects/rosbot_description.git -b offworld-gym
    chmod +x $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/scripts/command_server.py
    chmod +x $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/scripts/robot_commands.py

    cd $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws
    sudo /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
}


build_gym_shell_script() {
    # Appended to the python system path at runtime to import ROS python modules regardless of existing env setup.
    export GAZEBO_GYM_PYTHON_DEPENDENCIES=/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3.8/dist-packages

    # Replaces PYTHONPATH in the subprocess that calls roslaunch within a Gazebo Gym (So that your python3 site-packages don't get imported by accident)
    export ROSLAUNCH_PYTHONPATH_OVERRIDE=/opt/ros/noetic/lib/python3.8/dist-packages:/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages

    # 2nd line below is a hack. Something in the original astra_minimal.dae causes rendering to crash in chrome, but astra.dae works.
    cp -r /usr/share/gazebo-11/media /gzweb/http/client/assets/ 
    cp -r /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models/* /gzweb/http/client/assets/ 
    cp -r /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/rosbot_description/src/rosbot_description /gzweb/http/client/assets/ 
    cp /gzweb/http/client/assets/rosbot_description/meshes/astra.dae /gzweb/http/client/assets/rosbot_description/meshes/astra_minimal.dae


    # Add to .bashrc for root user
    export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH
    export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT
    export PYTHONPATH=/usr/local/lib/python3.8/dist-packages/:/root/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages/:/usr/lib/python3/dist-packages:$PYTHONPATH
    export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH
    chmod +x $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/docker_entrypoint.sh
    cd $OFFWORLD_GYM_ROOT
}

"$@"