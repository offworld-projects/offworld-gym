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

install_ros_dep_lib() {
    # sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' 
    # apt install -y curl 
    # curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
    apt update -y
    # apt install -y ros-noetic-desktop-full git
    # apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
    # apt install -y python3-rosdep
    # rosdep init
    # sudo rosdep fix-permissions
    # rosdep update
}

install_gazebo_dep_lib() {
    apt update -y

    # apt install -y gazebo9 libgazebo9-dev
    # apt install -y libjansson-dev libboost-dev imagemagick libtinyxml-dev mercurial cmake build-essential

    # # install nvm
    # curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
    # nvm install 8
    # cd /; git clone https://github.com/osrf/gzweb
    # cd /gzweb
    # git checkout gzweb_1.4.1 
}

install_python_dep_lib() {
    apt-get update -y
}

"$@"