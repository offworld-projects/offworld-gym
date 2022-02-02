# Dockerfile for OffWorld-Gym Remote Env Server.
# Encapsulates ROS to allow for less user installation fuss and parallel Gazebo environments on a single host

FROM ubuntu:20.04

# set the image timezone to be UTC
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Ubuntu system level dependecies
RUN apt-get update -y
RUN apt-get install -y -q gnupg2 apt-utils lsb-core lsb-release software-properties-common dialog 
RUN apt-get clean all 
RUN apt-get install -y -q nano net-tools xvfb tmux glew-utils mesa-utils

# Install ROS and other Python dependencies
ENV DEBIAN_FRONTEND=noninteractive
#RUN ./scripts/install-real-and-sim.sh install_ros_dep_lib
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' 
RUN apt install -y curl 
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
RUN apt update -y
RUN apt install -y ros-noetic-desktop-full git
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN apt install -y python3-rosdep
RUN rosdep init
RUN sudo rosdep fix-permissions
RUN rosdep update
# ENV source /opt/ros/noetic/setup.bash

# install additional ROS packages
RUN apt-get update -y
RUN apt install --allow-unauthenticated -y ros-noetic-multirobot-map-merge ros-noetic-explore-lite \
                    ros-noetic-ros-controllers ros-noetic-rospack \
                    python3-tk libeigen3-dev \
                    ros-noetic-roslint ros-noetic-catkin python3-catkin-tools

###########################################################################
# Install Python3.8
RUN add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y python3.8 python3.8-dev 

# Install Offworld Client Library and Python dependecies
RUN export OFFWORLD_GYM_ROOT=`pwd`/..
RUN apt install -y libbullet-dev git curl wget python3-distutils
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3/bin
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

RUN pip3.8 install --user --upgrade setuptools      
RUN pip3.8 install --user --upgrade pip
RUN pip3.8 install --user --upgrade numpy
RUN pip3.8 install --user --upgrade scipy
RUN pip3.8 install --user --upgrade opencv-python
RUN pip3.8 install --user --upgrade catkin_pkg
RUN pip3.8 install --user --upgrade empy
RUN pip3.8 install --user --upgrade requests
RUN pip3.8 install --user --upgrade defusedxml 
RUN pip3.8 install --user --upgrade rospkg
RUN pip3.8 install --user --upgrade matplotlib
RUN pip3.8 install --user --upgrade netifaces
RUN pip3.8 install --user --upgrade regex
RUN pip3.8 install --user --upgrade psutil
RUN pip3.8 install --user --upgrade gym
RUN pip3.8 install --user --upgrade python-socketio
RUN pip3.8 install --user --upgrade scikit-image
RUN pip3.8 install --user --upgrade pyquaternion
RUN pip3.8 install --user --upgrade imageio
RUN pip3.8 install --user docker

# Copy and Initialize workspace
RUN mkdir -p /offworld-gym
ENV  OFFWORLD_GYM_ROOT='/offworld-gym'
COPY . /offworld-gym
WORKDIR /offworld-gym

WORKDIR $OFFWORLD_GYM_ROOT
RUN pip3.8 install  -e .
RUN chmod +x ./offworld_gym/envs/gazebo_docker/docker_entrypoint.sh

# RUN ./scripts/install-real-and-sim.sh install_python_dep_lib
WORKDIR /usr/lib/x86_64-linux-gnu
RUN ln -s libboost_python37.a libboost_python3.a
RUN ln -s libboost_python37.so libboost_python3.so
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
RUN apt-get update
RUN apt-get install -y libignition-math4-dev  
RUN mkdir -p $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
WORKDIR $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src

RUN git clone https://github.com/ros-perception/vision_opencv.git -b noetic
RUN git clone https://github.com/offworld-projects/rosbot_description.git -b offworld-gym

WORKDIR $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws
RUN pwd
RUN ls -lh
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
# RUN ./scripts/install-real-and-sim.sh build_gym_shell_script

###########################################################################
## Gazebo and nvm
ENV NVM_DIR /usr/local/nvm
RUN mkdir -p $NVM_DIR
ENV NODE_VERSION 11

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get install -y libjansson-dev npm libboost-dev imagemagick libtinyxml-dev mercurial cmake build-essential 

# Install nvm with node and npm
RUN curl https://raw.githubusercontent.com/creationix/nvm/v0.39.1/install.sh | bash \
    && source $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default \
    && cd / \
    && git clone https://github.com/osrf/gzweb \
    && cd gzweb \
    && git checkout gzweb_1.4.1 \
    && source /usr/share/gazebo/setup.sh\
    && mkdir -p /gzweb/http/client/assets/ \
    && npm run deploy --- -m local

ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Additional ENV Variables to separete elements of PYTHONPATH dependencies among the specific systems that require them.
# (May have unnecessary elements in these paths, e.g. catkin_ws packages)

# Appended to the python system path at runtime to import ROS python modules regardless of existing env setup.
ENV GAZEBO_GYM_PYTHON_DEPENDENCIES /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3.8/dist-packages

# Replaces PYTHONPATH in the subprocess that calls roslaunch within a Gazebo Gym (So that your python3 site-packages don't get imported by accident)
ENV ROSLAUNCH_PYTHONPATH_OVERRIDE /opt/ros/noetic/lib/python3.8/dist-packages:/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages

# 2nd line below is a hack. Something in the original astra_minimal.dae causes rendering to crash in chrome, but astra.dae works.
RUN cp -r /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/rosbot_description/src/rosbot_description /gzweb/http/client/assets/ \
    && cp /gzweb/http/client/assets/rosbot_description/meshes/astra.dae /gzweb/http/client/assets/rosbot_description/meshes/astra_minimal.dae

# build the Gym Shell script
RUN echo '#!/usr/bin/env bash' > $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN echo "source $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/devel/setup.bash" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN echo "source /opt/ros/noetic/setup.bash --extend" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN echo "export GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN echo "export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN echo 'export PYTHONPATH= ~/.local/lib/python3.8/site-packages/:$PYTHONPATH' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN echo 'export OFFWORLD_GYM_ACCESS_TOKEN="COPY IT HERE"' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
RUN chmod +x $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
