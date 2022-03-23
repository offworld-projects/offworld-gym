# Dockerfile for OffWorld-Gym Remote Env Server.
# Encapsulates ROS to allow for less user installation fuss and parallel Gazebo environments on a single host

# FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
FROM ubuntu:20.04

# set the image timezone to be UTC
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Ubuntu system level dependecies
RUN apt-get update -y
RUN apt-get install -y -q gnupg2 apt-utils lsb-core lsb-release software-properties-common dialog 
RUN apt-get clean all 
RUN apt-get install -y -q nano net-tools xvfb glew-utils mesa-utils

# Install ROS and other Python dependencies
# RUN $OFFWORLD_GYM_ROOT/scripts/install-real-and-sim.sh install_ros_dep_lib
ENV DEBIAN_FRONTEND=noninteractive
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' 
RUN apt install -y curl 
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - 
RUN apt update -y
RUN apt install -y ros-noetic-desktop-full git
RUN apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
RUN rosdep init
RUN sudo rosdep fix-permissions
RUN rosdep update
ENV source /opt/ros/noetic/setup.bash

# install additional ROS packages
RUN apt-get update -y
RUN apt install --allow-unauthenticated -y ros-noetic-rosbridge-suite ros-noetic-multirobot-map-merge ros-noetic-explore-lite \
                    ros-noetic-ros-controllers ros-noetic-rospack \
                    python3-tk libeigen3-dev \
                    ros-noetic-roslint ros-noetic-catkin python3-catkin-tools  ros-noetic-robot-state-publisher

###########################################################################
# Install Python3.8
# RUN $OFFWORLD_GYM_ROOT/scripts/install-real-and-sim.sh install_python_dep_lib
RUN add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y python3.8 python3.8-dev 

# Install Offworld Client Library and Python dependecies
RUN export OFFWORLD_GYM_ROOT=`pwd`/..
RUN apt install -y libbullet-dev git curl wget python3-distutils
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/python3/bin
ENV PATH=$PATH:/root/.local/bin
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

RUN python3.8 -m pip install --upgrade pip
RUN pip3.8 install --user --upgrade testresources==2.0.1
RUN pip3.8 install --user --upgrade setuptools==60.9.3
RUN pip3.8 install --user --upgrade numpy==1.22.3
RUN pip3.8 install --user --upgrade scipy==1.8.0
RUN pip3.8 install --user --upgrade opencv-python==4.5.5.64
RUN pip3.8 install --user --upgrade catkin_pkg==0.4.24
RUN pip3.8 install --user --upgrade empy==3.3.4
RUN pip3.8 install --user --upgrade requests==2.27.1
RUN pip3.8 install --user --upgrade defusedxml==0.7.1 
RUN pip3.8 install --user --upgrade rospkg==1.4.0
RUN pip3.8 install --user --upgrade matplotlib==3.5.1
RUN pip3.8 install --user --upgrade netifaces==0.11.0
RUN pip3.8 install --user --upgrade regex==2022.3.2
RUN pip3.8 install --user --upgrade psutil==5.9.0
RUN pip3.8 install --user --upgrade gym==0.23.0
RUN pip3.8 install --user --upgrade python-socketio==5.5.2
RUN pip3.8 install --user --upgrade scikit-image==0.19.2
RUN pip3.8 install --user --upgrade pyquaternion==0.9.5
RUN pip3.8 install --user --upgrade imageio==2.16.1
RUN pip3.8 install --user --upgrade importlib-metadata==4.11.2

###########################################################################
# Gazebo and nvm
# RUN $OFFWORLD_GYM_ROOT/scripts/install-real-and-sim.sh install_gazebo_nvm_dep_lib
# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh


RUN curl -sSL http://get.gazebosim.org | sh
RUN apt-get install -y libjansson-dev npm libboost-dev imagemagick libtinyxml-dev mercurial cmake build-essential 
RUN apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control

ENV NVM_DIR=/usr/local/nvm
RUN mkdir -p $NVM_DIR
ENV NODE_VERSION=9.11.2

# # Install nvm 
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash \
    && source $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default 

ENV NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Copy and Initialize workspace
RUN mkdir -p /offworld-gym
ENV  OFFWORLD_GYM_ROOT='/offworld-gym'
COPY . /offworld-gym
WORKDIR /offworld-gym

# Install gzweb 
WORKDIR /
# RUN mv $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gzweb /
RUN git clone https://github.com/offworld-projects/offworld-gzweb.git -b fix-camera-angle/gzweb_1.4.1
RUN mv offworld-gzweb gzweb
RUN cd /gzweb \
    && source /usr/share/gazebo/setup.sh \
    && mkdir -p /gzweb/http/client/assets/ \ 
    && npm run deploy --- -m local \
    && npm run update

WORKDIR $OFFWORLD_GYM_ROOT
RUN pip3.8 install --user -e .


RUN mkdir -p $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
WORKDIR $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src
RUN git clone https://github.com/offworld-projects/rosbot_description.git -b offworld-gym

WORKDIR $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws
RUN chmod +x $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/scripts/command_server.py
RUN chmod +x $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/scripts/robot_commands.py

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
# RUN $OFFWORLD_GYM_ROOT/scripts/install-real-and-sim.sh build_gym_shell_script

###########################################################################
# Setting environment variables 
# Appended to the python system path at runtime to import ROS python modules regardless of existing env setup.
ENV GAZEBO_GYM_PYTHON_DEPENDENCIES=/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3.8/dist-packages

# Replaces PYTHONPATH in the subprocess that calls roslaunch within a Gazebo Gym (So that your python3 site-packages don't get imported by accident)
ENV ROSLAUNCH_PYTHONPATH_OVERRIDE=/opt/ros/noetic/lib/python3.8/dist-packages:/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages

# 2nd line below is a hack. Something in the original astra_minimal.dae causes rendering to crash in chrome, but astra.dae works.
RUN cp -r /usr/share/gazebo-11/media /gzweb/http/client/assets/ \
    && cp -r /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models/* /gzweb/http/client/assets/ \
    && cp -r /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/rosbot_description/src/rosbot_description /gzweb/http/client/assets/ \
    && cp /gzweb/http/client/assets/rosbot_description/meshes/astra.dae /gzweb/http/client/assets/rosbot_description/meshes/astra_minimal.dae


# Add to .bashrc for root user
ENV GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH
ENV OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT
ENV PYTHONPATH=/usr/local/lib/python3.8/dist-packages/:/root/.local/lib/python3.8/site-packages:/usr/lib/python3/dist-packages/:/usr/lib/python3/dist-packages:$PYTHONPATH
ENV GAZEBO_MODEL_PATH=$OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/catkin_ws/src/gym_offworld_monolith/models:$GAZEBO_MODEL_PATH
RUN chmod +x $OFFWORLD_GYM_ROOT/offworld_gym/envs/gazebo/docker_entrypoint.sh
WORKDIR $OFFWORLD_GYM_ROOT

CMD ["source /offworld-gym/offworld_gym/envs/gazebo/docker_entrypoint.sh"]
ENTRYPOINT ["/bin/bash","-c"]



