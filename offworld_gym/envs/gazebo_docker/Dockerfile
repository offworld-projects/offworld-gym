# Dockerfile for OffWorld-Gym Remote Env Server.
# Encapsulates ROS to allow for less user installation fuss and parallel Gazebo environments on a single host

FROM ubuntu:20.04

# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Copy and Initialize workspace
RUN mkdir -p /offworld_gym
ENV  OFFWORLD_GYM_ROOT='/offworld_gym'
ADD  examples /offworld_gym/examples
ADD  offworld_gym /offworld_gym/offworld_gym
ADD  scripts /offworld_gym/scripts
WORKDIR /offworld_gym

# Install system level dependecies
RUN apt-get update -y
RUN apt-get install -y -q gnupg2 apt-utils lsb-core lsb-release software-properties-common dialog 
RUN apt-get clean all 
RUN apt-get install -y -q nano net-tools xvfb tmux glew-utils mesa-utils

# Install ROS Gazebo and other Python dependencies
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

# gazebo
RUN apt install -y gazebo9 libgazebo9-dev
RUN apt install -y libjansson-dev libboost-dev imagemagick libtinyxml-dev mercurial cmake build-essential

    # install nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.3/install.sh | bash
RUN nvm install 8
WORKDIR /;
RUN git clone https://github.com/osrf/gzweb
WORKDIR /gzweb
RUN git checkout gzweb_1.4.1 


# RUN git clone -b fix/new-reward-logic-for-monolith https://github.com/offworld-projects/offworld-gym.git 

# WORKDIR /offworld-gym/scripts 

# ENV PATH "/root/.local/bin:$PATH"



# RUN pip3.6 install --user docker
# 	# adding this pip install here because we probably will want to separate docker env install from other env installs later

# RUN cd /offworld-gym \
# 	&& git pull \
# 	&& git checkout origin/main \
# 	&& chmod +x /offworld-gym/offworld_gym/envs/gazebo_docker/docker_entrypoint.sh

# # Additional ENV Variables to separete elements of PYTHONPATH dependencies among the specific systems that require them.
# # (May have unnecessary elements in these paths, e.g. catkin_ws packages)

# # Appended to the python system path at runtime to import ROS python modules regardless of existing env setup.
# ENV GAZEBO_GYM_PYTHON_DEPENDENCIES /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages

# # Replaces PYTHONPATH in the subprocess that calls roslaunch within a Gazebo Gym (So that your python3 site-packages don't get imported by accident)
# ENV ROSLAUNCH_PYTHONPATH_OVERRIDE /opt/ros/kinetic/lib/python2.7/dist-packages:/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages

# ENV NVM_DIR /usr/local/nvm
# ENV NODE_VERSION 8.17.0

# # Replace shell with bash so we can source files
# RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# RUN apt-get install -y libjansson-dev npm libboost-dev imagemagick libtinyxml-dev mercurial cmake build-essential libignition-math2-dev

# # Install nvm with node and npm
# RUN curl https://raw.githubusercontent.com/creationix/nvm/v0.30.1/install.sh | bash \
#     && source $NVM_DIR/nvm.sh \
#     && nvm install $NODE_VERSION \
#     && nvm alias default $NODE_VERSION \
#     && nvm use default \
#     && cd / \
#     && git clone https://github.com/osrf/gzweb \
#     && cd gzweb \
#     && git checkout gzweb_1.4.0 \
#     && source /offworld-gym/scripts/gymshell.sh \
#     && source /usr/share/gazebo/setup.sh \
#     && mkdir -p /gzweb/http/client/assets/ \
#     && npm run deploy --- -m local

# ENV NODE_PATH $NVM_DIR/v$NODE_VERSION/lib/node_modules
# ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH


# # 2nd line below is a hack. Something in the original astra_minimal.dae causes rendering to crash in chrome, but astra.dae works.
# RUN cp -r /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/src/rosbot_description/src/rosbot_description /gzweb/http/client/assets/ \
#     && cp /gzweb/http/client/assets/rosbot_description/meshes/astra.dae /gzweb/http/client/assets/rosbot_description/meshes/astra_minimal.dae