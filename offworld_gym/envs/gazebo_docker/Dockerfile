
# Dockerfile for OffWorld-Gym Remote Env Server.
# Encapsulates ROS to allow for less user installation fuss and parallel Gazebo environments on a single host

FROM nvidia/cudagl:10.0-devel-ubuntu16.04

RUN apt-get update \
	&& apt-get install -y apt-utils lsb-release software-properties-common \
	&& apt-get clean all

RUN add-apt-repository ppa:deadsnakes/ppa \
	&& apt-get update \
	&& apt-get install -y python3.6 python3.6-dev psmisc

# Just debugging tools, could be shed for a lighter install.
RUN apt-get install -y nano net-tools xvfb tmux

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
	&& apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
	&& apt-get update \
	&& apt-get install -y ros-kinetic-desktop-full git \
	&& rosdep init \
	&& rosdep update

RUN git clone https://github.com/offworld-projects/offworld-gym.git \
	&& cd offworld-gym/scripts \
	&& export OFFWORLD_GYM_ROOT=`pwd`/.. \
	&& ./install-real-and-sim.sh

RUN cd /offworld-gym \
	&& git pull \
	&& git checkout origin/docker-devel \
	&& chmod +x /offworld-gym/offworld_gym/envs/gazebo_docker/docker_entrypoint.sh

ENV PATH "/root/.local/bin:$PATH"

# Additional ENV Variables to separete elements of PYTHONPATH dependencies among the specific systems that require them.
# (May have unnecessary elements in these paths, e.g. catkin_ws packages)

# Appended to the python system path at runtime to import ROS python modules regardless of existing env setup.
ENV GAZEBO_GYM_PYTHON_DEPENDENCIES /offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages

# Replaces PYTHONPATH in the subprocess that calls roslaunch within a Gazebo Gym (So that your python3 site-packages don't get imported by accident)
ENV ROSLAUNCH_PYTHONPATH_OVERRIDE /opt/ros/kinetic/lib/python2.7/dist-packages:/offworld-gym/offworld_gym/envs/gazebo/catkin_ws/devel/lib/python3/dist-packages

# Currently calling this as a run command
# ENTRYPOINT /offworld-gym/offworld_gym/envs/gazebo/remote/docker_entrypoint.sh



