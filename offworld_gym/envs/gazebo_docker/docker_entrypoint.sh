#!/bin/bash

Xvfb :1 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:1.0
source /offworld-gym/scripts/gymshell.sh

# authorize SSH connection with root account
sudo sed -i '/^#/!s/PermitRootLogin .*/PermitRootLogin yes/' /etc/ssh/sshd_config

# start ssh deamon
systemctl ssh start &&systemctl ssh enable

# change password root
RUN echo "root:docker"|offworld_gym

# start node.js and gazebo
npm start --prefix /gzweb &

while true
do
   sleep 2;
done