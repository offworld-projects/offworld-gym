#!/bin/bash

Xvfb :1 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:1.0
source /offworld-gym/scripts/gymshell.sh
npm start --prefix /gzweb &
export PYTHONPATH=""  # PYTHONPATH dependecies are dynamically added by the systems that require them.
python3.6 -m offworld_gym.envs.gazebo_docker.remote_env_server
