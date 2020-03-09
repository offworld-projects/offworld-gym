#!/bin/bash

source /offworld-gym/scripts/gymshell.sh
export PYTHONPATH=""
python3 -m http.server

#python3.6 -m offworld_gym.envs.gazebo.remote.remote_env_remote_interface