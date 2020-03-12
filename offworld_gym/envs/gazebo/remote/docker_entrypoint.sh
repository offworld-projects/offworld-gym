#!/bin/bash

source /offworld-gym/scripts/gymshell.sh
export PYTHONPATH=""
cd /offworld_gym
git pull
python3.6 -m offworld_gym.envs.gazebo.remote.remote_env_server