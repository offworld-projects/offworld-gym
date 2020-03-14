#!/bin/bash

source /offworld-gym/scripts/gymshell.sh
export PYTHONPATH=""  # PYTHONPATH dependecies are dynamically added by the systems that require them.
python3.6 -m offworld_gym.envs.gazebo_docker.remote_env_server