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
cd ..
export OFFWORLD_GYM_ROOT=`pwd`

sudo apt update
sudo apt install -y curl

# Python 3.6
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.6 python3.6-dev
sudo apt-get install -y python3-distutils python3-testresources
curl https://bootstrap.pypa.io/get-pip.py | sudo -H python3.6

pip3.6 install --user --upgrade setuptools
pip3.6 install --user --upgrade pip
pip3.6 install --user --upgrade numpy
pip3.6 install --user --upgrade scipy
pip3.6 install --user --upgrade tensorflow-gpu==1.14.0
pip3.6 install --user --upgrade keras==2.2.4
pip3.6 install --user --upgrade opencv-python
pip3.6 install --user --upgrade requests
pip3.6 install --user --upgrade defusedxml 
pip3.6 install --user --upgrade matplotlib
pip3.6 install --user --upgrade netifaces
pip3.6 install --user --upgrade regex
pip3.6 install --user --upgrade psutil
pip3.6 install --user --upgrade gym
pip3.6 install --user --upgrade python-socketio
pip3.6 install --user --upgrade scikit-image
pip3.6 install --user --upgrade pyquaternion
pip3.6 install --user --upgrade imageio

cd $OFFWORLD_GYM_ROOT
pip3.6 install --user -e .
		
# put modified keras-rl under ./assets
# gymshell.sh will update the $PYTHONPATH to point here
mkdir -p $OFFWORLD_GYM_ROOT/assets
cd $OFFWORLD_GYM_ROOT/assets
git clone https://github.com/offworld-projects/keras-rl.git -b offworld-gym

# build the Gym Shell script
echo '#!/usr/bin/env bash' > $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo "export OFFWORLD_GYM_ROOT=$OFFWORLD_GYM_ROOT" >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo 'export PYTHONPATH=$OFFWORLD_GYM_ROOT/assets/keras-rl:$PYTHONPATH' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
echo 'export OFFWORLD_GYM_ACCESS_TOKEN="COPY IT HERE"' >> $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
chmod +x $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

printf "\n\nInstallation complete\n---------------------\n\n"
printf "To setup a shell for OffWorld Gym run\n\n\tsource $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\nin each new terminal to activate Gym Shell.\n"
printf "Or add to your ~/.bashrc by running\n\n\techo \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\" >> ~/.bashrc\n\n---------------------\n\n"
printf "To test the Real environment:\n\n\t* Book your time slot at https://gym.offworld.ai/book\n\t* Copy the access token from https://gym.offworld.ai/account to $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\n\t* Activate Gym Shell by calling \"source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh\"\n\t* Open https://gym.offworld.ai/cameras in your browser to see the real-time camera stream\n\t* Run the minimal example \"python3.6 examples/minimal_example_real.py\"\n\n"
