#~/bin bash

if [ ! -d "offworld_gym" ] ; then
  echo "ERROR: Please run this script from within the root directory of the Offworld Gym repository"
  exit 1
fi

export OFFWORLD_GYM_ROOT=`pwd`

# create and activate the environment
sudo apt-get install python3-venv
python3.6 -m venv assets/owgym-kerasrl

# pre-register necessary variables
echo " " >> assets/owgym-kerasrl/bin/activate
echo "export OFFWORLD_GYM_ROOT=`pwd`" >> assets/owgym-kerasrl/bin/activate
echo "export OFFWORLD_GYM_ACCESS_TOKEN=paste_it_here" >> assets/owgym-kerasrl/bin/activate

# activate the environment
source assets/owgym-kerasrl/bin/activate

# install dependencies
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade tensorflow==1.14.0
pip install --upgrade keras==2.2.4
pip install --upgrade requests
pip install --upgrade defusedxml
pip install --upgrade matplotlib
pip install --upgrade netifaces
pip install --upgrade regex
pip install --upgrade psutil
pip install --upgrade gym
pip install --upgrade python-socketio
pip install --upgrade scikit-image
pip install --upgrade pyquaternion
pip install --upgrade imageio

# install OffWorld fork of keras-rl
cd $OFFWORLD_GYM_ROOT/assets
git clone https://github.com/offworld-projects/keras-rl.git -b offworld-gym
cd $OFFWORLD_GYM_ROOT/assets/keras-rl
pip install -e .

# install  offworld-gym
cd $OFFWORLD_GYM_ROOT
pip install -e .

# a workaround to downgrade opencv version
pip uninstall -y opencv-python
pip install opencv-python==3.4.9.31

deactivate

echo ""
echo "Virtual environment has been conigured. Run 'source assets/owgym-kerasrl/bin/activate' to activate."
echo ""
