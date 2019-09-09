Installation
============

The installation was tested on: Ubuntu 16.04.6. Following these steps will prepare you for running both the Real and the Sim versions of the OffWorld Gym. You will be able to use Python 3 with this environemt.

Pre-requisites
--------------
Please install the following components using the corresponding installation instructions.

    * `ROS Kinetic <http://wiki.ros.org/kinetic/Installation/Ubuntu>`_
  
For GPU support also install

  * `CUDA 10.0 Library <https://developer.nvidia.com/cuda-10.0-download-archive>`_
  * `cuDNN 7.0 Library <https://developer.nvidia.com/cudnn>`_



Setup
-----

.. code:: bash

    git clone https://github.com/offworld-projects/offworld-gym.git
    cd offworld-gym/scripts
    export OFFWORLD_GYM_ROOT=`pwd`/..
    ./install.sh

To prepare a terminal for running OffWorld Gym, run

.. code:: bash

    source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

in each new terminal. Or add it  your ~/.bashrc by running

.. code:: bash

    echo "source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh" >> ~/.bashrc

To test Real environment

.. todo:: Add instructions for testing connection with real environment

To test Sim environment open two terminals, ``source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh`` in each of them, and run:  

.. code:: bash

    terminal one: roslaunch gym_offworld_monolith env_bringup.launch  
    terminal two: gzclient  
