Installation
============

The installation was tested on: Ubuntu 16.04.6. Following these steps will prepare you for running both the Real and the Sim versions of OffWorld Gym. You must use **Python 3** with this environemt, Python 2 is not supported.

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

Each time you will be running OffWorld Gym, execute

.. code:: bash

    source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh

in each new terminal. Or add it  your ~/.bashrc by running

.. code:: bash

    echo "source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh" >> ~/.bashrc

**To test the Real environment** you need to register as a user at `gym.offworld.ai <https://gym.offworld.ai>`_, copy "OffWorld Gym Access Token" from your `Profile  <https://gym.offworld.ai/account>`_ page into ``OFFWORLD_GYM_ACCESS_TOKEN`` variable in your ``offworld_gym/scripts/gymshell.sh`` script, and follow the instructions in the `Examples <./examples.html>`_ section.

**To test the Sim environment** open two terminals, ``source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh`` in each of them, and run:  

.. code:: bash

    terminal one: roslaunch gym_offworld_monolith env_bringup.launch  
    terminal two: gzclient  
