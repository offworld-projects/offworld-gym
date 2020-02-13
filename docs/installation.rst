Installation
============

There are two installation options: **real-only** (OS agnostic, no ROS dependencies) and **real-and-sim** (requires Ubuntu 16.04 and ROS Kinetic).

.. note::
    Python 2.7 is not supported. Please run your scripts using **Python 3.6**.


Real-only
---------

This installation only depends on python libraries and thus can be installed on any OS.  
We provide an `installation script for Ubuntu <https://github.com/offworld-projects/offworld-gym/blob/develop/scripts/install-real.sh>`_ (tested on Ubuntu 16.04 and 18.04), which you can use a guideline for installation on other systems (feel free to submit a pull request with an installation script for your system).

.. code:: bash

    git clone https://github.com/offworld-projects/offworld-gym.git
    cd offworld-gym/scripts
    export OFFWORLD_GYM_ROOT=`pwd`/..
    ./install-real.sh

To test the installation, do the following:

    1. Book your time slot at `https://gym.offworld.ai/book <https://gym.offworld.ai/book>`_
    2. Copy the **OffWorld Gym Access Token** from `https://gym.offworld.ai/account <https://gym.offworld.ai/account>`_ to ``scripts/gymshell.sh``
    3. Activate the Gym Shell by calling ``source scripts/gymshell.sh``
    4. Open `https://gym.offworld.ai/cameras <https://gym.offworld.ai/cameras>`_ in your browser to see the real-time camera stream!
    5. Run the minimal example ``python3.6 examples/minimal_example_real.py`` and see the robot move.



Real and Sim
------------

The simulated replicas of OffWorld Gym environments are build in Gazebo, rely on ROS Kinetic and thus require Ubuntu 16.04.
The following steps will prepare you for running both the Real and the Sim versions of OffWorld Gym.


Pre-requisites
^^^^^^^^^^^^^^

Please install the following components using the corresponding installation instructions.

    * `ROS Kinetic <http://wiki.ros.org/kinetic/Installation/Ubuntu>`_
  
For GPU support also install

  * `CUDA 10.0 Library <https://developer.nvidia.com/cuda-10.0-download-archive>`_
  * `cuDNN 7.0 Library <https://developer.nvidia.com/cudnn>`_



Setup
^^^^^
The installation scripts `scripts/install-real-and-sim.sh <https://github.com/offworld-projects/offworld-gym/blob/develop/scripts/install-real.sh>`_ has been tested on Ubuntu 16.04.

.. code:: bash

    git clone https://github.com/offworld-projects/offworld-gym.git
    cd offworld-gym/scripts
    export OFFWORLD_GYM_ROOT=`pwd`/..
    ./install-real-and-sim.sh

To test the Sim installation open two terminals, start the environment in the first one

.. code:: bash

    source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
    roslaunch gym_offworld_monolith env_bringup.launch

and a Gazebo instance in another

.. code:: bash

    source $OFFWORLD_GYM_ROOT/scripts/gymshell.sh
    gzclient

To test the real installation please follow the steps in the previous section.