Installation
============

The installation of the OffWorld Gym is OS agnostic and gives you access to real physical environments and their simulated replicas (via Docker).

.. note::
    Python 2.7 and 3.5 are not supported. Please run your scripts using **Python 3.6**.


Pre-requisites
^^^^^^^^^^^^^^

The communication with a real physical OffWorld Gym environment does not require anoython other than Python 3.6, but you will need Docker if you want to make use of the simulated environments:

  * Install Docker `https://docs.docker.com/get-docker <https://docs.docker.com/get-docker/>`_. When installing on Linux we recommend to make Docker runnable without root privileges `https://docs.docker.com/engine/install/linux-postinstall <https://docs.docker.com/engine/install/linux-postinstall>`_.


Setup
^^^^^

.. code:: bash

    git clone https://github.com/offworld-projects/offworld-gym.git
    cd offworld-gym
    pip3 install -e .


Verify the installation
^^^^^^^^^^^^^^^^^^^^^^^

To test your installtion and the connectivity with the physical enviroment, do the following:

    1. Book your time slot at `https://gym.offworld.ai/book <https://gym.offworld.ai/book>`_
    2. ``export OFFWORLD_GYM_ACCESS_TOKEN=paste_your_token_here`` using the OffWorld Gym Access Token from `https://gym.offworld.ai/account <https://gym.offworld.ai/account>`_ (consider adding it to your ``~/.bashrc``)
    3. Run the minimal example ``python3.6 examples/minimal_example_OffWorldMonolithDiscreteReal.py``
    4. Open `https://gym.offworld.ai/cameras <https://gym.offworld.ai/cameras>`_ in your browser to see the live camera feed from the environment!

To run a simulated environment:


    1. ``python3.6 examples/sim/random_monolith_continuous_sim.py``
    2. Open ``localhost:8080`` in your browser to visualize the Gazebo simulation


Additinal steps to run the examples that based on Pytorch, Tianshou and Stable-baselines 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you are interested in running or building upon the QR-DQN examples provided in the ``examples/`` directory you will need to install ``Pytorch``, ``tianshou`` and ``stable-baselines3`` libraries:

    1. ``pip isntall torch==1.10.0 tianshou==0.4.3 stable-baselines3==1.1.0``

we have created an easy script that will create a virtual environment that everything you need, please see the `Examples <./examples.html>`_ section for details.

Local Ubuntu installation without Docker (Deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If for some reason you would like to avoid using Docker for running simulated environments, you can install all the necessary components directly on your system (tested on Ubuntu 16.04).

Install ROS Kinetic by following the corresponding installation instructions.

    * `ROS Kinetic <http://wiki.ros.org/kinetic/Installation/Ubuntu>`_

For GPU support also install

  * `CUDA 10.0 Library <https://developer.nvidia.com/cuda-10.0-download-archive>`_
  * `cuDNN 7.0 Library <https://developer.nvidia.com/cudnn>`_

Run the installation script `scripts/install-real-and-sim.sh <https://github.com/offworld-projects/offworld-gym/blob/develop/scripts/install-real.sh>`_ (tested on Ubuntu 16.04):

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

To test the real installation please follow the steps in the section above.
