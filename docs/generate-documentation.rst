*****************************
Building sphinx documentation
*****************************

This must be execute on a system with ROS being installed, so that the API generatino process can import all the necessary Python modules

Install the necessary components:

.. code:: bash

    pip3 install Sphinx sphinx-rtd-theme matplotlib psutil rospkg

And the libraries that are mentioned in the documentation for automatic API doc generation:

.. code:: bash

    git clone git@github.com:offworld-projects/keras-rl.git -b offworld-gym
    cd keras-rl
    pip3 install -e .

To rebuild this documentation, enter the ``docs`` directory, run

.. code:: bash

    sphinx-apidoc -f -o source ..

to generate API documentation, then run

.. code:: bash

    rm -rf _build && make html

to build the documentation HTML files.
