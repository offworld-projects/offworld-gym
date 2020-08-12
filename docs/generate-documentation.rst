*****************************
Building sphinx documentation
*****************************

Install the necessary components:

.. code:: bash

    pip3 install Sphinx sphinx-rtd-theme matplotlib psutil rospkg


To rebuild this documentation, enter the ``docs`` directory, run

.. code:: bash

    sphinx-apidoc -f -o source ..

to generate API documentation, then run

.. code:: bash

    rm -rf _build && make html

to build the documentation HTML files.
