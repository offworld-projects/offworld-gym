*****************************
Building sphinx documentation
*****************************

To rebuild this documentation, enter ``docs`` directory, then run

.. code:: bash

    sphinx-apidoc -f -o source ..

to generate API documentation, and finally run

.. code:: bash

    rm -rf _build && make html

to build the documentation HTML files.
