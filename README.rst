Prerequisites
=============

* Python 3.4+ - You can download from https://www.python.org/downloads/
  or, using Anaconda/Miniconda, from https://www.continuum.io/downloads.
* NumPy - already contained in anaconda, in miniconda run
  ``conda install numpy``
* PyYAML - already contained in anaconda, in miniconda run
  ``conda install PyYAML``

  * You can also install both prerequisites in single command using
    ``conda install numpy PyYAML``.

Installation
============

No-install
----------

You can use the package "as is" without any installation:

#. Download the whole package (its root directory).
#. Copy its contents (should contain directory evo, README.rst,
   DESCRIPTION.rst and other files) to arbitrary location (here called
   ``DIR``).
#. Done. To run, ``cd`` to ``DIR`` and from there run
   ``python -m evo [arguments]``.

Install
-------

You can install the package so that it can be used from anywhere:

#. Download the whole package (its root directory).
#. Copy its contents (should contain directory ``evo``, files
   ``README.rst``, ``DESCRIPTION.rst`` and others) to arbitrary location
   (here called ``DIR``).
#. ``cd`` to the directory ``DIR``.
#. Run ``python setup.py install``.
   **NOTE**: the prerequisites (see above) should be installed before running
   this command.
#. Done, you can even delete the downloaded package (it is now
   installed in the conda environment).
   Run with ``python -m evo [arguments]`` from anywhere.

Basic usage
===========

The executable thing is the ``evo`` module.
To run it, enter ``python -m evo`` to the command line.
The basic usage is following:
``python -m evo [common options] algorithm [algorithm options]``.
All algorithms have a ``-h``/``--help`` options to show the reference of their
options.

Common options are:

``-h`` or ``--help``
    prints the basic usage and exits

``--version``
    prints version and exits (quite useless now)

``--logconf CONFFILE``
    sets the logging configuration file in yaml format (optional, there is a
    default logging set up)

Algorithms are:

``bpgp`` - BackPropagation Genetic Programming
    MGGP-like algorithm with nodes that apply additive and multiplicative
    constants at their inputs, use linear combinations of features (LCF) as
    leaf nodes, etc.
