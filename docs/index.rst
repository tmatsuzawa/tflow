.. tflow documentation master file, created by
   sphinx-quickstart on Sat Apr 16 15:31:57 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tflow's documentation!
=================================

Project webpage: https://tmatsuzawa.github.io/tflow

As the name 'tflow' indicates, the package includes useful modules to analyze turbulent velocity fields.

The only key assumption is the incompressibility of the medium; however, one should feel free to fork this repo to develop the package for the compressible fluids.
It would require many but minor modifications overall.

This package is perhaps more attractive to experimentally obtained velocity fields because all of the functions for analysis does not assume periodicity or finiteness of the velocity fields. (Numerically obtained velocity fields are often simulated in ideal conditions, which lead to many advantages.)

## Key modules
- velocity.py: a core analysis module
- graph.py: a wrapper of matplotlib to efficiently plot the output of velocity.py
- davis2hdf5.py: LaVision Inc. offers a cutting-edge PIV/PTV software called DaVis. This converts their output to a single hdf5 to store a velocity field data.

## A typical workflow
A. Exeperiments
1. Conduct PIV/PTV experiement using DaVis
2. Convert the velocity field data into a hdf5 (One may use davis2hdf5 for DaVis txt output)
3. Import tflow.velocity
4. Analyze and plot

B. Numerics
1. Generate a velocity field data (DNS, LES, etc.)
2. Import tflow.velocity
3. Analyze and plot

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
