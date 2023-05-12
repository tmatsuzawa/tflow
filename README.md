# tflow- a Python package to analyze 2D/3D (laminar/turbulent) velocity fields
[Update on 5/12/2023]: The current version is not ready for public sharing. For anyone interested in using this package, I ask for your patience as I clean up the repo. The expected release date will be between July and August. You are welcome to check the current version of repo but the current version include some preliminary functions that needs debugging. 
[History] This Python analysis package started as my personal repo to handle experimental fluid mechanics data. (The setup file for example was customized for my needs.) Over the years, it expanded its functionalities for my own research, then my colleagues'. 
[Project website](https://tmatsuzawa.github.io/tflow)
[Research article which used tflow](https://www.nature.com/articles/s41567-023-02052-0) (Supplementary Information showcases quantities derived by this module. Some of them might be of your interest. For example, it computes inviscid invariants from a 4D velocity field with a single line of code.)

This package includes useful modules to analyze turbulent velocity fields.

The only key assumption is the incompressibility of the medium; however, one should feel free to fork this repo to develop the package for the compressible fluids.
It would require many but minor modifications overall.

This package is perhaps more attractive to experimentally obtained velocity fields because all of the functions for analysis does not assume periodicity or finiteness of the velocity fields. (Numerically obtained velocity fields are often simulated in ideal conditions, which lead to many advantages.)

## Key modules
- velocity.py: a core analysis module
- graph.py: a wrapper of matplotlib to efficiently plot the output of velocity.py
- davis2hdf5.py: LaVision Inc. offers a cutting-edge PIV/PTV software called DaVis. This converts their txt output to a single hdf5 to store a velocity field data.

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


<!---
Other than DaVis, one could use an open-source MATLAB plug-in called PIVLab to obtain a velocity field. This package includes some helpful files to automate this process for multiple files.

### DEPENDENCIES
matlab 2015b or newer

### SETUP PROCEDURES
1. In order to call MATLAB from Python, you must install a package to connect matlab using python. On MATLAB, type
cd (fullfile(matlabroot,'extern','engines','python'))
system('sudo python setup.py install')

2. Run python setup.py
This will make a path to ./matlab_codes on MATLAB.

### HOW TO USE
0. Prepare a PIV cine. (Call a path to the cine as cinepath)
1. Run python make_tiffs.py -cine cinepath
2. Open MALTAB
3. Run ./matlab_codes/process_dir
... Dirbase = /path/to/dir/where/cine_is_located
-->
