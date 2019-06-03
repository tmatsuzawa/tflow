DEPENDENCIES:
matlab 2015b or newer


SETUP PROCEDURES:
1. In order to call MATLAB from Python, you must install a package to connect matlab using python. On MATLAB, type
	cd (fullfile(matlabroot,'extern','engines','python'))
	system('sudo python setup.py install')

2. Run python setup.py
	This will make a path to ./matlab_codes on MATLAB.


HOW TO USE:
0. Prepare a PIV cine. (Call a path to the cine as cinepath)
1. Run python make_tiffs.py -cine cinepath
2. Open MALTAB
3. Run ./matlab_codes/process_dir
	... Dirbase = /path/to/dir/where/cine_is_located
4. Run 
