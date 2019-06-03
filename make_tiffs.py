"""
Makes a series of tiff files from a cine
Or
Makes a series of tiff files from each cine in a directory (NOT IMPLEMENTED YET)

Dependencies:
- cine module



"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import h5py
import subprocess
import argparse
import os
import sys
from tqdm import tqdm
from pymatbridge import Matlab

from PIL import Image
import cv2

sys.path.append('/Users/stephane/Documents/git/takumi/fapm/cine_local')
#----------LOCAL DEPENDENCE----------
import library.field.velocity as vel
import fapm.cine_local.cine.cine as cine  # use local cine package
import fapm.cine_local.cine.tiff as tiff  # use local cine package
# To find out where Matlab is installed, type matlabroot on Matlab.
matlab_path = '/Applications/MATLAB_R2019a.app/bin/matlab'


faqm_dir = os.path.split(os.path.realpath(__file__))[0]
matlabdir = os.path.join(faqm_dir, 'matlab_codes')

def cine2tiff(cinepath, start=0, end=None, step=1, header='im', overwrite=False):
    """
    Generate a list of tiff files extracted from a cinefile.
        Different modes of processing can be used, that are typically useful for PIV processings :
        test : log samples the i;ages, using the function test_sample. Default is 10 intervals log spaced, every 500 images.
        Sample : standard extraction from start to stop, every step images, with an interval ctime between images A and B.
        File : read directly the start, stop and ctime from a external file. Read automatically if the .txt file is in format :
        'PIV_timestep'+cine_basename+.'txt'
    INPUT
    -----
    file : str
        filename of the cine file
    mode : str.
        Can be either 'test','Sample', 'File'
        single : list of images specified
        pair : pair of images, separated by a ctime interval
    step : int
        interval between two successive images to processed.
    start : int. default 0
        starting index
    stop : int. default 0.
        The cine will be processed 'till its end
    ctime :
    folder : str. Default '/Tiff_folder'
        Name of the root folder where the images will be saved.
    post : str. Default ''
        post string to add to the title of the tiff folder name
    OUTPUT
    OUTPUT
    None
    """
    cc = cine.Cine(cinepath)
    cinepath_, ext = os.path.splitext(cinepath)
    cinedir, cinename = os.path.split(cinepath_)
    savedir = cinedir + '/Tiff_folder/' + cinename + '/'

    # create a directory where images will be stored
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if end is None:
        end = len(cc)

    indices_to_save = range(start, end, step)

    for index in tqdm(indices_to_save):
        # get frames from cine file
        filename = savedir + header + '%05d.tiff' % index
        # save only if the image does not already exist !
        if not os.path.exists(filename) or overwrite:
            # print (filename, index)
            Tiff = tiff.Tiff(filename, 'w')
            Tiff.close()


            data = cc.get_frame(index)
            im = Image.fromarray(data)
            im.save(filename)

            # misc.imsave(filename, data, 'tiff')
    cc.close()




#Sample cine path
cinepath_sample = '/Volumes/labshared4/takumi/old_data/sample_piv_cine/PIV_fv_vp_left_micro105mm_fps2000_Dp120p0mm_D25p6mm_piston10p5mm_freq5Hz_v200mms_setting1_inj1p0s_trig5p0s_fx0p0615mmpx.cine'


parser = argparse.ArgumentParser(description='Comprehensive interactive tool to analyze PIV cine')
parser.add_argument('-cine', '--cine', help='path to cine file', default=cinepath_sample)
parser.add_argument('-start', '--start', help='', default=0)
parser.add_argument('-end', '--end', help='', default=10)
parser.add_argument('-overwrite', '--overwrite', help='overwrite images', default=False)
args = parser.parse_args()

cinepath = args.cine

# DATA ARCHITECTURE
basedir, cinename = os.path.split(cinepath)
scriptdir, scriptname = os.path.split(os.path.realpath(__file__))

# OPEN A CINE FILE
cc = cine.Cine(args.cine)
time = cc.get_time_list()

# MAKE TIFFS FROM CINE
## 1. Make a multi-tiff file (using cine2tiff)
# cine2tiff_path = os.path.join(scriptdir, 'cine/cine2tiff')
# subprocess.call([cine2tiff_path, cinepath_sample])

## 2. Make n tiff files
cine2tiff(cinepath, start=args.start, end=args.end, overwrite=args.overwrite)


# ## Execute matlab script
# print matlabdir
# print os.path.exists(matlabdir + '/test2.m')
# mlab = Matlab(executable=matlab_path)
# mlabstart = mlab.start()
# # result = mlab.run_func(matlabdir + '/process_dir.m', {'Dirbase': basedir, 'codedir': matlabdir})
# # result = mlab.run_func(matlabdir + '/process_dir.m', {'arg1': 3, 'arg2': 5})
# result = mlab.run_func(matlabdir + '/test.m', {'arg1': 3, 'arg2': 5})
# print result['result']
# # mlab.quit()
#
#


cc.close()
# print time, len(time)







