"""
Dependencies:
- cine module



"""
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/fapm/')

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import h5py
import subprocess
import argparse
import os
from tqdm import tqdm
import pivlab2hdf5
import analyze_flows
from PIL import Image
import cv2


#----------LOCAL DEPENDENCE----------
import fapm.cine_local.cine.cine as cine  # use local cine package
import fapm.cine_local.cine.tiff as tiff  # use local cine package


description= '---------------------------------------------------------------------\n' \
             'You must respect the file architecture as summarized below.\n' \
             'The top directory is where cine lives. (cinedir)\n' \
             'Inside a basedir, PIVLab outputs are sorted by (Inter. Area, Dt, step).\n' \
             'Inside a parentdir, you find many datadirs. Each datadir corresponds to a cine.\n' \
             'Inside a datadir, you find raw PIVLab outputs.\n \n' \
       'Nomenclature: \n' \
       'cinedir: directory where cine is stored.\n\
   |    e.g.- /Volumes/labshared4/takumi/old_data/sample_piv_cine/\n\
   |\n\
basedir: directory where parentdirs are housed. All PIVLab outputs are sorted by (Inter. Area, Dt, step)\n\
   |    e.g.- /Volumes/labshared4/takumi/old_data/sample_piv_cine/pivlab_outputs\n\
   |\n\
parentdir: directory where PIVLab outputs of many cines (but with the same PIV parameters) are stored\n\
   |    e.g.- .../PIV_W16_step1_data\n\
   |\n\
datadir: directory where PIV outputs are stored.\n\
   |    e.g.- .../cinename_no_extension\n\
   |    ... In datadir, one finds D000000.txt, D000001.txt, etc. as well as piv_settings\n\
   |\n\
piv_settings and piv_settings_2: directories where piv-related parameters are stored.\n\
        ... The information in both dirs are identical but format is different. Choose your preferred format to use.\n' \
             '---------------------------------------------------------------------\n' \
             '\n\n'

print description

parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
parser.add_argument('-basedir', '--basedir', help='Use all available data to make a master hdf5. i.e. PROCESS ALL', type=str,
                    default=None)
parser.add_argument('-parentdir', '--parentdir', help='Use all available data inside a single PARENTDIR to make a master hdf5', type=str,
                    default=None)
parser.add_argument('-dir', '--dir', help='Use a single DATADIR to make a master hdf5', type=str,
                    default=None)
parser.add_argument('-cine', '--cine', help='Find all available PIVLab outputs about a PARTICULAR CINE.', type=str,
                    default=None)
parser.add_argument('-overwrite', '--overwrite',
                    help='overwrite pivlab outputs. This is handy if code fails and force code to insert pivlab outputs to hdf5. Default: False',
                    type=bool,
                    default=False)
args = parser.parse_args()


# make pivlab outputs into hdf5
hdf5datapaths = pivlab2hdf5.main(args)

print '-------------------------------------------------------------------------------------------------'
print 'master hdf5 files are successfully generated.'
print 'Running basic analysis...'
print '-------------------------------------------------------------------------------------------------'

for hdf5datapath in hdf5datapaths:
    analyze_flows.main(hdf5datapath, overwrite=args.overwrite)



