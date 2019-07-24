"""
Convert DaVis txt outputs into hdf5
"""

import argparse
import h5py
import glob
import numpy as np
from tqdm import tqdm
import velocity as vel
import os
import re

def write_hdf5_dict(filepath, data_dict):
    """
    Stores data_dict
    Parameters
    ----------
    filepath :  str
                file path where data will be stored. (Do not include extension- .h5)
    data_dict : dictionary
                data should be stored as data_dict[key]= data_arrays

    Returns
    -------

    """
    filedir = os.path.split(filepath)[0]
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    ext = '.h5'
    filename = filepath + ext
    hf = h5py.File(filename, 'w')
    for key in data_dict:
        hf.create_dataset(key, data=data_dict[key])
    hf.close()
    print 'Data was successfully saved as ' + filename


def davis2hdf5_dirbase(dirbase, savedir=None, header='B', scale=1000.):
    """
    Convert multiple davis outputs into hdf5 files

    Parameters
    ----------
    dirbase
    savedir

    Returns
    -------

    """
    if savedir is None:
        if dirbase[-1] == '/':
            savedir = os.path.split(dirbase[:-1])
        else:
            savedir = os.path.split(dirbase)


    datadirs = glob.glob(dirbase + '/*')
    for datadir in tqdm(datadirs, desc='datadir'):
        davis2hdf5(datadir, savedir=savedir, header=header, scale=scale)
    print '... Done'


def davis2hdf5(datadir, savedir=None, savepath=None, header='B', scale=1000.):
    """
     Convert multiple davis outputs into hdf5 files

     Parameters
     ----------
     dirbase
     savedir

     Returns
     -------

     """
    davis_dpaths = glob.glob(datadir + '/%s*' % header)
    davis_dpaths = natural_sort(davis_dpaths)

    duration = len(davis_dpaths)
    for t, dpath in enumerate(tqdm(davis_dpaths)):
        with open(dpath, 'r') as fyle:
            xlist, ylist, ulist, vlist = [], [], [], []
            lines = fyle.readlines()
            height, width = int(lines[0].split()[4]), int(lines[0].split()[5])
            shape = (height, width)
            for i, line in enumerate(lines):
                if i > 0:
                    line = line.replace(',', '.').split()
                    x, y, u, v = [float(i) for i in line]
                    if u == 0:
                        u = np.nan
                    if v == 0:
                        v = np.nan
                    xlist.append(x)
                    ylist.append(y)
                    ulist.append(u)
                    vlist.append(v)
            x_arr = np.asarray(xlist).reshape(shape)
            y_arr = np.asarray(ylist).reshape(shape)

            # dx_d = x_arr[0, 1] - x_arr[0, 0]
            # dy_d = y_arr[1, 0] - y_arr[0, 0]

            u_arr = np.asarray(ulist).reshape(shape) * scale  # davis data is already converted to physical dimensions.
            v_arr = np.asarray(vlist).reshape(shape) * scale

        if t == 0:
            shape_d = (shape[0], shape[1], duration)
            uxdata_d, uydata_d = np.empty(shape_d), np.empty(shape_d)
        uxdata_d[..., t] = u_arr
        uydata_d[..., t] = v_arr
    udata_d = np.stack((uxdata_d, uydata_d))
    # xx_d, yy_d = vel.get_equally_spaced_grid(udata_d)

    if savedir is None:
        if datadir[-1] == '/':
            savedir, dirname = os.path.split(datadir[:-1])
        else:
            savedir, dirname = os.path.split(datadir)
        savepath = savedir + '/davis_piv_outputs/' + dirname

    data2write = {}
    data2write['ux'] = udata_d[0, ...]
    data2write['uy'] = udata_d[1, ...]
    data2write['x'] = x_arr
    data2write['y'] = y_arr
    write_hdf5_dict(savepath, data2write)
    print '... Done'

# make a hdf5 file from a simple dictionary
def write_hdf5_dict(filepath, data_dict):
    """
    Stores data_dict
    Parameters
    ----------
    filepath :  str
                file path where data will be stored. (Do not include extension- .h5)
    data_dict : dictionary
                data should be stored as data_dict[key]= data_arrays

    Returns
    -------

    """
    filedir = os.path.split(filepath)[0]
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    ext = '.h5'
    filename = filepath + ext
    hf = h5py.File(filename, 'w')
    for key in data_dict:
        hf.create_dataset(key, data=data_dict[key])
    hf.close()
    print 'Data was successfully saved as ' + filename

##### misc. ####
def natural_sort(arr):
    def atoi(text):
        'natural sorting'
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        '''
        natural sorting
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split('(\d+)', text)]

    return sorted(arr, key=natural_keys)



##### main method ####
def main(args):
    if args.dir is None:
        print 'Make hdf5 files for directories under ' + args.dirbase
        davis2hdf5_dirbase(args.dirbase)
    else:
        print 'Make a hdf5 file for the following directory: ' + args.dir
        davis2hdf5(args.dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
    parser.add_argument('-dirbase', '--dirbase', help='Parent directory of PIVlab output directories', type=str,
                        default=None)
    parser.add_argument('-dir', '--dir', help='Name of a directory which contains pivlab txt outputs', type=str,
                        default=None)
    args = parser.parse_args()

    main(args)