"""
Convert DaVis txt outputs into hdf5
"""

import argparse
import h5py
import glob
import numpy as np
from tqdm import tqdm
import os
import re

def write_hdf5_dict(filepath, data_dict, chunks=None, overwrite=True, verbose=True):
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

    if not os.path.exists(filename) or overwrite:
        hf = h5py.File(filename, 'w')
        for key in data_dict:
            if chunks is None or (key == 'x' or key == 'y' or key == 'z'):
                hf.create_dataset(key, data=data_dict[key])
            else:
                hf.create_dataset(key, data=data_dict[key], chunks=chunks)

        hf.close()

        if verbose:
            print('Data was successfully saved as ' + filename)
    else:
        print('... File already exists! No overwriting...')


def convert_dat2h5files(dpath, savedir=None, verbose=False, verbose_fn=True, overwrite=True, start=0, fn_max=np.inf):
    """
    Converts tecplot data files (.data format) to a set of hdf5 files
    Parameters
    ----------
    dpath: str, path to tecplot data file (.data)
    savedir: str, default: None
        ... directories where hdf5 files are going to be saved
    verbose: bool, default: False
    overwrite: bool, default: True
        ... boolean which determines whether hdf5 files will be overwritten
    start: int, default:10
        ... this will be used to name the h5 files

    Returns
    -------

    """
    if not os.path.exists(dpath):
        print('... data does not exist!')
        return None
    else:
        if savedir is None:
            savedir = os.path.split(dpath)[0]
        savedir = os.path.join(savedir, 'hdf5')
        savedir = os.path.join(savedir, os.path.split(dpath)[1][:-4])

        COLUMNS = ["x", "y", "z", "I", "u", "v", "w", "|V|", "trackID", "ax", "ay", "az",
                   "|a|"]  # vel and acc are in m/s or m/s^2

        skiprows = 6
        fn = start
        ln = 0  # Line count

        # initialization
        data_lists = [[] for column in COLUMNS]
        with open(dpath, 'r') as f:
            while fn < fn_max:
                ln += 1
                line = f.readline()

                # Get out of the loop at the end of the file
                if len(line) == 0:
                    break
                if ln > skiprows:
                    qtys = line.split()
                    if "ZONE T" in line:
                        datadict = {}
                        for i, column in enumerate(COLUMNS):
                            datadict[COLUMNS[i]] = np.asarray(data_lists[i])
                        write_hdf5_dict(os.path.join(savedir, 'frame%05d' % fn), datadict,
                                        overwrite=overwrite, verbose=verbose)
                        fn += 1

                        if verbose_fn:
                            if fn % 50 == 0:
                                print("... frame %d" % fn)
                        # initialization
                        data_lists = [[] for column in COLUMNS]
                    else:
                        try:
                            data = {name: float(qty) for name, qty in zip(COLUMNS, qtys)}
                            for i, column in enumerate(COLUMNS):
                                data_lists[i].append(data[COLUMNS[i]])
                        except:
                            continue
    print('... dat2h5files- Done')


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
        input_list = args.input
        for i, dpath in enumerate(input_list):
            convert_dat2h5files(dpath, start=args.start, savedir=args.savedir, overwrite=args.overwrite)
    else:
        print('... Making hdf5 files for dat files in ' + args.dir)
        dat_paths = natural_sort(glob.glob(os.path.join(args.dir, '*.dat')))
        for i, dpath in enumerate(dat_paths):
            print(".dat being processed: ", os.path.split(dpath)[1])
            convert_dat2h5files(dpath, start=args.start, savedir=args.savedir, overwrite=args.overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
    parser.add_argument('input', metavar='input', type=str, nargs='+', default=None, help='path to a .dat file')
    parser.add_argument('--dir', help='Name of a directory which contains dat files', type=str,
                        default=None)
    parser.add_argument('-start', '--start', help='starting index used to label the output h5 files'
                        'default: 0', type=int, default=0)
    parser.add_argument('--savedir', help='Name of a directory where files are output', type=str,
                        default=None)
    parser.add_argument('--overwrite', help='bool to determine whether ouput h5files will be overwritten if it exists',
                        type=bool,
                        default=True)
    args = parser.parse_args()
    main(args)

