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

def write_hdf5_dict(filepath, data_dict, chunks=None):
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
        if chunks is None or (key == 'x' or key == 'y' or key == 'z'):
            hf.create_dataset(key, data=data_dict[key])
        else:
            hf.create_dataset(key, data=data_dict[key], chunks=chunks)

    hf.close()
    print('Data was successfully saved as ' + filename)


def davis2hdf5_dirbase(dirbase, use_chunks, savedir=None, header='B', scale=1000., fps=1., mode='piv',
                       start=0, end=None):
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
            savedir = os.path.split(dirbase[:-1])[0]
        else:
            savedir = os.path.split(dirbase)[0]

    datadirs = glob.glob(dirbase + '/*')
    for datadir in tqdm(datadirs, desc='datadir'):
        davis2hdf5(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps, mode=mode,
                   start=start, end=end)
    print('... Done')

def davis2hdf5(datadir, use_chunks, savedir=None, savepath=None, header='B', scale=1000., chunks=None, fps=1., mode='piv',
               start=0, end=None):
    if mode == 'piv':
        davis2hdf5_piv(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps,
                       start=start, end=end)
    elif mode == 'stb':
        davis2hdf5_stb(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps,
                       start=start, end=end)
    else:
        raise ValueError('... mode must be either \'piv \' or \'stb\'')

def davis2hdf5_piv(datadir, use_chunks, savedir=None, savepath=None, header='B', scale=1000., chunks=None, fps=1.,
                   start=0, end=None):
    """
     Convert multiple DaVis output (PIV) into a hdf5 file


     Parameters
     ----------
     dirbase
     savedir

     Returns
     -------

     """
    davis_dpaths = glob.glob(datadir + '/%s*' % header)
    davis_dpaths = natural_sort(davis_dpaths)
    davis_dpaths = davis_dpaths[start:end]

    duration = len(davis_dpaths)
    for t, dpath in enumerate(tqdm(davis_dpaths)):
        with open(dpath, 'r') as fyle:
            xlist, ylist, ulist, vlist = [], [], [], []
            lines = fyle.readlines()
            height, width = int(lines[0].split()[4]), int(lines[0].split()[5])
            shape = (height, width)
            for i, line in enumerate(lines):
                if i==0:
                    if line.__contains__("\"Position\" \"mm\""):
                        scale = 1.
                        pos_unit = 'mm'
                    else:
                        pos_unit = 'px'
                    if line.__contains__("\"velocity\" \"m/s\""):
                        vscale = 1000.
                        vel_unit = 'm/s'
                    elif line.__contains__("\"displacement\" \"pixel\""):
                        vscale = scale * fps
                        vel_unit = 'px/frame'
                    else:
                        vscale = 1.
                        vel_unit = '????'
                    if t==0:
                        print('\n Units of Position and Velocity: ' + pos_unit, vel_unit)
                        if vel_unit == 'px/frame':
                            print('scale (mm/px), frame rate(fps): %.5f, %.1f' % (scale, fps))
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
            x_arr = np.asarray(xlist).reshape(shape) * scale
            y_arr = np.asarray(ylist).reshape(shape) * scale

            # dx_d = x_arr[0, 1] - x_arr[0, 0]
            # dy_d = y_arr[1, 0] - y_arr[0, 0]

            u_arr = np.asarray(ulist).reshape(shape) * vscale  # davis data is already converted to physical dimensions.
            v_arr = np.asarray(vlist).reshape(shape) * vscale

        if t == 0:
            shape_d = (shape[0], shape[1], duration)
            uxdata_d, uydata_d = np.empty(shape_d), np.empty(shape_d)
        uxdata_d[..., t] = u_arr
        uydata_d[..., t] = v_arr
    udata_d = np.stack((uxdata_d, uydata_d))
    # xx_d, yy_d = vel.get_equally_spaced_grid(udata_d)

    # Path where hdf5 is stored
    if datadir[-1] == '/':
        savedir_default, dirname = os.path.split(datadir[:-1])
    else:
        savedir_default, dirname = os.path.split(datadir)
    if savedir is None:
        savedir = savedir_default
    savepath = savedir + '/davis_piv_outputs/' + dirname

    data2write = {}
    data2write['ux'] = udata_d[0, ...]
    data2write['uy'] = udata_d[1, ...]
    data2write['x'] = x_arr
    data2write['y'] = y_arr
    if use_chunks:
        chunks = udata_d.shape[1:-1] + (1, )
    else:
        chunks = None

    write_hdf5_dict(savepath, data2write, chunks=chunks)
    print('... Done')

def davis2hdf5_stb(datadir, use_chunks, savedir=None, savepath=None, header='B', scale=1000., chunks=None, fps=1.,
                   start=0, end=None):
    """
     Convert multiple DaVis output (PIV) into a hdf5 file


     Parameters
     ----------
     dirbase
     savedir

     Returns
     -------

     """

    def format_array(arr1d, shape):
        """
        Formats a 1d array output by the DaVis STB export feature into the convention used by udata
        ... Convention is (y, x, z). The array must have a shape (height, width, depth).

        Parameters
        ----------
        arr1d: 1d array-like
        shape: tuple,
            ... size of the 3D array (height, width, depth)

        Returns
        -------
        arr: array
            ...
        """
        # shape = (height, width, depth)
        newshape = (shape[2], shape[0], shape[1])
        arr1d = np.asarray(arr1d)
        arr = arr1d.reshape(newshape)  # z, y, x
        arr = np.swapaxes(arr, 0, 2)  # x, y, z
        arr = np.swapaxes(arr, 0, 1)  # y, x, z
        return arr


    davis_dpaths = glob.glob(datadir + '/%s*' % header)
    davis_dpaths = natural_sort(davis_dpaths)
    davis_dpaths = davis_dpaths[start:end]

    duration = len(davis_dpaths)
    for t, dpath in enumerate(tqdm(davis_dpaths)):
        with open(dpath, 'r') as fyle:
            xlist, ylist, zlist, ulist, vlist, wlist = [], [], [], [], [], []
            lines = fyle.readlines()
            line = lines[0].replace(';', ' ').split()
            height, width, depth = int(line[5]), int(line[6]),int(line[7][:-3])
            shape = (height, width, depth)
            for i, line in enumerate(lines):
                if i <= height*width*depth:
                    if i==0:
                        if line.__contains__("\"X\";\"mm\""):
                            scale = 1.
                            pos_unit = 'mm'
                        else:
                            pos_unit = 'px'
                        if line.__contains__("\"velocity\";\"m/s\""):
                            vscale = 1000.
                            vel_unit = 'm/s'
                        elif line.__contains__("\"displacement\" \"pixel\""):
                            vscale = scale * fps
                            vel_unit = 'px/frame'
                        else:
                            vscale = 1.
                            vel_unit = '????'
                        if t==0:
                            print('\n Units of Position and Velocity: ' + pos_unit, vel_unit)
                            if vel_unit == 'px/frame':
                                print('scale (mm/px), frame rate(fps): %.5f, %.1f' % (scale, fps))
                    if i > 0:
                        line = line.replace(';', ' ').split()
                        x, y, z, u, v, w = [float(i) for i in line]
                        if u == 0:
                            u = np.nan
                        if v == 0:
                            v = np.nan
                        if w == 0:
                            w = np.nan
                        xlist.append(x)
                        ylist.append(y)
                        zlist.append(z)
                        ulist.append(u)
                        vlist.append(v)
                        wlist.append(w)

            x_arr = format_array(xlist, shape) * scale
            y_arr = format_array(ylist, shape) * scale
            z_arr = format_array(zlist, shape) * scale

            u_arr = format_array(ulist, shape) * vscale  # davis data should be already converted to physical dimensions.
            v_arr = format_array(vlist, shape) * vscale
            w_arr = format_array(wlist, shape) * vscale
        if t == 0:
            shape_d = shape + (duration, )
            uxdata_d, uydata_d, uzdata_d = np.empty(shape_d), np.empty(shape_d), np.empty(shape_d)
        uxdata_d[..., t] = u_arr
        uydata_d[..., t] = v_arr
        uzdata_d[..., t] = w_arr
    udata_d = np.stack((uxdata_d, uydata_d, uzdata_d))
    # xx_d, yy_d = vel.get_equally_spaced_grid(udata_d)

    # Path where hdf5 is stored
    if datadir[-1] == '/':
        savedir_default, dirname = os.path.split(datadir[:-1])
    else:
        savedir_default, dirname = os.path.split(datadir)
    if savedir is None:
        savedir = savedir_default
    savepath = savedir + '/davis_stb_outputs/' + dirname
    if start != 0 or end is not None:
        if end is None:
            end = len(davis_dpaths)-1
        savepath += '%05d_%05d' % (start, end)

    data2write = {}
    data2write['ux'] = udata_d[0, ...]
    data2write['uy'] = udata_d[1, ...]
    data2write['uz'] = udata_d[2, ...]
    data2write['x'] = x_arr
    data2write['y'] = y_arr
    data2write['z'] = z_arr
    if use_chunks:
        chunks = udata_d.shape[1:-1] + (1, )
    else:
        chunks = None

    write_hdf5_dict(savepath, data2write, chunks=chunks)
    print('... Done')


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
        print('... Making hdf5 files for directories under ' + args.dirbase)
        davis2hdf5_dirbase(args.dirbase, args.chunks, scale=args.scale, fps=args.fps, mode=args.mode,
                           start=args.start, end=args.end)
    else:
        print('... Making a hdf5 file for the following directory: ' + args.dir)
        davis2hdf5(args.dir, args.chunks, scale=args.scale, fps=args.fps, mode=args.mode,
                   start=args.start, end=args.end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
    parser.add_argument('-dirbase', '--dirbase', help='Parent directory of PIVlab output directories', type=str,
                        default=None)
    parser.add_argument('-dir', '--dir', help='Name of a directory which contains pivlab txt outputs', type=str,
                        default=None)
    parser.add_argument('-chunks', '--chunks', help='Use chunked storage. default: True', type=bool,
                        default=True)
    parser.add_argument('-scale', '--scale', help='In mm/px. ux = ux*scale*fps', type=float,
                        default=1000.)
    parser.add_argument('-fps', '--fps', help='In frame per second. ux = ux*scale*fps.', type=float,
                        default=1.)
    parser.add_argument('-mode', '--mode', help='Type of the experiment: PIV, STB? Choose from piv or stb', type=str,
                        default='piv')
    parser.add_argument('-start', '--start', help='index which specifies the range of data data to be used [start, end)'
                                                  'default: 0', type=int,
                        default=0)
    parser.add_argument('-end', '--end', help='index which specifies the range of data data to be used [start, end)'
                                                  'default: None', type=int,
                        default=None)
    args = parser.parse_args()

    main(args)