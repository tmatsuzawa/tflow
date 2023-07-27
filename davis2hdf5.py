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
import datetime
import cine

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
    with h5py.File(filename, 'w') as hf:
        for key in data_dict:
            if not isinstance(data_dict[key], dict):
                if chunks is not None and key in ['ux', 'uy', 'uz', 'p']:
                    hf.create_dataset(key, data=data_dict[key], chunks=chunks)
                else:
                    hf.create_dataset(key, data=data_dict[key])
            else:
                grp = hf.create_group(key)
                for subkey in data_dict[key]:
                    print(key, subkey, '/n')
                    if chunks is not None and subkey in ['ux', 'uy', 'uz', 'p']:
                        grp.create_dataset(subkey, data=data_dict[key][subkey], chunks=chunks)
                    else:
                        grp.create_dataset(subkey, data=data_dict[key][subkey])

    print('Data was successfully saved at ' + filename)


def davis2hdf5_custom(dirbase, use_chunks, savedir=None, header='B', scale=1000., fps=1., mode='piv',
                       start=0, end=None, savetime=False):
    """
    An ad-hoc function to create hdf5 files from the project directories (This method is not generalizable because of the different hierarchies used in the project
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
                   start=start, end=end, savetime=savetime)
    print('... Done')



def davis2hdf5_dirbase(dirbase, use_chunks, savedir=None, header='B', scale=1000., fps=1., mode='piv',
                       start=0, end=None, savetime=False, step=1):
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

    datadirs = glob.glob(os.path.join(dirbase + '/*'))
    print(dirbase, len(datadirs))
    for datadir in tqdm(datadirs, desc='datadir'):
        # try:
        davis2hdf5(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps, mode=mode,
                   start=start, end=end, savetime=savetime, step=step)
        # except:
        #     print("Error: ", datadir)
        #     print("... Probably, this is not a data directory. Skipping...")
        #     continue
    print('... Done')

def davis2hdf5(datadir, use_chunks, savedir=None, savepath=None, header='B', scale=1000., chunks=None, fps=1., mode='piv',
               start=0, end=None, h5name=None, savetime=False, step=1):
    if mode == 'piv':
        davis2hdf5_piv(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps,
                       start=start, end=end, h5name=h5name, savepath=savepath, savetime=savetime, step=step)
    elif mode == 'stb':
        davis2hdf5_stb(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps,
                       start=start, end=end, h5name=h5name, savetime=savetime, step=step)
    else:
        raise ValueError('... mode must be either \'piv \' or \'stb\'')

def davis2hdf5_piv(datadir, use_chunks, savedir=None, savepath=None, header='B', scale=1000., chunks=None, fps=1.,
                   start=0, end=None, h5name=None, savetime=False ,step=1):
    """
     Convert multiple DaVis output (PIV) into a hdf5 file


     Parameters
     ----------
     dirbase: str
     savedir

     Returns
     -------

     """
    davis_dpaths = glob.glob(datadir + '/%s*' % header)
    davis_dpaths = natural_sort(davis_dpaths)
    davis_dpaths = davis_dpaths[start:end:step]

    duration = len(davis_dpaths)
    for t, dpath in enumerate(tqdm(davis_dpaths)):
        with open(dpath, 'r') as fyle:
            xlist, ylist, ulist, vlist = [], [], [], []
            lines = fyle.readlines()

            # File format changed since DaVis 10.1
            if lines[0].__contains__("DaVis;"): # DaVis 10.1
                delimitter = ";"
            else: # default for versions older than DaVis 10.1
                delimitter = " "

            if delimitter == " ":
                height, width = int(lines[0].split(delimitter)[4]), int(lines[0].split(delimitter)[5])
            else:
                height, width = int(lines[0].split(delimitter)[3]), int(lines[0].split(delimitter)[4])
            shape = (height, width)
            for i, line in enumerate(lines):
                if i==0:
                    if line.__contains__("\"Position\"%s\"mm\"" % delimitter):
                        scale = 1.
                        pos_unit = 'mm'
                    else:
                        pos_unit = 'px'
                    if line.__contains__("\"velocity\"%s\"m/s\"" % delimitter):
                        vscale = 1000.
                        vel_unit = 'm/s'
                    elif line.__contains__("\"displacement\"%s\"pixel\"" % delimitter):
                        vscale = scale * fps
                        vel_unit = 'px/frame'
                    else:
                        # vscale = 1.
                        # vel_unit = '????'
                        vscale = scale * fps
                        vel_unit = 'px/frame'
                    if t==0:
                        print('\n Units of Position and Velocity: ' + pos_unit, vel_unit)
                        if vel_unit == 'px/frame':
                            print('scale (mm/px), frame rate(fps): %.5f, %.1f' % (scale, fps))
                        elif vel_unit == 'm/s':
                            print('... velocity gets converted to mm/s')
                if i > 0:
                    if delimitter == " ":
                        line = line.replace(',', '.').split()
                    else:
                        line = line.split(delimitter)
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
    if h5name is not None:
        dirname = h5name
    if savepath is None:
        savepath = os.path.join(os.path.join(savedir, 'davis_piv_output'), dirname)

    data2write = {}
    data2write['ux'] = udata_d[0, ...]
    data2write['uy'] = udata_d[1, ...]
    data2write['x'] = x_arr
    data2write['y'] = y_arr
    data2write['conversion_factors'] = {'scale': scale, 'fps': fps}
    if use_chunks:
        chunks = udata_d.shape[1:-1] + (1, )
    else:
        chunks = None
    if savetime:
        dt = 1./fps
        time = np.arange(udata_d.shape[-1]) * dt
        data2write['t'] = time

    data2write['date'] = datetime2int(datetime.datetime.now())

    write_hdf5_dict(savepath, data2write, chunks=chunks)
    print('... Done')


def datetime2int(now):
    return int(now.year*1e8 + now.month*1e6 + now.day*1e4 + now.hour*1e2 + now.minute*1e0)

def davis2hdf5_stb(datadir, use_chunks, savedir=None, savepath=None, header='B', scale=1000., chunks=None, fps=1.,
                   start=0, end=None, h5name=None, savetime=False, step=1):
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
    davis_dpaths = davis_dpaths[start:end:step]

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
    savepath = savedir + '/davis_stb_output/' + dirname
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
    data2write['conversion_factors'] = {'scale': scale, 'fps': fps}
    if use_chunks:
        chunks = udata_d.shape[1:-1] + (1, )
    else:
        chunks = None

    if savetime:
        dt = 1./fps
        time = np.arange(udata_d.shape[-1]) * dt
        data2write['t'] = time
    data2write['date'] = datetime2int(datetime.datetime.now())

    write_hdf5_dict(savepath, data2write, chunks=chunks)
    print('... Done')

# SAVING SCALAR DATA
## Job assigning functions
def davis2hdf5_dirbase_scalar(dirbase, use_chunks, savedir=None, header='B', scale=1., fps=1., mode='piv',
                              start=0, end=None, step=1, h5name='vorticity', savetime=False):
    if savedir is None:
        if dirbase[-1] == '/':
            savedir = os.path.split(dirbase[:-1])[0]
        else:
            savedir = os.path.split(dirbase)[0]

    datadirs = glob.glob(os.path.join(dirbase, '*'))

    for datadir in tqdm(datadirs, desc='datadir'):
        # try:
        davis2hdf5_scalar(os.path.join(datadir), use_chunks, savedir=savedir, header=header, scale=scale, fps=fps, mode=mode,
                   start=start, end=end, h5name=h5name)
        # davis2hdf5_scalar(os.path.join(datadir, 'Export'), use_chunks, savedir=savedir, header=header, scale=scale, fps=fps, mode=mode,
        #            start=start, end=end, h5name=h5name)
        # except:
        #     print("Error: ", datadir)
        #     print("... Probably, this is not a data directory. Skipping...")
        #     continue
    print('... Done')

def davis2hdf5_scalar(datadir, use_chunks, arr_shape=None,
                      savedir=None, savepath=None, header='B', scale=1.,
                      fps=1., mode='stb',
                      start=0, end=None, h5name=None):
    if mode == 'piv':
        davis2hdf5_piv_scalar(datadir, use_chunks, savedir=savedir, header=header, scale=scale, fps=fps,
                       start=start, end=end, h5name=h5name, savepath=savepath)
    elif mode == 'stb':
        davis2hdf5_stb_scalar(datadir, use_chunks, arr_shape=arr_shape, savedir=savedir, header=header, scale=scale, fps=fps,
                       start=start, end=end, h5name=h5name)
    else:
        raise ValueError('... mode must be either \'piv \' or \'stb\'')

## Job processing functions for scalar data
def davis2hdf5_piv_scalar(datadir, use_chunks, savedir=None, header=None, scale=1., fps=1.,
                          start=0, end=None, step=1, h5name=None, savepath=None, **kwargs):
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
    davis_dpaths = davis_dpaths[start:end:step]

    duration = len(davis_dpaths)
    for t, dpath in enumerate(tqdm(davis_dpaths)):
        with open(dpath, 'rt') as fyle:
            slist = []
            lines = fyle.readlines()
            line = lines[0].replace(';', ' ').split()
            height, width = int(line[3]), int(line[2])
            shape = (height, width)
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                line_list = line.replace(';', ' ').replace('\n', '').split()
                linedata = [float(value) for value in line_list]
                slist.append(linedata)
        if t == 0:
            shape_d = shape + (duration,)
            sdata_d = np.empty(shape_d)
        sdata_d[..., t] = np.asarray(slist)

    # Path where hdf5 is stored
    if datadir[-1] == '/':
        savedir_default, dirname = os.path.split(datadir[:-1])
    else:
        savedir_default, dirname = os.path.split(datadir)
        if dirname == 'Export':
            savedir_default, dirname = os.path.split(savedir_default)
    if savedir is None:
        savedir = savedir_default
    savepath = savedir + '/davis_piv_output/' + dirname + '_' + h5name
    if start != 0 or end is not None:
        if end is None:
            end = len(davis_dpaths) - 1
        savepath += '%05d_%05d' % (start, end)

    data2write = {}
    data2write[f'{h5name}'] = sdata_d
    if use_chunks:
        chunks = sdata_d.shape[:-1] + (1,)
    else:
        chunks = None

    # if savetime:
    #     dt = 1. / fps
    #     time = np.arange(udata_d.shape[-1]) * dt
    #     data2write['t'] = time
    # data2write['date'] = datetime2int(datetime.datetime.now())

    write_hdf5_dict(savepath, data2write, chunks=chunks)
    print('... Done')


def davis2hdf5_stb_scalar(datadir, use_chunks, arr_shape=None, savedir=None, savepath=None, header='B', scale=1.,
                          chunks=None, fps=1.,
                   start=0, end=None, h5name=None, **kwargs):
    """
    Saves the Davis-generated data of a scalar field into a single hdf5 file
    ... Intended for a 3D scalar field. For 2D, use davis2hdf5_piv_scalar
    ... The original data is assumed to be in the txt or csv format.
    ...... if txt, this function requires the shape of the array of the scalar field (height, width, depth)
    ...... csv version is not implemented yet but this could be a standard as it would not require arr_shape

    Parameters
    ----------
    datadir: str, a path to the directory where DaVis exported a scalar data
    use_chunks: bool, If True, you pass a chunk size to h5py when the data is stored in hdf5.
        ... You should almost always set this true for optimal reading/writing of the data later.
        ... The recommended chunk size is (height, width, depth) as one often reads the data per frame.
    arr_shape: 1d array-like of integers, the shape of the scalar data. default is None
        ... This argument should not be necessary if one decides to store the data in csv; however, it is if you save the data in txt.
        ... The reason is that DaVis stores the 3D data as a 2D image, which makes it impossible to reconstruct the original 3D array without
        knowing the dimensions of the array.
    savedir: str, a path to the directory where the generated hdf5 file is stored. If not given, it will create a directory under datadir.
    savepath: str, a path of the generated hdf5 file is stored
    header: str, default: "B", the header of the DaVis generated files
    scale:
    chunks
    fps
    start
    end
    h5name
    kwargs

    Returns
    -------

    """
    davis_dpaths = glob.glob(datadir + '/%s*' % header)
    davis_dpaths = natural_sort(davis_dpaths)
    davis_dpaths = davis_dpaths[start:end]
    if davis_dpaths[0][-3:] == 'txt':
        print("... davis2hdf5_stb_scalar: You must specify the array dimensions of the volume in which the scalar field is extracted.\n"
              "This is because DaVis output reduces a 3D array into a 2D array if you chose to ouput in the ASCII format.")
        if arr_shape is None:
            raise ValueError("... Pass the array shape of the measurement volume through -shape. e.g. -shape 40 80 35")
    duration = len(davis_dpaths)
    height, width, depth = arr_shape
    pdata = np.empty((height, width, depth, duration))
    for t, dpath in enumerate(tqdm(davis_dpaths)):
        pdata_raw = np.loadtxt(open(dpath, 'rb'), delimiter=';', skiprows=1)
        for j in range(width):
            for k in range(depth):
                pdata[:, j, k, t] = pdata_raw[k*height:(k+1)*height, j]
                # keep = pdata[:, j, k, t] == 0.0
                # pdata[~keep, j, k, t] -= 2000

    # Path where hdf5 is stored
    if datadir[-1] == '/':
        savedir_default, dirname = os.path.split(datadir[:-1])
    else:
        savedir_default, dirname = os.path.split(datadir)
    if savedir is None:
        savedir = savedir_default
    if h5name is not None:
        dirname = h5name
    if savepath is None:
        savepath = savedir + '/davis_stb_output_pressure/' + dirname

    data2write = {}
    data2write['pressure'] = pdata

    if use_chunks:
        chunks = pdata.shape[:-1] + (1,)
    else:
        chunks = None

    write_hdf5_dict(savepath, data2write, chunks=chunks)
    print('... Done')

## Job processing for decay experiments

def davis2hdf5_piv_decay(datadir, fps_list, frame_list, use_chunks, savedir=None, savepath=None, header='B', scale=1000., fps=1.,
                   start=0, end=None, h5name=None, savetime=False ,step=1, cinepath=None):
    """
     Convert multiple DaVis output (PIV) into a hdf5 file


     Parameters
     ----------
     dirbase: str
     savedir

     Returns
     -------

     """
    davis_dpaths = glob.glob(os.path.join(datadir, f'{header}*.txt'))
    nDavisOutput = len(davis_dpaths)
    davis_dpaths = natural_sort(davis_dpaths)
    davis_dpaths = davis_dpaths[start:end:step]

    if cinepath is None:
        nFrames = sum(frame_list)
        # Error handling
        if nFrames != nDavisOutput:
            print(f'(Number of Davis Output Files, Sum of frame_list), ({nDavisOutput}, {nFrames})')
            raise ValueError('Sum of frame_list must match the number of Davis Output in the given directory')
        elif len(fps_list) != len(frame_list):
            raise ValueError('fps_list and frame_list must have the same length!')
    else:
        cc = cine.Cine(cinepath)
        tt = cc.get_time_list()
        cc.close()
        dt = np.diff(tt)
        frame_rates = list(1./dt)
        nFrames = len(dt)
        # Error handling
        if nFrames != nDavisOutput:
            print(f'(Number of Davis Output Files, Sum of frame_list), ({nDavisOutput}, {nFrames})')
            print('Sum of frame_list must match the number of Davis Output in the given directory')
            isDataPyramid = input('Did you process this data with the pyramid algorithm? (y/n)')
            if isDataPyramid.lower() in ['y', 'yes']:
                pass
            else:
                raise ValueError('Aborting')
        
    if cinepath is None:
        frame_rates = []
        tt = [0]
        for fps, nFrame in zip(fps_list, frame_list):
            frame_rates += [fps] * nFrame
            tt += list(np.arange(1, nFrame + 1, ) / fps + tt[-1])
        frame_rates = frame_rates[start:end:step]
        tt = tt[:-1]
    duration = len(davis_dpaths)

    for t, dpath in enumerate(tqdm(davis_dpaths)):
        fps = frame_rates[t]
        with open(dpath, 'r') as fyle:
            xlist, ylist, ulist, vlist = [], [], [], []
            lines = fyle.readlines()

            # File format changed since DaVis 10.1
            if lines[0].__contains__("DaVis;"): # DaVis 10.1
                delimitter = ";"
            else: # default for versions older than DaVis 10.1
                delimitter = " "

            if delimitter == " ":
                height, width = int(lines[0].split(delimitter)[4]), int(lines[0].split(delimitter)[5])
            else:
                height, width = int(lines[0].split(delimitter)[3]), int(lines[0].split(delimitter)[4])
            shape = (height, width)
            for i, line in enumerate(lines):
                if i==0:
                    if line.__contains__("\"Position\"%s\"mm\"" % delimitter):
                        scale = 1.
                        pos_unit = 'mm'
                    else:
                        pos_unit = 'px'
                    if line.__contains__("\"velocity\"%s\"m/s\"" % delimitter):
                        vscale = 1000.
                        vel_unit = 'm/s'
                    elif line.__contains__("\"displacement\"%s\"pixel\"" % delimitter):
                        vscale = scale * fps
                        vel_unit = 'px/frame'
                    else:
                        vscale = scale * fps
                        vel_unit = 'px/frame'
                    if t==0:
                        print('\n Units of Position and Velocity: ' + pos_unit, vel_unit)
                        if vel_unit == 'px/frame':
                            print('scale (mm/px), frame rate(fps): %.5f, %.1f' % (scale, fps))
                        elif vel_unit == 'm/s':
                            print('... velocity gets converted to mm/s')
                if i > 0:
                    if delimitter == " ":
                        line = line.replace(',', '.').split()
                    else:
                        line = line.split(delimitter)
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
    if h5name is not None:
        dirname = h5name
    if savepath is None:
        savepath = os.path.join(os.path.join(savedir, 'davis_piv_output'), dirname)

    data2write = {}
    data2write['ux'] = udata_d[0, ...]
    data2write['uy'] = udata_d[1, ...]
    data2write['x'] = x_arr
    data2write['y'] = y_arr
    data2write['conversion_factors'] = {'scale': scale, 'fps': fps}
    data2write['t'] = tt[:udata_d.shape[-1]]
    if use_chunks:
        chunks = udata_d.shape[1:-1] + (1, )
    else:
        chunks = None
    if savetime:
        dt = 1./fps
        time = np.arange(udata_d.shape[-1]) * dt
        data2write['t'] = time

    data2write['date'] = datetime2int(datetime.datetime.now())

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


def main(args):
    if args.mode == 'custom':
        print('... Custom... make hdf5 files for the project directories. This may fail for many cases.')
        print('project directory:', args.dirbase)
        # target_dirs_1 = glob.glob(os.path.join(args.dirbase, '210120_tlf*'))
        # target_dirs_2 = glob.glob(os.path.join(args.dirbase, '210121_tlf*'))
        # target_dirs = target_dirs_1 + target_dirs_2
        target_dirs = glob.glob(os.path.join(args.dirbase, 'phi55mm*'))
        for i, tdir in tqdm(enumerate(target_dirs)):
            print(i, ', processing', os.path.split(tdir)[1])
            try:
                subdir = [os.path.join(tdir, name) for name in os.listdir(tdir) if os.path.isdir(os.path.join(tdir, name)) ][0]
                input_dir = os.path.join(subdir, 'TR_PIV_MPd(1x32x32_75%ov)\\Export')
                # savepath = 'D:\\userdata\\hydrofoil_circulation\\' + '\\davis_piv_output\\' + os.path.split(tdir)[1]
                savepath = 'D:\\userdata\\PIV_20210128_confinement transition\\davis_piv_output\\' + os.path.split(tdir)[1]
                if os.path.exists(input_dir) and not os.path.exists(savepath + '.h5'):
                    davis2hdf5(input_dir, args.chunks, scale=args.scale, fps=args.fps, mode='piv',
                               start=args.start, end=args.end, savepath=savepath, savetime=args.save_time)
                elif os.path.exists(savepath + '.h5'):
                    print('... a corresponding h5 file already exists. skipping...')
                else:
                    print('... Export folder is missing. skipping...')

                # print(i, input_dir, os.path.exists(input_dir))
            except:
                print('... skipping...')
                continue
    elif args.mode == 'decay':
        if args.dirbase is None:
            print('... Making a hdf5 file for the following directory: ' + args.dir)
            davis2hdf5_piv_decay(args.dir, args.fps_list, args.frame_list, args.chunks, scale=args.scale, fps=args.fps,
                       start=args.start, end=args.end, savetime=args.save_time, step=args.step, cinepath=args.cine)
        else:
            target_dirs = sorted(glob.glob(os.path.join(args.dirbase, '*')))
            for i, tdir in tqdm(enumerate(target_dirs)):
                print('... Making a hdf5 file for the following directory: ' + tdir)
                davis2hdf5_piv_decay(tdir, args.fps_list, args.frame_list, args.chunks, scale=args.scale, fps=args.fps,
                                     start=args.start, end=args.end, savetime=args.save_time, step=args.step, cinepath=args.cine)

    else: # STANDARD PROCESSING SCHEMES
        if args.qty == 'velocity':
            if args.dir is None:
                print('... Making hdf5 files for directories under ' + args.dirbase)
                davis2hdf5_dirbase(args.dirbase, args.chunks, scale=args.scale, fps=args.fps, mode=args.mode,
                                   start=args.start, end=args.end, savetime=args.save_time, step=args.step)
            else:
                print('... Making a hdf5 file for the following directory: ' + args.dir)
                davis2hdf5(args.dir, args.chunks, scale=args.scale, fps=args.fps, mode=args.mode,
                           start=args.start, end=args.end, savetime=args.save_time, step=args.step)
        elif args.qty == 'scalar':
            if args.dir is None:
                print('... Making hdf5 files for directories under ' + args.dirbase)
                davis2hdf5_dirbase_scalar(args.dirbase, args.chunks, scale=args.scale, fps=args.fps, mode=args.mode,
                                   start=args.start, end=args.end, h5name=args.h5name)
            else:
                print('... Making a hdf5 file for the following directory: ' + args.dir)
                davis2hdf5_scalar(args.dir, args.chunks, arr_shape=args.shape, scale=args.scale, fps=args.fps, mode=args.mode,
                           start=args.start, end=args.end, h5name=args.h5name)

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
    parser.add_argument('-mode', '--mode', help='Type of the experiment: PIV, STB? Choose from [piv, stb, custom, decay]', type=str,
                        default='piv')
    parser.add_argument('-start', '--start', help='index which specifies the range of data data to be used [start, end)'
                                                  'default: 0', type=int,
                        default=0)
    parser.add_argument('-end', '--end', help='index which specifies the range of data data to be used [start, end)'
                                                  'default: None', type=int,
                        default=None)
    parser.add_argument('-step', '--step', help='creates a h5 using start, start+step, start+2*step, ..., end'
                                              'default: 1', type=int,
                        default=1)
    parser.add_argument('-save_time', '--save_time', help='Adds a time based on the input frame rate', action='store_true')
    parser.add_argument('-qty', '--qty', help='Quantity of data stored in the input directory- choose from [velocity, pressure]',
                        type=str, default='velocity')
    parser.add_argument('-shape', '--shape', nargs='+', type=int,
                        help='Required if you are storing scalar data AND chose to export it in txt on DaVis')
    parser.set_defaults(feature=False)
    parser.add_argument('-h5name', help='Name of a scalar quantity whose image is being read from DaVis', type=str,
                        default='vorticity')
    # decay mode
    parser.add_argument('-fps_list', '--fps_list', type=int, nargs='+',
                        help='List of frame rates used for decay PIV experiments')
    parser.add_argument('-frame_list', '--frame_list', type=int, nargs='+',
                        help='List of number of frames used for decay PIV experiments')
    parser.add_argument('-cine', '--cine', type=str,
                        help='A path to a corresponding cinefile- used to extract the frame rate (if mode==decay')

    args = parser.parse_args()

    main(args)