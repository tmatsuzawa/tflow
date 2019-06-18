"""
Make an organized hdf5 file out of PIVLab txt outputs and other measurements



Nomenclature:
cinedir: directory where cine is stored.
   |    e.g.- /Volumes/labshared4/takumi/old_data/sample_piv_cine/
   |
basedir: directory where parentdirs are housed. All PIVLab outputs are sorted by (Inter. Area, Dt, step)
   |    e.g.- /Volumes/labshared4/takumi/old_data/sample_piv_cine/pivlab_outputs
   |
parentdir: directory where PIVLab outputs of many cines (but with the same PIV parameters) are stored
   |    e.g.- .../PIV_W16_step1_data
   |
datadir: directory where PIV outputs are stored.
   |    e.g.- .../cinename_no_extension
   |    ... In datadir, one finds D000000.txt, D000001.txt, etc. as well as piv_settings
   |
piv_settings and piv_settings_2: directories where piv-related parameters are stored.
        ... The information in both dirs are identical but format is different. Choose your preferred format to use.

"""


import argparse
import glob
import os
import numpy as np
import h5py
from tqdm import tqdm
from scipy import integrate, interpolate
import sys
import fapm.formatarray as fa
import fapm.cine_local.cine.cine as cine  # use local cine package

def read_exp_settings(cinedir):
    exp_settings_path= cinedir + '/setup.txt'
    fyle = open(exp_settings_path, 'r')
    lines = []
    for i, line in enumerate(fyle.read().split('\n')):
        lines.append(line)

    params = lines[0].split('\t')
    values_tmp = lines[1].split('\t')
    values = []
    for i, value_tmp in enumerate(values_tmp):
        try:
            value = float(value_tmp)
        except ValueError:
            value = value_tmp
            pass
        values.append(value)
    piv_settings = dict(zip(params, values))
    return piv_settings

def read_piv_settings(datadir):
    """
    Returns used piv-related parameters in a dictionary when a path to pivlab output directory is provided

    Parameters
    ----------
    datadir: str
            location of directory where piv outputs are stored

    Returns
    -------
    piv_settings: dict
            keys are name of piv parameters
            values are parameter values (either str or float).

    """

    piv_settings_path = datadir + '/piv_settings/piv_settings.txt'
    fyle = open(piv_settings_path, 'r')
    lines = []
    for i, line in enumerate(fyle.read().split('\n')):
        if i > 1:
            lines.append(line)

    params = lines[0][:-1].split('\t')
    values_tmp = lines[1][:-1].split('\t')
    values = []
    for i, value_tmp in enumerate(values_tmp):
        try:
            value = float(value_tmp)
        except ValueError:
            value = value_tmp
            pass
        values.append(value)
    piv_settings = dict(zip(params, values))
    return piv_settings

def compute_form_no(stroke_length, orifice_d=25.6, piston_d=160., num_orifices=1):
    """
    Compute formation number (L/D) of a vortex ring
    Parameters
    ----------
    stroke_length: float
        stroke length of piston in mm
    orifice_d: float
        diameter of orfice in mm
    piston_d: float
        diameter of piston in mm
    num_orifices: int or float
        Number of orfices in a box

    Returns
    -------
    ld: float
        formation number of vortex ring
    """
    ld = (piston_d / orifice_d)**2 * stroke_length / orifice_d / float(num_orifices)
    return ld

def return_piston_tracking_data(cinedir, cinename):
    print 'Looking for piston measurements'
    pistontrack_dir = cinedir + '/pistontracking'
    pistontrack_data_path = os.path.join(pistontrack_dir, cinename + '_position_data.txt')
    pistontrack_data_path2 = os.path.join(pistontrack_dir, cinename[:-5] + '_position_data.txt')
    if os.path.exists(pistontrack_data_path):
        print '... Found!'
    elif os.path.exists(pistontrack_data_path2):
        print '... Found!'
        pistontrack_data_path = pistontrack_data_path2
    else:
        print '... NOT found... Expected file path is:'
        print pistontrack_data_path
        return [np.nan], [np.nan]

    piston_data = np.loadtxt(pistontrack_data_path)
    time, position = piston_data[0], piston_data[1]
    return time, position

def return_piston_measurements(time, position, freq, shiftper=0.25,
                               orifice_d=25.6, piston_d=160., num_orifices=8):
    """
    Returns effective velocity, average velocity, stroke length, and L/D when time, position, expected frequency is provided
    Parameters
    ----------
    time
    position
    freq
    shiftper

    Returns
    -------

    """
    if any(np.isnan(time)) or any(np.isnan(position)):
        return np.nan, np.nan, np.nan, np.nan, [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan]
    else:
        # Roll data for plot
        time_v, vel = compute_velocity_simple(time, position)

        # Average data over periods
        time_short, position_short, position_short_std = fa.get_average_data_from_periodic_data(time, position, freq=freq)
        time_short_v, vel_short, vel_short_std, \
        time_chunks, vel_chunks, time_chunks_int, vel_chunks_int = fa.get_average_data_from_periodic_data(time, vel,
                                                                                                          freq=freq,
                                                                                                          returnChunks=True)

        # Roll averaged data for plot and computation
        position_short = np.roll(position_short, int(len(position_short) * shiftper))
        position_short_std = np.roll(position_short_std, int(len(position_short_std) * shiftper))
        vel_short = np.roll(vel_short, int(len(vel_short) * shiftper))
        vel_short_std = np.roll(vel_short_std, int(len(vel_short_std) * shiftper))

        # Calculate effective velocity
        ## Interpolate the data set (time_short and v_short)
        time_short_int, v_short_int = interpolate_1Darrays(time_short, vel_short)

        ## Get two points where y=0
        ### Method: Detect a sign flip
        # Find a minimum value and its index
        v_min_ind, v_min = np.argmin(v_short_int), np.amin(v_short_int)
        # Split an array into two parts using the minimum value
        v_short_1, v_short_2 = v_short_int[:v_min_ind], v_short_int[v_min_ind:]
        # Detect a sign flip of the left array
        signflip_indices = fa.detect_sign_flip(v_short_1)
        v_short_left_ind = signflip_indices[-1]
        # Detect a sign flip of the right array
        signflip_indices = fa.detect_sign_flip(v_short_2)
        v_short_right_ind = len(v_short_1) + signflip_indices[0]

        # Compute effective velocity
        veff = compute_eff_velocity(v_short_int, time_short_int, v_short_left_ind, v_short_right_ind)
        vavg = compute_mean_velocity(v_short_int, time_short_int, v_short_left_ind, v_short_right_ind)

        ## Actual stroke length
        sl = np.max(position_short) - np.min(position_short)
        ld = compute_form_no(sl, orifice_d=orifice_d, piston_d=piston_d, num_orifices=num_orifices)
        return veff, vavg, sl, ld, time_short, position_short, position_short_std, time_short_v, vel_short, vel_short_std

def get_time_for_pivoutput(time_cine, Dt, step, mode='left'):
    """
    Returns a 1D numpy array about time for a given piv parameters (Dt and step)
    Parameters
    ----------
    time_cine: list
        time recorded in a cine file
    Dt: int/float
        spacing between imageA and imageB in PIV processing
    step: int/float
        spacing between successive image pairs.
    mode: str, default: 'left'
        There is ambiguity in how to assign time for a quantity (say u for example).
        Should u be an instantaneous measurement at t=t_imageA (moment when imageA was taken) ?
        'mode'=='left': u(t_imageA)
        'mode'=='middle': u( (t_imageA + t_imageB) / 2)
        'mode'=='right': u(t_imageB)

    Returns
    -------
    time: 1d numpy array
        time array suitable for piv outputs
    """
    # If cine contains odd number of images, ignore the unused, last data point in time.
    if len(time_cine) % 2 == 1:
        time_cine = np.delete(time_cine, -1)

    time = time_cine[:-int(Dt)][::int(step)]
    # time = np.delete(time, range(len(time)-int(Dt), len(time)))
    if mode == 'left':
        return time
    elif mode == 'middle':
        return time + (Dt / 2.)
    elif mode == 'right':
        return time + Dt

def get_frame2sec(time_cine, Dt, step):
    """
    Returns time increments between imageA and imageB for ALL pairs in a cine

    This is useful for converting raw data in units of px and frame to mm and sec.
    i.e.-
        ux_mms = ux_pxframe * px2mm /frame2sec

    Parameters
    ----------
    time_cine: list
        time recorded in a cine file
    Dt: int/float
        spacing between imageA and imageB in PIV processing
    step: int/float
        spacing between successive image pairs.

    Returns
    -------
    frame2sec: 1D numpy arr
        time increments (Delta t) between imageA and imageB for ALL pairs in a cine
    """
    # If cine contains odd number of images, ignore the unused, last data point in time.
    if len(time_cine) % 2 == 1:
        time_cine = np.delete(time_cine, -1)

    timeA = time_cine[::int(step)]
    timeB = time_cine[int(Dt)::int(step)]
    if len(timeB) != len(timeA):
        timeA = np.delete(timeA, -1)
    frame2sec = timeB - timeA
    return frame2sec

def get_sec2frame(time_cine, Dt, step):
    """
    Returns the inverse of frame2sec obtained by get_frame2sec

    Parameters
    ----------
    time_cine: list
        time recorded in a cine file
    Dt: int/float
        spacing between imageA and imageB in PIV processing
    step: int/float
        spacing between successive image pairs.

    Returns
    -------
    sec2frame: 1D numpy arr
        inverse of frame2sec
    """
    frame2sec = get_frame2sec(time_cine, Dt, step)
    return 1. / frame2sec

############ USEFUL FUNCTIONS ############

def get_velocity_extrema_near_lift(velocity, plot=False):
    """
    Returns indices of array where the sign of elements change near the minimum value of the array
    Parameters
    ----------
    velocity
    plot

    Returns
    -------

    """
    # Find a minimum value and its index
    v_min_ind, v_min = np.argmin(velocity), np.amin(velocity)
    # Split an array into two parts using the minimum value
    v1, v2 = velocity[:v_min_ind], velocity[v_min_ind:]
    # Detect a sign flip of the left array
    signflip_indices = detect_sign_flip(v1)
    v_mean_left_ind = signflip_indices[-1]
    # Detect a sign flip of the right array
    signflip_indices = detect_sign_flip(v2)
    v_mean_right_ind = len(v1) + signflip_indices[0]

    return v_mean_left_ind, v_mean_right_ind

def detect_sign_flip(arr):
    """
    Returns indices of an 1D array where its elements flip the sign
    Parameters
    ----------
    arr

    Returns
    -------
    indices: tuple

    """
    arr = np.array(arr)
    arrsign = np.sign(arr)
    signchange = ((np.roll(arrsign, 1) - arrsign) != 0).astype(int)
    indices = np.where(signchange == 1)
    return indices

def compute_eff_velocity(vel, time, ind1, ind2):
    """ Computes effective velocity

    Parameters
    ----------
    vel
    ind1
    ind2

    Returns
    -------

    """
    if not len(vel) == len(time):
        print 'velocity array and time array have different sizes!... Continue computing effective velocity.'

    # Clip an array for computation
    vel = vel[ind1-1:ind2]
    time = time[ind1-1:ind2]
    # Prepare velocity squared array
    vel2 = [v**2 for v in vel]
    # Integrate (composite trapezoid)
    vel_int = integrate.trapz(vel, time)
    vel2_int = integrate.trapz(vel2, time)
    # Compute effective velocity
    v_eff = vel2_int / vel_int
    return v_eff

def compute_mean_velocity(vel, time, ind1, ind2):
    """ Computes mean velocity

    Parameters
    ----------
    vel
    ind1
    ind2

    Returns
    -------

    """
    if not len(vel) == len(time):
        print 'velocity array and time array have different sizes!... Continue computing mean velocity.'

    # Clip an array for computation
    vel = vel[ind1-1:ind2]
    vel_avg = np.nanmean(vel)
    return vel_avg

def compute_velocity_simple(time, pos):
    """
    Compute velocity given that position and time arrays are provided
    - Use np.gradient should be enough for most of the purposes, but this method is much simpler, and more versatile
    - This does not care if time array is not evenly spaced.

    Parameters
    ----------
    pos : 1d array with length N
    time : 1d array with length N

    Returns
    -------
    velocity : 1d array with length N-1
    time_new : 1d array with length N-1
    """
    if any(np.isnan(time)) or any(np.isnan(pos)):
        return [np.nan], [np.nan]
    else:
        pos, time = np.array(pos), np.array(time)
        delta_pos = np.ediff1d(pos)
        delta_time = np.ediff1d(time)

        time_new = (time[1:] + time[:-1]) / 2.

        velocity = delta_pos / delta_time
        return time_new, velocity

def interpolate_1Darrays(x, data, xint=None, xnum=None, xmax=None, xmin=None, mode='cubic'):
    """
    Conduct interpolation on a 1d array (N elements) to generate a 1d array (xnum elements)
    One can also specify x-spacing (xint) instead of the number of elements of the interpolated array
    Parameters
    ----------
    x
    data
    xint
    xnum
    mode

    Returns
    -------

    """
    if xmax is None:
        xmax = np.max(x)
    if xmin is None:
        xmin = np.min(x)
    if xmax > np.max(x):
        x = np.concatenate([x, [xmax]])
        data = np.concatenate([data, [data[-1]]])
    if xmin < np.min(x):
        x = np.concatenate([[xmin], x])
        data = np.concatenate([ [data[0]], data])

    if xint is None and xnum is None:
        # Default is generate 10 times more data points
        xnum = len(x)*10
        xint = np.abs(xmax-xmin)/float(xnum)
    elif xint is None and xnum is not None:
        xint = np.abs(xmax - xmin) / float(xnum)
    elif xint is not None and xnum is not None:
        print 'WARNING: Both x interval and xnum were provided! Ignoring provided x interval...'
        xint = np.abs(xmax - xmin) / float(xnum)

    xnew = np.arange(xmin, xmax, xint)
    # check xnew has a length xnum
    if xnum is not None:
        if len(xnew) > xnum:
            excess = len(xnew) - xnum
            xnew = xnew[:-excess]
    f = interpolate.interp1d(x, data, kind=mode)
    datanew = f(xnew)
    return xnew, datanew

##########################################

def pivlab2hdf5_cine(cinepath, header='PIVlab', overwrite=False):
    cinedir, cinename = os.path.split(cinepath)
    cinename_no_ext = cinename[:-5]
    savedir = cinedir + '/hdf5data'

    parent_datadirs = glob.glob(cinedir + '/pivlab_outputs/*')

    for parent_datadir in parent_datadirs:
        datadirs = glob.glob(parent_datadir + '/' + header + '*')
        for datadir in datadirs:
            if cinename_no_ext in datadir:
                pivlab2hdf5(datadir, savedir=savedir, overwrite=overwrite)

    print '...... Done'
    return savedir

def pivlab2hdf5_basedir(basedir, header='PIV', overwrite=False):
    cinedir, basedirname = os.path.split(basedir)
    savedir = cinedir + '/hdf5data'
    parentdirs = glob.glob(basedir + '/' + header + '*')
    for parentdir in parentdirs:
        pivlab2hdf5_parentdir(parentdir, overwrite=overwrite)
    print '......... Done'
    return savedir

def pivlab2hdf5_parentdir(parentdir, header='PIVlab', overwrite=False):
    basedir, pdirname = os.path.split(parentdir)
    cinedir, basedirname = os.path.split(basedir)
    savedir = cinedir + '/hdf5data'
    datadirs = glob.glob(parentdir + '/' + header + '*')
    for datadir in datadirs:
        # print '... processing ' + os.path.split(datadir)[1]
        pivlab2hdf5(datadir, savedir=savedir, overwrite=overwrite)
    print '...... Done'
    return savedir

def pivlab2hdf5(dir, savedir=None, header='PIV_', overwrite=False):
    """
    For each cine, this stores all related informaton (ux, uy), piston tracking records,
    piv/experimental settings into a master hdf5 file

    Parameters
    ----------
    dir: str, name of the directory where pivlab outputs lie
    savedir: str, location where the master hdf5 file will be created
    header: str, header of dir. This is used for naming stuff, and to find the corresponding cine file.
    overwrite: bool, If True, it overwrite all data in hdf5 file. Default: False

    Returns
    -------
    savedir: str, location where the master hdf5 file will be created
    """
    # FILE ARCHITECTURE
    parentdir = os.path.dirname(dir)
    basedir = os.path.dirname(parentdir)
    cinedir = os.path.dirname(basedir)

    parentdirname = os.path.split(dir)[1]
    cinename_no_ext = parentdirname[parentdirname.find(header):]  # cinename (no extension) will be the name of the master hdf5 file
    cinepath = os.path.join(cinedir, cinename_no_ext) + '.cine'

    if savedir is None:
        savedir = parentdir + '/hdf5data'


    print 'Processing: %s' % os.path.split(dir)[1]

    # MAKE HDF5
    hdf5path = savedir + '/' + cinename_no_ext + '.h5'
    if not os.path.exists(savedir):
        print '... mkdir ', savedir
        os.makedirs(savedir)
    hdf5path_perhaps = savedir + '/' + cinename_no_ext[:-3] + '.h5'
    if os.path.exists(hdf5path_perhaps):
        cinename_no_ext = cinename_no_ext[:-3]
        cinepath = os.path.join(cinedir, cinename_no_ext) + '.cine'
        hdf5path = hdf5path_perhaps


    with h5py.File(hdf5path, 'a') as fyle:
        ################ /exp #############################
        # /exp.attrs: experimental parameters such as commanded velocity, commanded frequency etc.
        # /exp/...: subgroups store piston tracking measurements if available
        # Effective piston velocity and avg. piston velocity are also computed here
        ###################################################
        if not fyle.__contains__('exp'):
            exp = fyle.create_group('exp')
        else:
            exp = fyle['exp']
        # The root has experimental parameters as attributes
        exp_params = read_exp_settings(cinedir)
        for i, key in enumerate(exp_params.keys()):
            exp.attrs[key] = exp_params[key]

        ## Get time record in cine to insert to the master hdf5 later
        cc = cine.Cine(cinepath)
        time_cine = np.asarray(cc.get_time_list())

        # exp group also stores piston tracking data if available
        # Always check if experimental measurements are in hdf5
        t_piston, pos_piston = return_piston_tracking_data(cinedir, cinename_no_ext + '.cine')
        t_vel_piston, vel_piston = compute_velocity_simple(t_piston, pos_piston)
        veff, vavg, sl, ld, t_short, pos_short, pos_short_std, t_short_v, vel_short, vel_short_std \
            = return_piston_measurements(t_piston, pos_piston, freq=exp.attrs['frequency'])

        # Create datasets under /exp and attributes of /exp
        exp_datasetnames = ['t_piston', 'pos_piston', 't_v_piston', 'vel_piston',
                            't_short_piston', 'pos_short_piston', 'pos_std_short_piston',
                            't_v_short_piston', 'vel_short_piston', 'vel_std_short_piston',
                            'time']
        exp_dataset = [t_piston, pos_piston, t_vel_piston, vel_piston,
                       t_short, pos_short, pos_short_std,
                       t_short_v, vel_short, vel_short_std,
                       time_cine]
        exp_attrs = ['veff', 'vavg', 'stroke_length_measured', 'ld_measured']
        exp_attrvalues = [veff, vavg, sl, ld]
        for i, exp_datasetname in enumerate(exp_datasetnames):
            if not exp_datasetname in exp.keys():
                exp.create_dataset(exp_datasetname, data=exp_dataset[i])
        for i, exp_attr in enumerate(exp_attrs):
            if not exp_attr in exp.attrs.keys():
                exp.attrs[exp_attr] = exp_attrvalues[i]



        ################ /piv #############################
        # /piv.attrs: piv-related parameters such as Dt, step, Int. area_1, Number of passes, etc.
        # /piv/piv???: subgroups store pivlab outputs (ux, uy, omega) and spatial grids (x, y).
        # /piv/piv???: each subgroup corresponds to pivlab outputs with a given set of piv-related parameters
        ###################################################

        if not fyle.__contains__('piv'):
            piv = fyle.create_group('piv')
            # /piv has attributes: no_piv_data and software
            piv.attrs['no_piv_data'] = 0
            piv.attrs['software'] = 'MATLAB_PIVLab'
            piv.attrs['units'] = 'mm, s'
        else:
            piv = fyle['piv']

        # Load piv-related parameters (PIVLab parameters)
        piv_params = read_piv_settings(dir)

        # Check if the hdf5 file already stores this PIVLab output
        redundant_pivlab_outputs = False
        for subgroup_name in piv.keys():
            piv_paramnames_0 = piv[subgroup_name].attrs.keys()
            piv_paramvalues_0 = piv[subgroup_name].attrs.values()
            piv_params_0 = dict(zip(piv_paramnames_0, piv_paramvalues_0))
            # if piv_params_0 == piv_params:
            #     print '... this piv outputs already exist in the hdf5'
            #     redundant_pivlab_outputs = True
            #     break
            if piv_params.viewitems() <= piv_params_0.viewitems():
                print '... this piv outputs already exist in the hdf5'
                redundant_pivlab_outputs = True
                break

        if redundant_pivlab_outputs and not overwrite:
            print '... skipping'
            fyle.close()
        else:
            if not overwrite:
                # This piv outputs are not saved in the master hdf5 file! Save it to the master hdf5.
                # Naming convention: /piv/piv000, /piv/piv0001
                piv_working = piv.create_group('piv%03d' % piv.attrs['no_piv_data'])
            else:
                print '... Probably, datasets such as u and v exist under %s but overwrite data anyway' % piv[subgroup_name].name
                piv_working = piv[subgroup_name]

            # Attribute piv-related parameters to piv_working (group)
            for i, key in enumerate(piv_params.keys()):
                piv_working.attrs[key] = piv_params[key]
            # Also add data spacing (interrogation size / 2)
            W = piv_working.attrs['Int._area_%d' % piv_working.attrs['Nr._of_passes']]
            piv_working.attrs['W'] = W # smallest interrogation area size

            # Insert pivlab outputs
            datafiles = glob.glob(dir + '/*.txt')
            datafiles = sorted(datafiles)

            for i, datafile in enumerate(tqdm(datafiles, desc='... making nd arrays from PIVLab output')):
                # if i % 100 == 0:
                #     print '%d / %d' % (i, len(datafiles))
                data = np.loadtxt(datafile, delimiter=',', skiprows=3)

                xx, yy = data[:, 0], data[:, 1]
                ux, uy = data[:, 2], data[:, 3]
                omega = data[:, 4]

                if i == 0:
                    delta_y = np.diff(yy)[0]
                    delta_x = delta_y

                    ncols = int((np.max(xx) - np.min(xx)) / delta_x) + 1
                    nrows = int((np.max(yy) - np.min(yy)) / delta_y) + 1
                    shape_temp = (ncols, nrows)

                    xgrid, ygrid = xx.reshape(shape_temp).T, yy.reshape(shape_temp).T

                ux_grid, uy_grid, omega_grid = ux.reshape(shape_temp).T, uy.reshape(shape_temp).T, omega.reshape(
                    shape_temp).T

                if i == 0:
                    uxdata = np.zeros((nrows, ncols, len(datafiles)))
                    uydata = np.zeros((nrows, ncols, len(datafiles)))
                    omegadata = np.zeros((nrows, ncols, len(datafiles)))
                uxdata[..., i] = ux_grid
                uydata[..., i] = uy_grid
                omegadata[..., i] = omega_grid

            frame2sec = get_frame2sec(time_cine, piv_working.attrs['Dt'], piv_working.attrs['step'])
            piv_working_datasetnames = ['ux',
                                        'uy',
                                        'omega',
                                        'x', 'y',
                                        't',
                                        'deltat',
                                        'deltax',
                                        'deltay']
            piv_working_datasets = [uxdata * exp.attrs['scale'] / frame2sec,
                                    uydata * exp.attrs['scale'] / frame2sec,
                                    omegadata / frame2sec,
                                    xgrid * exp.attrs['scale'], ygrid * exp.attrs['scale'],
                                    get_time_for_pivoutput(time_cine, piv_working.attrs['Dt'], piv_working.attrs['step']),
                                    frame2sec,
                                    (xgrid[0, 1] - xgrid[0, 0]) * exp.attrs['scale'],
                                    (ygrid[1, 0] - ygrid[0, 0]) * exp.attrs['scale']]
            # Insert data into hdf5
            if not overwrite:
                for i, piv_working_datasetname in enumerate(piv_working_datasetnames):
                    if not piv_working_datasetname in piv_working.keys():
                        piv_working.create_dataset(piv_working_datasetname, data=piv_working_datasets[i])
                # Keep track how many piv data exists for a giveb cine
                piv.attrs['no_piv_data'] += 1
            else:
                for i, piv_working_datasetname in enumerate(piv_working_datasetnames):
                    try:
                        piv_working[piv_working_datasetname][...] = piv_working_datasets[i]
                    except KeyError:
                        piv_working.create_dataset(piv_working_datasetname, data=piv_working_datasets[i])
            fyle.close()

    print '... Done'
    return savedir


def main(args):
    if args.basedir is not None:
        print 'Use all available PIVLab outputs to create master hdf5 files'
        savedir = pivlab2hdf5_basedir(args.basedir, overwrite=args.overwrite)
        hdf5datapaths = glob.glob(savedir + '/*.h5')
    if args.parentdir is not None:
        print 'Make hdf5 files for directories under ' + args.parentdir
        savedir = pivlab2hdf5_parentdir(args.parentdir, overwrite=args.overwrite)
        hdf5datapaths = glob.glob(savedir + '/*.h5')
    elif args.dir is not None:
        print 'Make a hdf5 file for the following directory: ' + args.dir
        savedir = pivlab2hdf5(args.dir, overwrite=args.overwrite)
        hdf5datapaths = glob.glob(savedir + '/*.h5')
    elif args.cine is not None:
        print 'Make a hdf5 file for the following cine. Process all existing PIVLab outputs: ' + args.cine
        savedir = pivlab2hdf5_cine(args.cine, overwrite=args.overwrite)
        cinedir, cinename = os.path.split(args.cine)
        cinename_no_ext = cinename[:-5]
        hdf5datapaths = glob.glob(savedir + '/%s*.h5' % cinename_no_ext)
    return hdf5datapaths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
    parser.add_argument('-basedir', '--basedir', help='Use all available data to make a master hdf5. i.e. PROCESS ALL', type=str,
                        default=None)
    parser.add_argument('-parentdir', '--parentdir',
                        help='Use all available data inside a single PARENTDIR to make a master hdf5', type=str,
                        default=None)
    parser.add_argument('-dir', '--dir', help='Use a single DATADIR to make a master hdf5', type=str,
                        default=None)
    parser.add_argument('-cine', '--cine', help='Find all available PIVLab outputs about a PARTICULAR CINE.', type=str,
                        default=None)
    parser.add_argument('-overwrite', '--overwrite',
                        help='overwrite pivlab outputs. This is handy if code fails and force code to insert pivlab outputs to hdf5. Default: False',
                        type=bool,
                        default=False)
    parser.add_argument('-verbose', '--verbose',
                        help='If False, do not print trivial inputs',
                        type=bool,
                        default=False)
    args = parser.parse_args()
    main(args)


