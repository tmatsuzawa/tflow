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
from pathlib import Path

############ USEFUL FUNCTIONS ############
def pivlab2hdf5_basedir(basedir, header='PIV', overwrite=False, dx=1, dt=1):
    cinedir, basedirname = os.path.split(basedir)
    savedir = cinedir + '/hdf5data'
    parentdirs = glob.glob(basedir + '/' + header + '*')
    for parentdir in parentdirs:
        pivlab2hdf5_parentdir(parentdir, overwrite=overwrite)
    print('......... Done')
    return savedir

def pivlab2hdf5_parentdir(parentdir, header='PIVlab', overwrite=False, dx=1, dt=1):
    basedir, pdirname = os.path.split(parentdir)
    cinedir, basedirname = os.path.split(basedir)
    savedir = cinedir + '/hdf5data'
    datadirs = glob.glob(parentdir + '/' + header + '*')
    for datadir in datadirs:
        # print '... processing ' + os.path.split(datadir)[1]
        pivlab2hdf5(datadir, savedir=savedir, overwrite=overwrite)
    print('...... Done')
    return savedir

def pivlab2hdf5(dir, savedir=None, header='PIV_', overwrite=False, dx=1, dt=1):
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
    dir = Path(dir) # convert to Path object for easier path manipulation
    parentdir = dir.parent

    # parentdir = os.path.dirname(dir)
    # basedir = os.path.dirname(parentdir)
    # parentdirname = os.path.split(dir)[1]

    if savedir is None:
        savedir = parentdir
    #
    print('Processing: %s' % os.path.split(dir)[1])

    # MAKE HDF5
    # hdf5path
    # hdf5path = savedir + '/' + cinename_no_ext + '.h5'
    if not os.path.exists(savedir):
        print('... mkdir ', savedir)
        os.makedirs(savedir)
    hdf5path_perhaps = savedir + '/' + cinename_no_ext[:-3] + '.h5'
    if os.path.exists(hdf5path_perhaps):
        cinename_no_ext = cinename_no_ext[:-3]
        cinepath = os.path.join(cinedir, cinename_no_ext) + '.cine'
        hdf5path = hdf5path_perhaps


    with h5py.File(hdf5path, 'a') as fyle:
        ################ /exp #############################
        # /exp.attrs: experimental parameters such as commanded velocity, commanded frequency etc.

        if not fyle.__contains__('exp'):
            exp = fyle.create_group('exp')
        else:
            exp = fyle['exp']

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


        # Check if the hdf5 file already stores this PIVLab output
        redundant_pivlab_outputs = False
        for subgroup_name in list(piv.keys()):
            piv_paramnames_0 = list(piv[subgroup_name].attrs.keys())
            piv_paramvalues_0 = list(piv[subgroup_name].attrs.values())
            piv_params_0 = dict(list(zip(piv_paramnames_0, piv_paramvalues_0)))

        if redundant_pivlab_outputs and not overwrite:
            print('... skipping')
            fyle.close()
        else:
            if not overwrite:
                # This piv outputs are not saved in the master hdf5 file! Save it to the master hdf5.
                # Naming convention: /piv/piv000, /piv/piv0001
                piv_working = piv.create_group('piv%03d' % piv.attrs['no_piv_data'])
            else:
                print('... Probably, datasets such as u and v exist under %s but overwrite data anyway' % piv[subgroup_name].name)
                piv_working = piv[subgroup_name]

            # Also add data spacing (interrogation size / 2)
            W = piv_working.attrs['Int._area_%d' % piv_working.attrs['Nr._of_passes']]
            piv_working.attrs['W'] = W # smallest interrogation area size

            # Insert pivlab outputs
            datafiles = glob.glob(dir + '/*.txt')
            datafiles = sorted(datafiles)

            for i, datafile in enumerate(tqdm(datafiles, desc='... making nd arrays from PIVLab output')):
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

            # Insert data into hdf5
            if not overwrite:
                for i, piv_working_datasetname in enumerate(piv_working_datasetnames):
                    if not piv_working_datasetname in list(piv_working.keys()):
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

    print('... Done')
    return savedir


def main(args):
    if args.basedir is not None:
        print('Use all available PIVLab outputs to create master hdf5 files')
        savedir = pivlab2hdf5_basedir(args.basedir, overwrite=args.overwrite, dx=args.scale, dt=args.dt)
        hdf5datapaths = glob.glob(savedir + '/*.h5')
    if args.parentdir is not None:
        print('Make hdf5 files for directories under ' + args.parentdir)
        savedir = pivlab2hdf5_parentdir(args.parentdir, overwrite=args.overwrite, dx=args.scale, dt=args.dt)
        hdf5datapaths = glob.glob(savedir + '/*.h5')
    elif args.dir is not None:
        print('Make a hdf5 file for the following directory: ' + args.dir)
        savedir = pivlab2hdf5(args.dir, overwrite=args.overwrite, dx=args.scale, dt=args.dt)
        hdf5datapaths = glob.glob(savedir + '/*.h5')
    elif args.cine is not None:
        print('Make a hdf5 file for the following cine. Process all existing PIVLab outputs: ' + args.cine)
        savedir = pivlab2hdf5_cine(args.cine, overwrite=args.overwrite, dx=args.scale, dt=args.dt)
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
    parser.add_argument('-scale', '--scale', help='um per pixel. Default: 1', type=float, default=1.)
    parser.add_argument('-dt', '--dt', help='Time interval between frames in seconds. Default: 1', type=float, default=1.)
    
    args = parser.parse_args()
    main(args)


