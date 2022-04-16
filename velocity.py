# coding=utf-8
from tqdm import tqdm
import numpy as np
import numpy.ma as ma
import re
import h5py
import pickle
from scipy import fftpack, signal, integrate
from scipy.interpolate import interp2d, griddata, UnivariateSpline, RegularGridInterpolator, LinearNDInterpolator
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd, multivariate_normal
from scipy.optimize import minimize, curve_fit
from scipy import ndimage, interpolate, signal, special
from scipy.signal import butter, medfilt, filtfilt  # filters for signal processing
import itertools
import os, copy, sys, re, copy  # fundamentals
import time as time_mod
import subprocess, glob
import cv2
import math

import warnings
import matplotlib.cbook
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', FutureWarning)

import tflow.vector as vec
import tflow.graph as graph

global bezier_installed
try:
    import bezier  # required to plot bezier curves (not critical)
    bezier_installed = True
except:
    bezier_installed = False
    print('velocity module: bezier is missing in a current environment. pip install bezier')

path_mod = os.path.abspath(__file__)
moddirpath = os.path.dirname(path_mod)

"""
Module designed to process a planar/volumetric velocity field
- energy, enstrophy, vorticity fields
- energy spectra
- n-th order structure functions
- dissipation rate
- turbulent length scales


Philosophy: 
Prepare a velocity field array "udata". 
Pass it to functions in this module to obtain any quantities related to turbulence.
It should require a single line to obtain the desired quantity from a velocity field
unless an intermediate step is computationally expensive. 
The primary example for this is an autocorrelation function which is used for various quantities like Taylor microscale.


udata = (ux, uy, uz) or (ux, uy)
each ui has a shape (height, width, (depth), duration)

If ui-s are individually given, make udata like 
udata = np.stack((ux, uy))

Dependencies:
h5py, tqdm, numpy, scipy, matplotlib

author: takumi matsuzawa
"""


########## Fundamental operations ##########
def get_duidxj_tensor(udata, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]),
                      xx=None, yy=None, zz=None):
    """
    Assumes udata has a shape (d, nrows, ncols, duration) or  (d, nrows, ncols)
    ... one can easily make udata by np.stack((ux, uy))

    Important Warning:
    ... udata is np.stack((ux, uy, uz))
    ... udata.shape = dim, nrows, ncols, duration
    Parameters
    ----------
    udata: numpy array with shape (ux, uy) or (ux, uy, uz)
        ... assumes ux/uy/uz has a shape (nrows, ncols, duration) or (nrows, ncols, nstacks, duration)
        ... can handle udata without a temporal axis
    dx: float, x spacing
    dy: float, y spacing
    dz: float, z spacing
    xyz_orientations: 1d array-like with shape (3,)
        ... xyz_orientations = [djdx, didy, dkdz]
        ... HOW TO DETERMINE xyz_orientations:
                1. Does the xx[0, :] (or xx[0, :, 0] for 3D) monotonically increase as the index increases?
                    If True, djdx = 1
                    If False, djdx = -1
                2. Does the yy[:, 0] (or yy[:, 0, 0] for 3D) monotonically increase as the index increases?
                    If True, didy = 1
                    If False, didy = -1
                3. Does the zz[0, 0, :] monotonically increase as the index increases?
                    If True, dkdz = 1
                    If False, dkdz = -1
            ... If you are not sure what this is, use

        ... Factors between index space (ijk) and physical space (xyz)

        ... This factor is necessary since the conventions used for the index space and the physical space are different.
            Consider a 3D array a. a[i, j, k]. The convention used for this module is to interpret this array as a[y, x, z].
            (In case of udata, udata[dim, y, x, z, t])
            All useful modules such as numpy are written in the index space, but this convention is not ideal for physicists
            for several reasons.
            1. many experimental data are not organized in the index space.
            2. One always requires conversion between the index space and physical space especially at the end of the analysis.
            (physicists like to present in the units of physical units not in terms of ijk)

        ... This array is essentially a Jacobian between the index space basis (ijk) and the physical space basis (xyz)
            All information needed is just dx/dj, dy/di, dz/dk because ijk and xyz are both orthogonal bases.
            There is no off-diagonal elements in the Jacobian matrix, and one needs to supply only 3 elements for 3D udata.
            If I strictly use the Cartesian basis for xyz (as it should), then I could say they are both orthonormal.
            This makes each element of the Jacobian array to be either 1 or -1, reflecting the directions of +x/+y/+z
            with respect to +j/+i/+k


    Returns
    -------
    sij: numpy array with shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
        ... idea is... sij[spacial coordinates, time, tensor indices]
            e.g.-  sij(x, y, t) = sij[y, x, t, i, j]
        ... sij = d ui / dxj
    """

    if xx is not None and yy is not None:
        xyz_orientations = get_jacobian_xyz_ijk(xx, yy, zz)
        if zz is None:
            dx, dy = get_grid_spacing(xx, yy)
        else:
            dx, dy, dz = get_grid_spacing(xx, yy, zz)
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = shape[0]

    if dim == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))

        duxdx = np.gradient(ux, dx, axis=1) * xyz_orientations[0]
        duxdy = np.gradient(ux, dy, axis=0) * xyz_orientations[
            1]  # +dy is the column up. np gradient computes difference by going DOWN in the column, which is the opposite
        duydx = np.gradient(uy, dx, axis=1) * xyz_orientations[0]
        duydy = np.gradient(uy, dy, axis=0) * xyz_orientations[1]
        sij = np.zeros((nrows, ncols, duration, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
    elif dim == 3:
        ux, uy, uz = udata[0, ...], udata[1, ...], udata[2, ...]
        try:
            # print ux.shape
            nrows, ncols, nstacks, duration = ux.shape
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
        duxdx = np.gradient(ux, dx, axis=1) * xyz_orientations[0]
        duxdy = np.gradient(ux, dy, axis=0) * xyz_orientations[1]
        duxdz = np.gradient(ux, dz, axis=2) * xyz_orientations[2]
        duydx = np.gradient(uy, dx, axis=1) * xyz_orientations[0]
        duydy = np.gradient(uy, dy, axis=0) * xyz_orientations[1]
        duydz = np.gradient(uy, dz, axis=2) * xyz_orientations[2]
        duzdx = np.gradient(uz, dx, axis=1) * xyz_orientations[0]
        duzdy = np.gradient(uz, dy, axis=0) * xyz_orientations[1]
        duzdz = np.gradient(uz, dz, axis=2) * xyz_orientations[2]

        sij = np.zeros((nrows, ncols, nstacks, duration, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 0, 2] = duxdz
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
        sij[..., 1, 2] = duydz
        sij[..., 2, 0] = duzdx
        sij[..., 2, 1] = duzdy
        sij[..., 2, 2] = duzdz
    elif dim > 3:
        print('...Not implemented yet.')
        return None
    return sij


def decompose_duidxj(sij):
    """
    Decompose a duidxj tensor into a symmetric and an antisymmetric parts
    Returns symmetric part (eij) and anti-symmetric part (gij)

    Parameters
    ----------
    sij, 5d or 6d numpy array (x, y, t, i, j) or (x, y, z, t, i, j)

    Returns
    -------
    eij: 5d or 6d numpy array, symmetric part of rate-of-strain tensor.
         5d if spatial dimensions are x and y. 6d if spatial dimensions are x, y, and z.
    gij: 5d or 6d numpy array, anti-symmetric part of rate-of-stxxain tensor.
         5d if spatial dimensions are x and y. 6d if spatial dimensions are x, y, and z.

    """
    dim = len(sij.shape) - 3  # spatial dim
    if dim == 2:
        duration = sij.shape[2]
    elif dim == 3:
        duration = sij.shape[3]

    eij = np.zeros(sij.shape)
    # gij = np.zeros(sij.shape) #anti-symmetric part
    for t in range(duration):
        for i in range(dim):
            for j in range(dim):
                if j >= i:
                    eij[..., t, i, j] = 1. / 2. * (sij[..., t, i, j] + sij[..., t, j, i])
                    # gij[..., i, j] += 1./2. * (sij[..., i, j] - sij[..., j, i]) #anti-symmetric part
                else:
                    eij[..., t, i, j] = eij[..., t, j, i]
                    # gij[..., i, j] = -gij[..., j, i] #anti-symmetric part

    gij = sij - eij
    return eij, gij


def reynolds_decomposition(udata, t0=0, t1=None):
    """
    Apply the Reynolds decomposition to a velocity field
    Returns a mean field (time-averaged) and a fluctuating field

    Parameters
    ----------
    udata: numpy array
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)

    Returns
    -------
    u_mean: nd array, mean velocity field
    u_turb: nd array, turbulent velocity field
    """
    udata = fix_udata_shape(udata)
    dim = len(udata)

    # Initialization
    if dim == 2:
        u_mean = np.zeros((udata.shape[0], udata.shape[1], udata.shape[2]))
    if dim == 3:
        u_mean = np.zeros((udata.shape[0], udata.shape[1], udata.shape[2], udata.shape[3]))
    u_turb = np.zeros_like(udata)
    for i in range(dim):
        u_mean[i] = np.nanmean(udata[i, ..., t0:t1], axis=-1)  # axis=dim is always the time axis in this convention
        for t in range(udata.shape[-1]):
            u_turb[i, ..., t] = udata[i, ..., t] - u_mean[i]
    return u_mean, u_turb


def get_mean_flow_field_using_udatapath(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                        t0=0, t1=None, inc=1,
                                        clean=True, cutoff=np.inf, method='nn', median_filter=False, verbose=False,
                                        replace_zeros=True,
                                        notebook=True):
    """
    Returns mean_field when a path to udata is provided
    ... recommended to use if udata is large (> half of your memory size)
    ... only a snapshot of data exists on memory while time-averaging is performed

    Parameters
    ----------
    udatapath: str
        ... path to udata
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    slicez: int, default: None
        ... Option to return a 2D time-averaged field at z=slicez from 4D udata
        ... This is mostly for the sake of fast turnout of analysis
    inc: int, default: 1
        ... time increment to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    thd: float, default: np.inf
        ... energy > thd will be replaced by fill_value. (Manual screening of data)
    fill_value: float, default: np.nan
        ... value used to fill the data when data value is greater than a threshold

    Returns
    -------
    udata_m: 2d or 3d arrya
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    with h5py.File(udatapath, mode='r') as f:
        try:
            height, width, depth, duration = f['ux'].shape
            height, width, depth = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
            dim = 3
            shape = (dim, height, width, depth)
        except:
            height, width, duration = f['ux'].shape
            height, width = f['ux'][y0:y1, x0:x1, 0].shape
            dim = 2
            shape = (dim, height, width)
        if t1 is None:
            t1 = duration
        udata_m = np.zeros(shape)

        counter = 0
        for t in tqdm(range(t0, t1, inc), desc='mean flow'):
            udata_tmp = get_udata_from_path(udatapath,
                                            x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                            t0=t, t1=t + 1, verbose=verbose)
            if clean:
                udata_tmp = clean_udata(udata_tmp, cutoff=cutoff, method=method, median_filter=median_filter,
                                        replace_zeros=replace_zeros, verbose=verbose, showtqdm=verbose)
                udata_tmp = fix_udata_shape(udata_tmp)
            else:
                udata_tmp = fix_udata_shape(udata_tmp)
            udata_m += udata_tmp[..., 0]
            counter += 1
        udata_m /= counter

    if notebook:
        from tqdm import tqdm
    return udata_m


########## vector operations ##########
def div(udata, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]), xx=None, yy=None, zz=None):
    """
    Computes divergence of a velocity field

    Parameters
    ----------
    udata: numpy array
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)

    Returns
    -------
    div_u: numpy array
          ... div_u has a shape (height, width, depth, duration) (3D) or (height, width, duration) (2D)
    """
    sij = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz, xyz_orientations=xyz_orientations, xx=xx, yy=yy,
                            zz=zz)  # shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    dim = len(sij.shape) - 3  # spatial dim
    div_u = np.zeros(sij.shape[:-2])
    for d in range(dim):
        div_u += sij[..., d, d]

    return div_u


def grad(U, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]), xx=None, yy=None, zz=None):
    """
    Computes divergence of a velocity field

    Parameters
    ----------
    U: numpy array
          ... U has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... U may have a shape (height, width, duration) or (height, width) (2D)

    Returns
    -------
    grad_U: numpy array
          ... grad_U has a shape (3, height, width, depth, duration) (3D) or (2, height, width, duration) (2D)
    """

    if xx is not None and yy is not None:
        xyz_orientations = get_jacobian_xyz_ijk(xx, yy, zz)
        if zz is None:
            dx, dy = get_grid_spacing(xx, yy)
        else:
            dx, dy, dz = get_grid_spacing(xx, yy, zz)
    shape = U.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if zz is not None:
        dim = 3
    else:
        dim = 2

    if dim == 2:
        try:
            nrows, ncols, duration = U.shape
        except:
            nrows, ncols = U.shape
            duration = 1
            U = U.reshape((U.shape[0], U.shape[1], duration))

        dUdx = np.gradient(U, dx, axis=1) * xyz_orientations[0]
        dUdy = np.gradient(U, dy, axis=0) * xyz_orientations[1]
        grad_U = np.stack((dUdx, dUdy))
    elif dim == 3:
        try:
            nrows, ncols, nstacks, duration = U.shape
        except:
            nrows, ncols, nstacks = U.shape
            duration = 1
            U = U.reshape((U.shape[0], U.shape[1], U.shape[2], duration))
        dUdx = np.gradient(U, dx, axis=1) * xyz_orientations[0]
        dUdy = np.gradient(U, dy, axis=0) * xyz_orientations[1]
        dUdz = np.gradient(U, dz, axis=2) * xyz_orientations[2]
        grad_U = np.stack((dUdx, dUdy, dUdz))
    return grad_U


def curl(udata, dx=1., dy=1., dz=1., xyz_orientations=np.asarray([1, -1, 1]),
         xx=None, yy=None, zz=None, verbose=False):
    """
    Computes curl of a velocity field using a rate of strain tensor
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = np.stack((ugrid1, vgrid1))
        omega = vec.curl(udata)
    Parameters
    ----------
    udata: (ux, uy, uz) or (ux, uy)
    dx, dy, dz: float, spatial spating of a 2D/3D grid
    xyz_orientations: 1d array
        ... differentiation in the index space and the physical space must be conducted properly.
        tflow convention is to treat the row, column, and the depth (i,j,k) are parallel to x, y, z in the physical space;
        however, this does not specify the direction of the axes. (+i direction is only equal to EITHER +x or -x).
        This ambiguity causes a problem during differentiation, and the choice is somewhat arbitrary to the users.
        This function offers a solution by two ways. One way is to give a 2d/3d array of the positional grids.
        If xx, yy, zz are given, it would automatically figures out how +i,+j,+k are aligned with +x,+y,+z.
        The second way is give delta x/ delta_i, dy/delta_j, dz/delta_k. This argument is related to this method.
        ... e.g.
        xyz_orientations = [1, 1, 1]... means +x // +i, +y//+j, +z//+k
        xyz_orientations = [-1, -1, -1]... means +x // -i, +y//-j, +z//-k
        xyz_orientations = [1, -1, 1]... means +x // +i, +y//-j, +z//+k
    xx: 2d/3d array, positional grid
        ... If given, it would automatically figure out whether +x and +i point at the same direction,
         and the curl is computed based on that
    yy: 2d/3d array, positional grid
        ... If given, it would automatically figure out whether +y and +j point at the same direction,
         and the curl is computed based on that
    zz: 2d/3d array, positional grid
        ... If given, it would automatically figure out whether +z and +k point at the same direction,
         and the curl is computed based on that

    Returns
    -------
    omega: numpy array
        shape: (height, width, duration) (2D) or (height, width, depth, duration) (3D)

    """
    if verbose:
        print('... curl(): If the result is not sensible, consult altering xyz_orientations.\n'
              'A common mistake is that udata is not origanized properly such that '
              '+x direction is not equal to +direction along the row of an array'
              'or +y direction is not equal to +direction along the column of an array')

    sij = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz, xyz_orientations=xyz_orientations, xx=xx, yy=yy, zz=zz)
    dim = len(sij.shape) - 3  # spatial dim
    eij, gij = decompose_duidxj(sij)
    if dim == 2:
        omega = 2 * gij[..., 1, 0]  # checked. this is correct.
    elif dim == 3:
        # sign issue was checked. this is correct.
        omega1, omega2, omega3 = 2. * gij[..., 2, 1], 2. * gij[..., 0, 2], 2. * gij[..., 1, 0]
        # omega1, omega2, omega3 = -2. * gij[..., 2, 1], 2. * gij[..., 0, 2], -2. * gij[..., 1, 0]
        omega = np.stack((omega1, omega2, omega3))
    else:
        print('Not implemented yet!')
        return None
    return omega


def curl_2d(ux, uy, dx=1., dy=1., xyz_orientations=np.asarray([1, -1]), xx=None, yy=None, zz=None):
    """
    Calculate curl of 2D (or 2D+1) field

    Parameters
    ----------
    ux: 2D array
        x component of a 2D field
    uy: 2D array
        y component of a 2D field
    dx: float
        data spacing (mm/px)
    dy: float
        data spacing (mm/px)

    Returns
    -------
    omega: 2D numpy array
        vorticity field
    """
    if xx is not None and yy is not None:
        xyz_orientations = get_jacobian_xyz_ijk(xx, yy, zz)

    # duxdx = np.gradient(ux, axis=1)
    duxdy = np.gradient(ux, dy, axis=0) * xyz_orientations[0]
    duydx = np.gradient(uy, dx, axis=1) * xyz_orientations[1]
    # duydy = np.gradient(uy, axis=0)

    omega = duydx - duxdy

    return omega


########## Elementary analysis ##########
def get_energy(udata):
    """
    Returns energy(\vec{x}, t) of udata
    ... energy = np.nansum(udata**2, axis=0) / 2

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    energy: nd array
    """

    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = udata.shape[0]
    energy = np.zeros(shape[1:])
    for d in range(dim):
        energy += udata[d, ...] ** 2
    energy /= 2.

    if type(udata) == np.ma.core.MaskedArray:
        mask = udata.mask
        energy = np.ma.masked_array(energy, mask=mask[1:])
        energy = energy.filled(np.nan)
    return energy


def get_enstrophy(udata, dx=1., dy=1., dz=1., xx=None, yy=None, zz=None):
    """
    Returns enstrophy(\vec{x}, t) of udata
    ... enstropy = omega ** 2
    ... omega = curl(udata)

    Parameters
    ----------
    udata: nd array
    dx: float
        data spacing along x
    dy: float
        data spacing along y
    dz: float
        data spacing along z

    Returns
    -------
    enstrophy: nd array
    """
    dim = udata.shape[0]
    omega = curl(udata, dx=dx, dy=dy, dz=dz, xx=xx, yy=yy, zz=zz)
    shape = omega.shape  # shape=(dim, nrows, ncols, nstacks, duration) if nstacks=0, shape=(dim, nrows, ncols, duration)
    if dim == 2:
        enstrophy = omega ** 2
    elif dim == 3:
        enstrophy = np.zeros(shape[1:])
        for d in range(dim):
            enstrophy += omega[d, ...] ** 2
    return enstrophy


def get_time_avg_energy(udata):
    """
    Returns a time-averaged energy field

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    energy_avg:
        time-averaged energy field
    """
    energy = get_energy(udata)
    energy_avg = np.nanmean(energy, axis=-1)
    return energy_avg


def get_time_avg_enstrophy(udata, dx=1., dy=1., dz=1., xx=None, yy=None, zz=None):
    """
    Returns a time-averaged enstrophy field

    Parameters
    ----------
    udata: nd array, velocity field
    dx: float, spacing along x
    dy: float, spacing along y
    dz: float, spacing along z (optional, applicable to 3D data)
    xx: nd array, x coordinates
    yy: nd array, y coordinates
    zz: nd array, z coordinates (optional, applicable to 3D data)

    Returns
    -------
    enstrophy_avg: nd array, time-averaged enstrophy field
    """
    enstrophy = get_enstrophy(udata, dx=dx, dy=dy, dz=dz, xx=xx, yy=yy, zz=zz)
    enstrophy_avg = np.nanmean(enstrophy, axis=-1)
    return enstrophy_avg


def get_spatial_avg_energy(udata, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, use3comp=True, ):
    """
    Returns energy averaged over a spatial region

    Parameters
    ----------
    udata: nd array, velocity field
    x0: int, start index along x- udata[:, y0:y1, x0:x1, z0:z1, :]
    x1: int, end index along x
    y0: int, start index along y
    y1: int, end index along y
    z0: int, start index along z
    z1: int, end index along z
    use3comp: bool, If udata is 3D v-field, this function returns 3D energy if True, 2D energy (ux^2 + uy^2)/2 if False

    Returns
    -------
    energy_vs_t: nd array, spatially averaged energy
    ... shape: (duration, )
    ... this array can be plotted against time
    energy_vs_t_err: nd array, standard error of spatially averaged energy
    """
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        if use3comp:
            udata = udata[:, y0:y1, x0:x1, z0:z1, :]
        else:  # use only ux, uy
            udata = udata[:2, y0:y1, x0:x1, z0:z1, :]
    energy = get_energy(udata)
    energy_vs_t = np.nanmean(energy, axis=tuple(range(dim)))
    energy_vs_t_std = np.nanstd(energy, axis=tuple(range(dim)))

    n_elements = np.empty(energy.shape[-1])
    for t in range(energy.shape[-1]):
        n_elements[t] = np.sum(~np.isnan(energy[..., t]))
    energy_vs_t_err = energy_vs_t_std / np.sqrt(n_elements)
    return energy_vs_t, energy_vs_t_err


def get_spatial_avg_energy_inside_r_from_path(dpath, r, xc=0., yc=0.,
                                              x0=0, x1=None, y0=0, y1=None,
                                              t0=0, t1=None,
                                              notebook=True):
    """
    Returns energy averaged inside a circle with radius r at (xc, yc)
    .. ONLY AVAILABLE TO 2D v-field

    Parameters
    ----------
    dpath: str, a path to the udata
    r: float, radius
    xc: float, x-coordinate of the circle (in real space)
    yc: float, y-coordinate of the circle (in real space)
    x0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, t0:t1]
    x1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, t0:t1]
    y0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, t0:t1]
    y1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, t0:t1]
    t0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, t0:t1]
    t1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, t0:t1]
    notebook: bool, if True, it uses tqdm_notebook instead of tqdm

    Returns
    -------
    esavg_in: 1d array, energy averaged over space inside the circle
    esavg_out: 1d array, energy averaged over space outside the circle
    esavg_in_err: 1d array, standard error for esavg_in
    esavg_out_err: 1d array, standard error for esavg_out
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm

    height, width, duration = get_udata_dim(dpath)
    if x1 is None:
        x1 = width
    if y1 is None:
        y1 = height
    if t1 is None:
        t1 = duration

    udata, xx, yy = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, t0=0, t1=1, return_xy=True,
                                        verbose=False)  # dummy
    rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
    inside = rr < r

    esavg_in, esavg_in_err = np.empty(t1 - t0), np.empty(t1 - t0)
    esavg_out, esavg_out_err = np.empty(t1 - t0), np.empty(t1 - t0)
    for i, t in enumerate(tqdm(range(t0, t1))):
        udata = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, t0=t, t1=t + 1, verbose=False)
        energy = get_energy(udata)
        esavg_in[i], esavg_in_err[i] = np.nanmean(energy[inside]), np.nanstd(energy[inside]) / np.sqrt(len(inside))
        esavg_out[i], esavg_out_err[i] = np.nanmean(energy[~inside]), np.nanstd(energy[~inside]) / np.sqrt(len(inside))

    if notebook:
        from tqdm import tqdm as tqdm
    return esavg_in, esavg_out, esavg_in_err, esavg_out_err


def get_spatial_avg_enstrophy(udata, dx=1., dy=1., dz=1., x0=0, x1=None, y0=0, y1=None,
                              z0=0, z1=None, xx=None, yy=None, zz=None):
    """
    Returns spatially averaged enstrophy

    Parameters
    ----------
    udata: nd array, v-field
    dx: float, x-spacing
    dy: float, y-spacing
    dz: float, z-spacing
    x0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    x1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    y0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    y1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    z0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, z0:z1, :]
    z1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, z0:z1, :]
    xx: 2/3d array, x-coordinates- this is alternative to passing (dx, dy, dz)
    yy: 2/3d array, y-coordinates- this is alternative to passing (dx, dy, dz)
    zz: 3d array, z-coordinates- this is alternative to passing (dx, dy, dz)

    Returns
    -------
    enstrophy_vs_t: 1d array, spatially averaged enstrophy
    enstrophy_vs_t_err: 1d array, standard error for enstrophy_vs_t
    """
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    enstrophy = get_enstrophy(udata, dx=dx, dy=dy, dz=dz, xx=xx, yy=yy, zz=zz)
    enstrophy_vs_t = np.nanmean(enstrophy, axis=tuple(range(dim)))
    enstrophy_vs_t_std = np.nanstd(enstrophy, axis=tuple(range(dim)))

    n_elements = np.empty(enstrophy.shape[-1])
    for t in range(enstrophy.shape[-1]):
        n_elements[t] = np.sum(~np.isnan(enstrophy[..., t]))
    enstrophy_vs_t_err = enstrophy_vs_t_std / np.sqrt(n_elements)
    return enstrophy_vs_t, enstrophy_vs_t_err


def get_radial_enstrophy_dist_from_path(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None, inc=1,
                                        notebook=True,
                                        xc=None, yc=None, zc=None, n=50,
                                        clean=False,
                                        clean_kwargs={'u_cutoff': np.inf, 'method': 'nn', 'median_filter': False}):
    """
    Returns the radial enstrophy profile (i.e. angular average)
    ... This function loads udata frame by frame.

    Parameters
    ----------
    udatapath: str, path (h5) to udata
    x0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    x1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    y0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    y1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, :]
    z0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, z0:z1, :]
    z1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, z0:z1, :]
    t0: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, z0:z1, t0:t1]
    t1: int, an index used to load udata from dpath- udata[:, y0:y1, x0:x1, z0:z1, t0:t1]
    inc: int, it uses every "inc" frames to compute the radial profile
    notebook: bool, if True, uses tqdm_notebook instead of tqdm to display progress
    xc: float, x-coordinate of the origin of the polar/spherical coordinate system
    yc: float, y-coordinate of the origin of the polar/spherical coordinate system
    zc: float, z-coordinate of the origin of the polar/spherical coordinate system
    n: int, number of bins in the radial profile
    clean: bool, if True, it cleans udata before computing the enstrophy
    clean_kwargs: dict, kwargs for clean_udata(udata, **clean_kwargs)

    Returns
    -------
    r: 1d array, radial coordinate
    enstrophy_vs_r: 2d array, average enstrophy at distance r with shape (# of bins about r, duration)
    enstrophy_vs_r_err: 2d array, standard error for enstrophy_vs_r
    """
    if notebook: from tqdm import tqdm_notebook as tqdm
    keys = get_h5_keys(udatapath)
    if 'uz' in keys:
        dim = 3
    else:
        dim = 2

    duration = get_udata_dim(udatapath)[-1]
    if t1 is None: t1 = duration

    for i, t in enumerate(tqdm(range(t0, t1, inc))):
        if i == 0:
            udata_and_grids = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=t, t1=t + 1,
                                                  return_xy=True, verbose=False)
            udata = udata_and_grids[0]
            if dim == 3:
                xx, yy, zz = udata_and_grids[1:]
                if xc is None or yc is None or zc is None:
                    try:
                        xc, yc, zc = read_data_from_h5(udatapath, ['xc', 'yc', 'zc'])
                    except:
                        xc, yc, zc = get_center_of_energy(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
            else:
                xx, yy = udata_and_grids[1:]
                zz = None
                if xc is None or yc is None:
                    try:
                        xc, yc = read_data_from_h5(udatapath, ['xc', 'yc'])
                    except:
                        xc, yc = get_center_of_energy(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        else:
            udata = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=t, t1=t + 1, verbose=False)
        if clean:
            udata = clean_udata(udata, **clean_kwargs)
        enst = get_enstrophy(udata, xx=xx, yy=yy, zz=zz)[..., 0]
        rs, enstR, enstR_std = get_spatial_profile(xx, yy, enst, zz=zz, xc=xc, yc=yc, zc=zc, n=n,
                                                   showtqdm=False)  # angular average

        if i == 0:
            enstRs = np.empty(enstR.shape[:1] + (len(range(t0, t1, inc)),))
            enstR_stds = np.empty(enstR_std.shape[:1] + (len(range(t0, t1, inc)),))
        enstRs[:, i], enstR_stds[:, i] = enstR[:, 0], enstR_std[:, 0]

    if notebook: from tqdm import tqdm as tqdm
    return rs[:, 0], enstRs, enstR_stds


def get_turbulence_intensity_local(udata):
    """
    Turbulence intensity is defined as u/U where
    u = sqrt((ux**2 + uy**2 + uz**2)/3) # characteristic turbulent velocity
    U = sqrt((Ux**2 + Uy**2 + Uz**2))   # norm of the rms velocity

    Note that this is ill-defined for turbulence with zero-mean flow !

    Parameters
    ----------
    udata: nd array, a velocity field

    Returns
    -------
    ti_local: nd array
        ... a field of a turbulent intensity (a scaler field)
    """
    dim = udata.shape[0]
    udata_mean, udata_t = reynolds_decomposition(udata)

    u_rms = np.zeros_like(udata_mean[0])
    u_t_rms = np.zeros_like(udata_t[0])
    ti_local = np.zeros_like(u_t_rms)
    for d in range(dim):
        u_rms += udata_mean[d, ...] ** 2
        u_t_rms += udata_t[d, ...] ** 2
    u_rms = np.sqrt(u_rms)
    u_t_rms = np.sqrt(u_t_rms / float(dim))
    for t in range(u_t_rms.shape[-1]):
        ti_local[..., t] = u_t_rms[..., t] / u_rms
    return ti_local


def get_turbulence_intensity_from_path(dpath, writeData=False, overwrite=False, notebook=True):
    """
    Computes the turbulence intensity of udata in a given path

    Parameters
    ----------
    dpath: str, a path (h5) to the udata
    writeData: bool, if True, it writes the turbulence intensity to the h5 file
    overwrite: bool, if True, it overwrites the turbulence intensity if it already exists
    notebook: bool, if True, it uses tqdm_notebook instead of tadm to display the progress

    Returns
    -------
    ti: nd array, turbulence intensity with shape (height, width, (optional: depth))
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm

    keys = get_h5_keys(dpath)
    dummy = get_mean_flow_field_using_udatapath(dpath, t0=0, t1=1)
    dim, duration = dummy.shape[0], get_udata_dim(dpath)[-1]

    if 'ti' in keys and not overwrite:
        ti = read_data_from_h5(dpath, ['ti'])[0]

        if notebook:
            from tqdm import tqdm as tqdm
        return ti
    else:
        if not all([item in keys for item in ['ux_m', 'uy_m']]):
            udata_m = get_mean_flow_field_using_udatapath(dpath)
            if dim == 3:
                add_data2udatapath(dpath, {'ux_m': udata_m[0, ...], 'uy_m': udata_m[1, ...], 'uz_m': udata_m[2, ...]})
            else:
                add_data2udatapath(dpath, {'ux_m': udata_m[0, ...], 'uy_m': udata_m[1, ...]})
        else:
            if dim == 3:
                keys2load = ['ux_m', 'uy_m', 'uz_m']
                ux_m, uy_m, uz_m = read_data_from_h5(dpath, keys2load)
            else:
                keys2load = ['ux_m', 'uy_m']
                ux_m, uy_m = read_data_from_h5(dpath, keys2load)

            if dim == 3:
                udata_m = np.stack((ux_m, uy_m, uz_m))
            else:
                udata_m = np.stack((ux_m, uy_m))

        for t in tqdm(range(duration), desc='turb. intensity'):
            udata = get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)[..., 0]
            udata_t = udata - udata_m
            if t == 0:
                ut2_sum = np.zeros_like(udata_m)
                counts = np.zeros_like(udata_m)
            keep = ~np.isnan(udata_t)
            counts += keep
            ut2_sum[keep] = ut2_sum[keep] + udata_t[keep] ** 2
        ut_rms = ut2_sum / counts
        U_rms = np.sqrt(udata_m ** 2)

        ti = np.nanmean(ut_rms, axis=0) / np.nanmean(U_rms, axis=0)
        datadict = {
            'ui_rms': U_rms,  # depends on (dim, x,y,z)
            'ui_t_rms': ut_rms,  # depends on (dim, x,y,z)
            'ti': ti,  # depends on (x,y,z)
        }
        if writeData:
            add_data2udatapath(dpath, datadict, overwrite=overwrite)

        if notebook:
            from tqdm import tqdm as tqdm
        return ti


# Circulation computation (Use of spatially 2D udata is recommended)
def compute_circulation(udata, xx, yy, zz=None, contour=None, rs=[10.], xc=0, yc=0, zc=0,
                        n=100, return_contours=False):
    """
    Returns circulation along a given contour (if None, a circular contour by default) with udata and spatial grids as inputs
        ... it is compatible with spatially 2D/3D udata
            Yet, 2D udata is probably easier to handle. If you need a sliced velocity field from a volumetric udata,
            consider using slice_udata_3d()
        ... For 3D udata, you probably want to pass a contour manually.
        ... Default contour is a circle(s) with radius of 10 and its center at (xc, yc) or (xc, yc, zc).
            If you want the contours with multiple radii, try something like rs = [10, 20, 25, ...]
        ... if you want to draw contours later, you can output pts on the contours by setting return_contours True.

    How it works:
        It computes a Riemann sum of addends defined by velocity \cdot dl at each point on the contour.
        ... In case of
                1) a part of the contour were out of the plane / volume
                2) udata contains nans at a point on the contour
            this function first computes the average of the addends, then multiples the avg by the number of pts along the contour.
            This way, it still computes the correct value of circulation if these cases occur, and still outputs
            somewhat meaningful values even if these cases occur.

    Parameters
    ----------
    udata: ndarray, velocity field
    xx: ndarray, x-coordinates of the grid (real space)
    yy: ndarray, y-coordinates of the grid (real space)
    zz: ndarray, z-coordinates of the grid (real space)
    contour: 2d array with shape (n, dim), default: None
        ... contour must be closed but it is not necessary to pass a closed contour.
            The contour will be closed by connecting the first and last pts in this array.
            (It is also okay to pass a contour with an array of a closed contour to this function)
        ... A default contour is circular with a radius (or radii) of 10 around the center (xc, yc, zc)
        ... contour[:, 0]- x-coords of pts on the contour
            contour[:, 1]- y-coords of pts on the contour
            contour[:, 2]- z-coords of pts on the contour
    rs: list, radii in the same units as xx/yy/zz
        ... only relevant if "contours" is None
    xc: float, x-coordinate of the center of a circular contour(s), default:0.
        ... only relevant if "contours" is None
    yc: float, y-coordinate of the center of a circular contour(s), default:0.
        ... only relevant if "contours" is None
    zc: float, z-coordinate of the center of a circular contour(s), default:0.
        ... only relevant if "contours" is None
    n: number of sampled pts on the circular contour
        ... only relevant if "contours" is None
    return_contours: bool, default: False
        ... If True, it returns pts on the contour(s)
        ... This feature could be handy if you'd like to plot the contour(s) later

    Returns
    -------
    gammas: nd array with shape (duration, ) or (len(rs), duration), circulation values
    contours: nd array (optional)
    """
    contours = []
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]

    if np.isnan([xc, yc, zc]).any():
        gammas = np.empty((len(rs), duration))
        gammas[:] = np.nan
        if return_contours:
            contours = np.empty((len(rs), n, 2))
            contours[:] = np.nan

    if dim == 3 and zz is None:
        raise ValueError(
            '... Spatial 3D udata is provided. grid for z-coordinates must be also supplied for interpolation.')

    if contour is None:
        if dim == 3:
            warnings.warn(
                "Depreciated! You probably want to pass a contour to compute circulation using volumetric udata.",
                DeprecationWarning)
            print('... you provided spatially 3D udata. By default, it computes a line integral along \
            a circular contour(s) with its center (xc, yc, zc=0) on the xy plane')
            print('... If you preferred to compute circulation using spatialy 3D udata along different contours, \
            you must pass a contour manually. See docstrings.')
            print('... Alternative is passing a planar udata by slicing a volumetric udata which can be obtained by \
            velocity.slice_udata_3d(udata, xx, yy, zz, surface_vector), cartesian_coords_of_a_pt_on_the_plane')

        thetas = np.linspace(0, 2 * np.pi, n)

        gammas = np.empty((len(rs), duration))
        for m, r in enumerate(rs):
            xs = xc + r * np.cos(thetas)
            ys = yc + r * np.sin(thetas)
            zs = np.zeros_like(xs) + zc

            dx = np.diff(xs, append=xs[0])
            dy = np.diff(ys, append=ys[0])
            dz = np.zeros(len(xs))

            if dim == 2:
                dl = np.stack((dx, dy)).T
                contour = np.stack((xs, ys)).T  # x, y
                contour4int_func = contour[:, [1, 0]]  # y, x
            else:
                dl = np.stack((dx, dy, dz)).T
                contour = np.stack((xs, ys, zs)).T  # x, y
                contour4int_func = contour[:, [1, 0, 2]]  # y, x, z
            contours.append(contour)

            # Interpolating function, output of interpolate_udata_at_instant_of_time(), requires arguments y, x, z
            # in that order (... this is just a convention)
            # To accomodate this, change the order of the contour array
            for t in range(duration):
                fs = interpolate_udata_at_instant_of_time(udata, xx, yy, zz=zz, t=t)
                gamma = 0  # initialization
                for i in range(dim):
                    uxi_along_cntr = fs[i](contour4int_func)  # fs[i]( (y, x, z) ): ui at (x, y, z)
                    gamma += uxi_along_cntr * dl[:, i]
                gammas[m, t] = np.nanmean(gamma) * len(uxi_along_cntr)
    else:
        contours.append(contour)
        if contour.shape[-1] == 2:
            contour4int_func = contour[:, [1, 0]]
        elif contour.shape[-1] == 3:
            contour4int_func = contour[:, [1, 0, 2]]

        if dim != contour.shape[-1]:
            print('... udata dimension and contour dimension does not match!')
            print('... This makes ambiguious how to compute a line integral. Aborting... ')
            sys.exit(1)
        else:
            dxis = []
            for i in range(dim):
                dxi = np.diff(contour[:, i], append=contour[0, i])
                dxis.append(dxi)
            dl = np.stack([dxi for dxi in dxis]).T

            gammas = np.empty(duration)
            for t in range(duration):
                fs = interpolate_udata_at_instant_of_time(udata, xx, yy, zz=zz, t=t)
                gamma = 0  # initialization
                for i in range(dim):
                    uxi_along_cntr = fs[i](contour4int_func)
                    gamma += uxi_along_cntr * dl[:, i]
                gammas[t] = np.nanmean(gamma) * len(uxi_along_cntr)
    if return_contours:
        contours = np.asarray(contours)
        return gammas, contours
    else:
        return gammas


def compute_line_integral(*args, **kwargs):
    """
    A wrapper for compute_circulation_along_contour()
    """
    results = compute_circulation(*args, **kwargs)
    return results


########## Energy spectrum, Dissipation spectrum ##########
def fft_nd(udata, dx=1, dy=1, dz=1,
           x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
           window=None, return_kgrid=True):
    """
    Compute the ND-FFT of a 2d/3D udata.
    ... udata(x, y, z, t) -> ukdata(kx, ky, kz, t)

    Parameters
    ----------
    udata: ndarray, a velocity field
    dx: float, the grid spacing in x direction
    dy: float, the grid spacing in y direction
    dz: float, the grid spacing in z direction
    x0: int, the starting index of x
    x1: int, the ending index of x
    y0: int, the starting index of y
    y1: int, the ending index of y
    z0: int, the starting index of z
    z1: int, the ending index of z
    window: str, a name of the window function
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs a decay scale),
        tukey (needs a taper fraction)
    return_kgrid: bool, whether to return the wavenumber grid
    ... If True, ukdata, np.asarray([kxx, kyy, (kzz: optional)]) will be returned.
    ... plt.pcolormesh(kxx, kyy, 0.5 * (ukdata[0, ...]**2 + ukdata[1, ...]**2)) plots the 2D energy spectrum

    Returns
    -------
    ukdata: ndarray, the ND-FFT of udata
    kgrid: ndarray, the wavenumber grid, optional
    ... kgrid[0/1/2, ...] is the wavenumber grid in x/y/z direction
    """
    if dx is None or dy is None:
        print('ERROR: dx or dy is not provided! dx is grid spacing in real space.')
        print('... k grid will be computed based on this spacing! Please provide.')
        raise ValueError
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    dim = len(udata)
    udata = fix_udata_shape(udata)
    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        if dz is None:
            print('ERROR: dz is not provided! dx is grid spacing in real space.')
            print('... k grid will be computed based on this spacing! Please provide.')
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    n_samples = 1
    for d in range(dim):
        n_samples *= udata.shape[d + 1]

    # Apply a window to get lean FFT spectrum for aperiodic signals
    duration = udata.shape[-1]
    if window is not None:
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows
        ukdata = np.fft.fftn(udata_tapered, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))
    else:
        ukdata = np.fft.fftn(udata, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))

    if return_kgrid:
        if dim == 2:
            height, width, duration = udata[0, ...].shape
            kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
            ky = np.fft.fftfreq(height, d=dy)
            kx = np.fft.fftshift(kx)
            ky = np.fft.fftshift(ky)
            kxx, kyy = np.meshgrid(kx, ky)
            kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi  # Convert inverse length into wavenumber
            return ukdata, np.asarray([kxx, kyy])

        elif dim == 3:
            height, width, depth, duration = udata[0, ...].shape
            kx = np.fft.fftfreq(width, d=dx)
            ky = np.fft.fftfreq(height, d=dy)
            kz = np.fft.fftfreq(depth, d=dz)
            kx = np.fft.fftshift(kx)
            ky = np.fft.fftshift(ky)
            kz = np.fft.fftshift(kz)
            kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
            kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
            return ukdata, np.asarray([kxx, kyy, kzz])
    else:
        return ukdata


#### Energy spectra computation (LEGACY)- used until 08/31/20
def get_energy_spectrum_nd_old(udata, x0=0, x1=None, y0=0, y1=None,
                               z0=0, z1=None, dx=None, dy=None, dz=None,
                               window=None, correct_signal_loss=True):
    """
    DEPRECATED: TM cleaned up the code, and improved the literacy and transparency of the algorithm- TM (Sep 2020)
    ... Please use the updated function: get_energy_spectrum_nd()
    ... the new function correctly returns the SPECTRAL DENSITY. T
    ...... This code might contain minor errors (like a factor of 2pi). The new code has no such ambiguity.

    Returns nd energy spectrum from velocity data (FFT of a velocity field)

    Parameters
    ----------
    udata: nd array
    dx: data spacing in x (units: mm/px)
    dy: data spacing in y (units: mm/px)
    dz: data spacing in z (units: mm/px)
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    energy_fft: nd array with shape (height, width, duration) or (height, width, depth, duration)
    ks: nd array with shape (ncomponents, height, width, duration) or (ncomponents, height, width, depth, duration)

    Example
    -----------------
    nx, ny = 100, 100
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 4*np.pi, ny)
    dx, dy = x[1]- x[0], y[1]-y[0]

    # Position grid
    xx, yy = np.meshgrid(x, y)

    # In Fourier space, energy will have a peak at (kx, ky) = (+/- 5, +/- 2)
    ux = np.sin(2.5*xx + yy)
    uy = np.sin(yy) * 0
    udata_test = np.stack((ux, uy))
    ek, ks = get_energy_spectrum_nd(udata_test, dx=dx, dy=dy)
    graph.color_plot(xx, yy, (ux**2 + uy**2 ) /2., fignum=2, subplot=121)
    fig22, ax22, cc22 = graph.color_plot(ks[0], ks[1], ek.real[..., 0], fignum=2, subplot=122, figsize=(16, 8))
    graph.setaxes(ax22, -10, 10, -10, 10)

    """
    if dx is None or dy is None:
        print('ERROR: dx or dy is not provided! dx is grid spacing in real space.')
        print('... k grid will be computed based on this spacing! Please provide.')
        raise ValueError
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    dim = len(udata)
    udata = fix_udata_shape(udata)
    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        if dz is None:
            print('ERROR: dz is not provided! dx is grid spacing in real space.')
            print('... k grid will be computed based on this spacing! Please provide.')
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    n_samples = 1
    for d in range(dim):
        n_samples *= udata.shape[d + 1]

    # Apply a window to get lean FFT spectrum for aperiodic signals
    duration = udata.shape[-1]
    if window is not None or window != 'rectangle':
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows
        ukdata = np.fft.fftn(udata_tapered, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))

        energy, energy_tapered = get_energy(udata), get_energy(udata_tapered)
        signal_intensity_losses = np.nanmean(energy_tapered, axis=tuple(range(dim))) / np.nanmean(energy, axis=tuple(
            range(dim)))

    else:
        ukdata = np.fft.fftn(udata, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))
        signal_intensity_losses = np.ones(duration)
    # compute E(k)
    ek = np.zeros(ukdata[0].shape)

    for i in range(dim):
        ek[...] += np.abs(ukdata[i, ...]) ** 2 / n_samples
    ek /= 2.

    if correct_signal_loss:
        # if window is not None:
        #     if dim == 2:
        #         xx, yy = get_equally_spaced_grid(udata, spacing=dx)
        #         window_arr = get_window_radial(xx, yy, wtype=window)
        #         signal_intensity_loss = np.nanmean(window_arr)
        #     elif dim == 3:
        #         xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
        #         window_arr = get_window_radial(xx, yy, wtype=window)
        #         signal_intensity_loss = np.nanmean(window_arr)
        # else:
        #     signal_intensity_loss = 1.
        # ek /= signal_intensity_loss
        for t in range(duration):
            # print signal_intensity_losses[t]
            ek[..., t] = ek[..., t] / signal_intensity_losses[t]
    if dim == 2:
        height, width, duration = ek.shape
        kx = np.fft.fftfreq(width, d=dx) * 2 * np.pi  # ANGULAR FREQUENCY (WAVENUMBER) NOT FREQ(INVERSE LENGTH)
        ky = np.fft.fftfreq(height, d=dy) * 2 * np.pi
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx, kyy

        # Output the SPECTRAL DENSITY
        # ... DFT outputs the integrated density (which is referred as POWER) in a pixel(delta_kx * delta_ky)
        # ... But energy spectrum is indeed plotting the SPECTRAL DENSITY!
        deltakx, deltaky = kx[1] - kx[0], ky[1] - ky[0]
        ek = ek / (deltakx * deltaky)
        return ek, np.asarray([kxx, kyy])

    elif dim == 3:
        height, width, depth, duration = ek.shape
        kx = np.fft.fftfreq(width, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(height, d=dy) * 2 * np.pi
        kz = np.fft.fftfreq(depth, d=dz) * 2 * np.pi
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)

        # Output the SPECTRAL DENSITY
        # ... DFT outputs the integrated density (which is referred as POWER) in a pixel(delta_kx * delta_ky)
        # ... But energy spectrum is indeed plotting the SPECTRAL DENSITY!
        deltakx, deltaky, deltakz = kx[1] - kx[0], ky[1] - ky[0], kz[1] - kz[0]
        ek = ek / (deltakx * deltaky * deltakz)

        return ek, np.asarray([kxx, kyy, kzz])


def get_energy_spectrum_old(udata, x0=0, x1=None, y0=0, y1=None,
                            z0=0, z1=None, dx=None, dy=None, dz=None, nkout=None,
                            window=None, correct_signal_loss=True, remove_undersampled_region=True,
                            cc=1.75, notebook=True):
    """
    DEPRECATED: TM cleaned up the code, and improved the literacy and transparency of the algorithm- TM (Sep 2020)

    Returns 1D energy spectrum from velocity field data
    ... The algorithm implemented in this function is VERY QUICK because it does not use the two-point  autorcorrelation tensor.
    ... Instead, it converts u(kx, ky, kz)u*(kx, ky, kz) into u(kr)u*(kr). (here * dentoes the complex conjugate)
    ... CAUTION: Must provide udata with aspect ratio ~ 1
    ...... The conversion process induces unnecessary error IF the dimension of u(kx, ky, kz) is skewed.
    ...... i.e. Make udata.shape like (800, 800), (1024, 1024), (512, 512) for accurate results.
    ... KNOWN ISSUES:
    ...... This function returns a bad result for udata with shape like (800, 800, 2)


    Parameters
    ----------
    udata: nd array
    epsilon: nd array or float, default: None
        dissipation rate used for scaling energy spectrum
        If not given, it uses the values estimated using the rate-of-strain tensor
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.
    remove_undersampled_region: bool, default: True
        If True, it will not sample the region with less statistics.
    cc: float, default: 1.75
        A numerical factor to compensate for the signal loss due to approximations.
        ... cc=1.75 was obtained from the JHTD data.
    Returns
    -------
    e_k: numpy array
        Energy spectrum with shape (number of data points, duration)
    e_k_err: numpy array
        Energy spectrum error with shape (number of data points, duration)
    kk: numpy array
        Wavenumber with shape (number of data points, duration)

    """
    print('get_energy_spectrum_old(): is DEPRECATED since 09/01/20')
    print('... Still works perfectly. Yet, TM highly recommends to use the updated function: get_energy_spectrum()')

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    def delete_masked_elements(data, mask):
        """
        Deletes elements of data using mask, and returns a 1d array
        Parameters
        ----------
        data: N-d array
        mask: N-d array, bool

        Returns
        -------
        compressed_data

        """
        data_masked = ma.array(data, mask=mask)
        compressed_data = data_masked.compressed()
        '...Reduced data using a given mask'
        return compressed_data

    def convert_nd_spec_to_1d(e_ks, ks, nkout=None, cc=1.75):
        """
        Convert the results of get_energy_spectrum_nd() into a 1D spectrum
        ... This is actually a tricky problem.
        Importantly, this will output the SPECTRAL DENSITY
        not power which is integrated spectral density (i.e.- spectral density * delta_kx * delta_ky * delta_ky.)
        ... Ask Takumi for derivation. The derivation goes like this.
        ...... 1. Start with the Parseval's theorem.
        ...... 2. Write the discretized equation about the TKE: Average TKE = sum deltak * E(k)
        ...... 3. Using 1, write down the avg TKE
        ...... 4. Equate 2 and 3. You get e_k1d * jacobian / (n_samples * deltak)
        ......   IF deltak = deltakr where deltakr = np.sqrt(deltakx**2 + deltaky**2) for 2D
        ......   where e_k1d is just a histogram value obtained from the DFT result (i.e. POWER- spectral density integrated over a px)
        ...... 5. Finally, convert this into the SPECTRAL DENSITY. This is two-fold.
        ...... 5.1.
        ......   e_k1d * jacobian / (n_samples * deltak) is not necessarily the correct density
        ......   if deltak is not equal to deltakr.
        ......   This is because e_k1d comes from the histogram of the input velocity field.
        ......   One can show that the correction is just (deltak / deltakr) ** dim
        ...... 5.2
        ......   After 5.1, this is finally the integrated power between k and k + deltak
        ......   Now divide this by deltak to get the spectral density.
        Parameters
        ----------
        e_ks
        ks
        nkout
        d: int/float, DIMENSION OF THE FLOW (NOT DIMENSION OF AVAILABLE VELOCITY FIELD)
            ... For 3D turbulence, d = 3
                ... d is equal to 3 even if udata is an 2D field embedded in an actual 3D field,
            ... For 2D turbulence, d = 2

        Returns
        -------

        """
        dim = ks.shape[0]
        duration = e_ks.shape[-1]
        if dim == 2:
            deltakx, deltaky = ks[0, 0, 1] - ks[0, 0, 0], \
                               ks[1, 1, 0] - ks[1, 0, 0]
            e_ks *= deltakx * deltaky  # use the raw DFT outputs (power=integrated density over a px)
            deltakr = np.sqrt(deltakx ** 2 + deltaky ** 2)  # radial k spacing of the velocity field
            dx, dy = 2. * np.pi / ks[0, 0, 0] * -0.5, 2. * np.pi / ks[1, 0, 0] * -0.5

        if dim == 3:
            deltakx, deltaky, deltakz = ks[0, 0, 1, 0] - ks[0, 0, 0, 0], \
                                        ks[1, 1, 0, 0] - ks[1, 0, 0, 0], \
                                        ks[2, 0, 0, 1] - ks[2, 0, 0, 0]
            e_ks *= deltakx * deltaky * deltakz  # use the raw DFT outputs (power=integrated density over a px)
            deltakr = np.sqrt(deltakx ** 2 + deltaky ** 2 + deltakz ** 2)  # radial k spacing of the velocity field
            dx, dy, dz = 2. * np.pi / ks[0, 0, 0] * -0.5, 2. * np.pi / ks[1, 0, 0] * -0.5, 2. * np.pi / ks[
                2, 0, 0] * -0.5

        kk = np.zeros((ks.shape[1:]))
        for i in range(dim):
            kk += ks[i, ...] ** 2
        kk = np.sqrt(kk)  # radial k

        if nkout is None:
            nkout = int(np.max(ks.shape[1:]) * 0.8)
        shape = (nkout, duration)

        e_k1ds = np.empty(shape)
        e_k1d_errs = np.empty(shape)
        k1ds = np.empty(shape)

        if remove_undersampled_region:
            kx_max, ky_max = np.nanmax(ks[0, ...]), np.nanmax(ks[1, ...])
            k_max = np.nanmin([kx_max, ky_max])
            if dim == 3:
                kz_max = np.nanmax(ks[2, ...])
                k_max = np.nanmin([k_max, kz_max])

        for t in range(duration):
            # flatten arrays to feed to binned_statistic\
            kk_flatten, e_knd_flatten = kk.flatten(), e_ks[..., t].flatten()

            if remove_undersampled_region:
                mask = np.abs(kk_flatten) > k_max
                kk_flatten = delete_masked_elements(kk_flatten, mask)
                e_knd_flatten = delete_masked_elements(e_knd_flatten, mask)

            # get a histogram
            k_means, k_edges, binnumber = binned_statistic(kk_flatten, kk_flatten, statistic='mean', bins=nkout)
            k_binwidth = (k_edges[1] - k_edges[0])
            k1d = k_edges[1:] - k_binwidth / 2
            e_k1d, _, _ = binned_statistic(kk_flatten, e_knd_flatten, statistic='mean', bins=nkout)
            e_k1d_err, _, _ = binned_statistic(kk_flatten, e_knd_flatten, statistic='std', bins=nkout)

            # # WEIGHTED AVERAGE
            # ke_k1d, _, _ = binned_statistic(kk_flatten, kk_flatten * e_knd_flatten, statistic='mean', bins=nkout)
            # e_k1d = ke_k1d / k1d
            # ke_k1d_err, _, _ = binned_statistic(kk_flatten, kk_flatten * e_knd_flatten, statistic='std', bins=nkout)
            # e_k1d_err = ke_k1d_err / k1d

            # One must fix the power by some numerical factor due to the DFT and the definition of E(k)
            n_samples = len(kk_flatten)
            deltak = k1d[1] - k1d[0]

            if dim == 2:
                jacobian = 2 * np.pi * k1d
            elif dim == 3:
                jacobian = 4 * np.pi * k1d ** 2

            # Insert to a big array
            # ... A quick derivation of this math is given in the docstring.
            k1ds[..., t] = k1d
            # OLD stuff
            # e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltak)
            # e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltak)
            # print deltak
            # Old stuff 2: scaling that works?
            # e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltak) * (deltak / deltakr) ** dim / deltak
            # e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltak) * (deltak / deltakr) ** dim / deltak
            # print(dx, dy, deltakr, deltakx * dx * ks.shape[2])
            print(deltakr, deltak)
            # 2019-2020 August
            # e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltakr ** 2) * cc
            # e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltakr ** 2) * cc

            # # Update in Aug, 2020- TM
            e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltakr ** 2) * cc
            e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltakr ** 2) * cc
        return e_k1ds, e_k1d_errs, k1ds

    dim, duration = len(udata), udata.shape[-1]

    e_ks, ks = get_energy_spectrum_nd_old(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz,
                                          window=window, correct_signal_loss=correct_signal_loss)
    e_k, e_k_err, kk = convert_nd_spec_to_1d(e_ks, ks, nkout=nkout, cc=cc)

    # #### NORMALIZATION IS NO LONGER NEEDED #### - Takumi, Apr 2019
    # # normalization
    # energy_avg, energy_avg_err = get_spatial_avg_energy(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
    #
    # for t in range(duration):
    #     I = np.trapz(e_k[0:, t], kk[0:, t])
    #     print I
    #     N = I / energy_avg[t] # normalizing factor
    #     e_k[:, t] /= N
    #     e_k_err[:, t] /= N

    if notebook:
        from tqdm import tqdm as tqdm

    return e_k, e_k_err, kk


def get_1d_energy_spectrum_old(udata, k='kx', x0=0, x1=None, y0=0, y1=None,
                               z0=0, z1=None, dx=None, dy=None, dz=None,
                               window=None, correct_signal_loss=True, debug=True):
    """
    Returns 1D energy spectrum from velocity field data

    Parameters
    ----------
    udata: nd array
    k: str, default: 'kx'
        string to specify the direction along which the given velocity field is Fourier-transformed
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of available window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    eiis: numpy array
        eiis[0] = E11, eiis[1] = E22
        ... 1D energy spectra with argument k="k" (kx by default)
    eii_errs: numpy array:
        eiis[0] = E11_error, eiis[1] = E22_error
    k: 1d numpy array
        Wavenumber with shape (number of data points, )
        ... Unlike get_energy_spectrum(...), this method NEVER outputs the wavenumber array with shape (number of data points, duration)
    """
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim, duration = len(udata), udata.shape[-1]
    if dim == 2:
        ux, uy = udata[0, y0:y1, x0:x1, :], udata[1, y0:y1, x0:x1, :]
        height, width, duration = ux.shape
        udata_tmp = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        ux, uy, uz = udata[0, y0:y1, x0:x1, z0:z1, :], udata[1, y0:y1, x0:x1, z0:z1, :], udata[2, y0:y1, x0:x1, z0:z1,
                                                                                         :]
        height, width, depth, duration = ux.shape
        udata_tmp = udata[:, y0:y1, x0:x1, z0:z1, :]
    else:
        raise ValueError('... Error: Invalid dimension is given. Use 2 or 3 for the number of spatial dimensions. ')

    if k == 'kx':
        ax_ind = 1  # axis number to take 1D DFT
        n = width
        d = dx
        if dim == 2:
            ax_ind_for_avg = 0  # axis number(s) to take statistics (along y)
        elif dim == 3:
            ax_ind_for_avg = (0, 2)  # axis number(s) to take statistics  (along y and z)
    elif k == 'ky':
        ax_ind = 0  # axis number to take 1D DFT
        n = height
        d = dy
        if dim == 2:
            ax_ind_for_avg = 1  # axis number(s) to take statistics  (along x)
        elif dim == 3:
            ax_ind_for_avg = (1, 2)  # axis number(s) to take statistics  (along x and z)
    elif k == 'kz':
        ax_ind = 2  # axis number to take 1D DFT
        n = depth
        d = dz
        ax_ind_for_avg = (0, 1)  # axis number(s) along which statistics is computed  (along x and y)
    n_samples = ux.shape[ax_ind]

    # Apply a hamming window to get lean FFT spectrum for aperiodic signals
    if window is not None:
        duration = udata.shape[-1]
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1)
            udata_tapered = np.empty_like(udata_tmp)
            for i in range(dim):
                udata_tapered[i, ...] = udata_tmp[i, ...] * windows
            ux, uy = udata_tapered
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0,
                                        z1=z1)
            udata_tapered = np.empty_like(udata_tmp)
            for i in range(dim):
                udata_tapered[i, ...] = udata_tmp[i, ...] * windows
            ux, uy, uz = udata_tapered

    if correct_signal_loss:
        if window is not None:
            if dim == 2:
                xx, yy = get_equally_spaced_grid(udata, spacing=dx)
                window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
                signal_intensity_loss = np.nanmean(window_arr)
            elif dim == 3:
                xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
                window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
                signal_intensity_loss = np.nanmean(window_arr)
        else:
            signal_intensity_loss = 1.
    # E11
    ux_k = np.fft.fft(ux, axis=ax_ind)
    ux_k = np.fft.fftshift(ux_k)
    e11_nd = np.abs(ux_k * np.conj(ux_k)) / n_samples  # \sum(ux**2) = \sum(e11_nd)- Parseval's theorem
    e11_nd /= n_samples  # Power (integrated spectral density) -> spectral density
    e11 = np.nanmean(e11_nd, axis=ax_ind_for_avg)
    e11_err = np.nanstd(e11_nd, axis=ax_ind_for_avg)

    # E22
    uy_k = np.fft.fft(uy, axis=ax_ind)
    uy_k = np.fft.fftshift(uy_k)
    e22_nd = np.abs(uy_k * np.conj(uy_k)) / n_samples  # \sum(uy**2) = \sum(e22_nd)- Parseval's theorem
    e22_nd /= n_samples  # Power (integrated spectral density) -> spectral density
    e22 = np.nanmean(e22_nd, axis=ax_ind_for_avg)
    e22_err = np.nanstd(e22_nd, axis=ax_ind_for_avg)

    # Get an array for wavenumber
    k = np.fft.fftfreq(n, d=d) * 2 * np.pi  # shape=(n, duration)
    k = np.fft.fftshift(k)
    deltak = k[1] - k[0]
    if dim == 3:
        # E33
        uz_k = np.fft.fft(uz, axis=ax_ind)
        uz_k = np.fft.fftshift(uz_k)
        e33_nd = np.abs(uz_k * np.conj(uz_k)) / n_samples  # \sum(uzy*2) = \sum(e33_nd)- Parseval's theorem
        e33_nd /= n_samples  # Power (integrated spectral density) -> spectral density
        e33 = np.nanmean(e33_nd, axis=ax_ind_for_avg)
        e33_err = np.nanstd(e33_nd, axis=ax_ind_for_avg)

        eiis, eii_errs = np.array([e11, e22, e33]) * 2, np.array([e11_err, e22_err, e33_err]) * 2
    elif dim == 2:
        eiis, eii_errs = np.array([e11, e22]) * 2, np.array([e11_err, e22_err]) * 2
    else:
        raise ValueError('... 1d spectrum: Check the dimension of udata! It must be 2 or 3!')

    # 1. Convert power to spectral density
    # ... DFT outputs the integrated power between k and k + deltak
    # ... One must divide the integrated power by deltak to account for this.
    for i in range(dim):
        eiis[i] *= 1 / deltak
        eii_errs[i] *= 1 / deltak

    # Windowing causes the loss of the signal (energy.)
    # ... This compensates for the loss.
    if correct_signal_loss:
        for i in range(dim):
            eiis[i] /= signal_intensity_loss
            eii_errs[i] /= signal_intensity_loss
    if debug:
        print(
            'get_1d_energy_spectrum_old(): debug is set True. It will check the property \int_0^\infty Eii = 2 <ui ui>')
        for i in range(dim):
            ui2_tavg = np.nanmean(udata[i, ...] ** 2, axis=tuple(range(dim)))
            k_i, eiis_i = clean_data_interp1d(eiis[i, ...], k)
            integral_i = np.trapz(eiis_i, x=k_i, axis=0)
            print('... <u%d squared> / integral of E_%d%d: ' % (i + 1, i + 1, i + 1), ui2_tavg / integral_i)
    return eiis, eii_errs, k


#### Energy spectra computation (UP-TO-DATE)
def get_energy_spectrum_nd_ver2(udata, x0=0, x1=None, y0=0, y1=None,
                                z0=0, z1=None, dx=None, dy=None, dz=None,
                                window=None, correct_signal_loss=True,
                                return_in='spectral density'):
    """
    Returns ND energy spectrum from velocity data (FFT of a velocity field)
    ... Returns the SPECTRUM DENSITY not power (integrated spectral density)
    ... Few Important Tips:
        ... np.fft.fft() returns POWER not spectrum density
            ... If you want to check Parseval's theorem, you must use the POWER not SPECTRAL DENSITY
                Parseval's theorem:
                    ukdata =  np.fft.fft(udata, axis=(1, dim+1)
                    1/n_samples * np.nansum( ukdata * np.conjucate(ukdata)) ) = np.nansum(udata ** 2)
        ... np.fft.fft() returns the DFT in frequency space (NOT ANGULAR FREQUENCY space)
        ... To convert the power (DFT output) to spectral density, one must do following.
                ukdata /= (dfx * dfy * dfz) * n_samples
            ... where dfx = 1 / dx, etc. and n_samples is the number of samples actually discrete-fourie transformed.

    Parameters
    ----------
    udata: nd array
    dx: data spacing in x (units: mm/px)
    dy: data spacing in y (units: mm/px)
    dz: data spacing in z (units: mm/px)
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    energy_fft: nd array with shape (height, width, duration) or (height, width, depth, duration)
    ks: nd array with shape (ncomponents, height, width, duration) or (ncomponents, height, width, depth, duration)

    Example
    -----------------
    nx, ny = 100, 100
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, 4*np.pi, ny)
    dx, dy = x[1]- x[0], y[1]-y[0]

    # Position grid
    xx, yy = np.meshgrid(x, y)

    # In Fourier space, energy will have a peak at (kx, ky) = (+/- 5, +/- 2)
    ux = np.sin(2.5*xx + yy)
    uy = np.sin(yy) * 0
    udata_test = np.stack((ux, uy))
    ek, ks = get_energy_spectrum_nd(udata_test, dx=dx, dy=dy)
    graph.color_plot(xx, yy, (ux**2 + uy**2 ) /2., fignum=2, subplot=121)
    fig22, ax22, cc22 = graph.color_plot(ks[0], ks[1], ek.real[..., 0], fignum=2, subplot=122, figsize=(16, 8))
    graph.setaxes(ax22, -10, 10, -10, 10)

    """
    if dx is None or dy is None:
        print('ERROR: dx or dy is not provided! dx is grid spacing in real space.')
        print('... k grid will be computed based on this spacing! Please provide.')
        raise ValueError
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    dim = len(udata)
    udata = fix_udata_shape(udata)
    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        if dz is None:
            print('ERROR: dz is not provided! dx is grid spacing in real space.')
            print('... k grid will be computed based on this spacing! Please provide.')
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    n_samples = 1
    for d in range(dim):
        n_samples *= udata.shape[d + 1]

    # WINDOWING
    duration = udata.shape[-1]
    if window is not None or window != 'rectangle':
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows

        # PERFORM DFT on the windowed field
        # DFT returns the POWER (INTEGRATED SPECTRAL DENSITY)
        ukdata = np.fft.fftn(udata_tapered, axes=list(range(1, dim + 1)))  # POWER
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))

        energy, energy_tapered = get_energy(udata), get_energy(udata_tapered)
        signal_intensity_losses = np.nanmean(energy_tapered, axis=tuple(range(dim))) / np.nanmean(energy, axis=tuple(
            range(dim)))

    else:
        ukdata = np.fft.fftn(udata, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))
        signal_intensity_losses = np.ones(duration)
    ##################################################
    # CONVERT TO SPECTRAL DENSITY
    ##################################################
    if dim == 2:
        dim, height, width, duration = ukdata.shape
        kx = np.fft.fftshift(np.fft.fftfreq(width, d=dx)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(height, d=dy)) * 2 * np.pi
        deltakx, deltaky = kx[1] - kx[0], ky[1] - ky[0]
        dfx, dfy = deltakx / 2 / np.pi, deltaky / 2 / np.pi
        dz = dfz = deltakz = 1  # for convenience
    elif dim == 3:
        dim, height, width, depth, duration = ukdata.shape
        kx = np.fft.fftshift(np.fft.fftfreq(width, d=dx)) * 2 * np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(height, d=dy)) * 2 * np.pi
        kz = np.fft.fftshift(np.fft.fftfreq(depth, d=dz)) * 2 * np.pi

        deltakx, deltaky, deltakz = kx[1] - kx[0], ky[1] - ky[0], kz[1] - kz[0]

        # Frequency: 1/dx, 1/dy, 1/dz
        dfx, dfy, dfz = deltakx / 2 / np.pi, deltaky / 2 / np.pi, deltakz / 2 / np.pi

    # CHOOSE WHETHER TO OUTPUT IN RAW SPECTRAL DFT OUTPUT OR SPECTRAL DENSITY
    # ... Raw DFT output: integrated spectral density (power)
    # ...... This is somnething you can use to check Parseval's theorem for example
    # ... For Espec calculation, spectral density is way easier to use.
    if return_in == 'spectral density':
        ukdata /= (dfx * dfy * dfz) * n_samples  # THIS IS THE CORRECT WAY TO CONVERT TO SPECTRAL DENSITY!

    ##################################################
    # compute E(\vec{k})
    ##################################################
    ek = np.zeros(ukdata[0].shape)

    for i in range(dim):
        ek[...] += np.abs(ukdata[i, ...]) ** 2
    ek /= 2.

    if correct_signal_loss:
        for t in range(duration):
            # print signal_intensity_losses[t]
            ek[..., t] = ek[..., t] / signal_intensity_losses[t]
    if dim == 2:
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx, kyy
        return ek, np.asarray([kxx, kyy])

    elif dim == 3:
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        return ek, np.asarray([kxx, kyy, kzz])


def get_energy_spectrum_ver2(udata, x0=0, x1=None, y0=0, y1=None,
                             z0=0, z1=None, dx=None, dy=None, dz=None, nkout=None,
                             window=None, correct_signal_loss=True, remove_undersampled_region=True,
                             cc=1, notebook=True, debug=False):
    """
    Returns 1D energy spectrum from velocity field data
    ... The algorithm implemented in this function is VERY QUICK because it does not use the two-point  autorcorrelation tensor.
    ... Instead, it converts u(kx, ky, kz)u*(kx, ky, kz) into u(kr)u*(kr). (here * dentoes the complex conjugate)
    ... CAUTION: Must provide udata with aspect ratio ~ 1
    ...... The conversion process induces unnecessary error IF the dimension of u(kx, ky, kz) is skewed.
    ...... i.e. Make udata.shape like (800, 800), (1024, 1024), (512, 512) for accurate results.
    ... KNOWN ISSUES:
    ...... This function returns a bad result for udata with shape like (800, 800, 2)


    Parameters
    ----------
    udata: nd array
    epsilon: nd array or float, default: None
        dissipation rate used for scaling energy spectrum
        If not given, it uses the values estimated using the rate-of-strain tensor
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.
    remove_undersampled_region: bool, default: True
        If True, it will not sample the region with less statistics.
    cc: float, default: 1.75
        A numerical factor to compensate for the signal loss due to approximations.
        ... cc=1.75 was obtained from the JHTD data.
    Returns
    -------
    e_k: numpy array
        Energy spectrum with shape (number of data points, duration)
    e_k_err: numpy array
        Energy spectrum error with shape (number of data points, duration)
    kk: numpy array
        Wavenumber with shape (number of data points, duration)

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    def delete_masked_elements(data, mask):
        """
        Deletes elements of data using mask, and returns a 1d array
        Parameters
        ----------
        data: N-d array
        mask: N-d array, bool

        Returns
        -------
        compressed_data

        """
        data_masked = ma.array(data, mask=mask)
        compressed_data = data_masked.compressed()
        '...Reduced data using a given mask'
        return compressed_data

    def convert_nd_spec_to_1d(ek_nd, ks, nkout=None, cc=1., d=3,
                              debug=False, udata=None):
        """


        Parameters
        ----------
        ek_nd
        ks
        nkout
        d: int/float, DIMENSION OF THE FLOW (NOT DIMENSION OF AVAILABLE VELOCITY FIELD)
            ... For 3D turbulence, d = 3
                ... d is equal to 3 even if udata is an 2D field embedded in an actual 3D field,
            ... For 2D turbulence, d = 2

        Returns
        -------

        """
        if debug:
            print('get_energy_spectrum/convert_nd_spec_to_1d: You set debug=True. '
                  'Will check the energy conservation (Parseval\'s theorem and isotropy approximation')
            if udata is None:
                print('... udata is required for debugging. Not supplied, terminating the process...')
                sys.exit()

        dim = ks.shape[0]
        duration = ek_nd.shape[-1]

        # INITIALIZATIONS
        if nkout is None:
            nkout = int(np.max(ks.shape[1:]) * 1.0)
        shape = (nkout, duration)

        eks = np.empty(shape)
        ek_errs = np.empty(shape)
        k1ds = np.empty(shape)

        # PREPARATION
        if dim == 2:
            kx = ks[0][0, :]
            ky = ks[1][:, 0]
            deltakx, deltaky = kx[1] - kx[0], ky[1] - ky[0]
            dfx, dfy = deltakx / 2 / np.pi, deltaky / 2 / np.pi
            depth, dz, dfz = 1, 1, 1  # for convenience for later computation
            kxx, kyy = np.meshgrid(kx, ky)
            kk = np.sqrt(kxx ** 2 + kyy ** 2)  # radial

            dx, dy = 2. * np.pi / ks[0, 0, 0] * -0.5, 2. * np.pi / ks[1, 0, 0] * -0.5
        if dim == 3:
            kx = ks[0][0, :, 0]
            ky = ks[1][:, 0, 0]
            kz = ks[2][0, 0, :]
            deltakx, deltaky, deltakz = kx[1] - kx[0], ky[1] - ky[0], kz[1] - kz[0]
            dfx, dfy, dfz = deltakx / 2 / np.pi, deltaky / 2 / np.pi, deltakz / 2 / np.pi
            kxx, kyy, kzz = np.meshgrid(ky, kx, kz)

            kk = np.sqrt(kxx ** 2 + kyy ** 2 + kzz ** 2)  # radial
            dx, dy, dz = 2. * np.pi / ks[0, 0, 0, 0] * -0.5, 2. * np.pi / ks[1, 0, 0, 0] * -0.5, 2. * np.pi / ks[
                2, 0, 0, 0] * -0.5
        n_samples = kk.size

        if remove_undersampled_region:
            kx_max, ky_max = np.nanmax(ks[0, ...]), np.nanmax(ks[1, ...])
            k_max = np.nanmin([kx_max, ky_max])
            if dim == 3:
                kz_max = np.nanmax(ks[2, ...])
                k_max = np.nanmin([k_max, kz_max])

        for t in range(duration):
            # flatten arrays to feed to binned_statistic\
            kk_flatten, e_knd_flatten = kk.flatten(), ek_nd[..., t].flatten()

            if remove_undersampled_region:
                mask = np.abs(kk_flatten) > k_max
                kk_flatten = delete_masked_elements(kk_flatten, mask)
                e_knd_flatten = delete_masked_elements(e_knd_flatten, mask)

            # print(kk.shape, ek_nd.shape, ukdata.shape, udata.shape)
            e_k1d, k_edges, binnumber = binned_statistic(kk_flatten, e_knd_flatten, statistic='mean', bins=nkout)
            e_k1d_std, k_edges, binnumber = binned_statistic(kk_flatten, e_knd_flatten, statistic='std', bins=nkout)
            counts, k_edges, binnumber = binned_statistic(kk_flatten, e_knd_flatten, statistic='count', bins=nkout)
            k_binwidth = (k_edges[1] - k_edges[0])
            k1d = k_edges[1:] - k_binwidth / 2

            if dim == 3:
                jacobian = 4 * np.pi * (k1d / 2 / np.pi) ** 2
            elif dim == 2:
                jacobian = 2 * np.pi * (k1d / 2 / np.pi)

            # ENERGY SPECTRUM
            freq1d_i, e_freq1d_i = clean_data_interp1d(e_k1d * jacobian, k1d / 2 / np.pi)
            k1d_i, e_k1d_i = freq1d_i * 2 * np.pi, e_freq1d_i[:, 0] / 2 / np.pi
            # ENERGY SPECTRUM ERROR
            # e_freq1d_i_err, e_freq1d_i = clean_data_interp1d(e_k1d_std * jacobian * np.sqrt(counts), k1d / 2 / np.pi)
            # k1d_i, e_k1d_i_err = freq1d_i * 2 * np.pi, e_freq1d_i_err[:, 0] / 2 / np.pi
            e_k1d_i_err = e_k1d_std / 2 / np.pi * np.sqrt(counts)

            # Finally, e(k) describes the distribution of AVG energy- divide it by volume
            volume = n_samples * dx * dy * dz
            eks[:, t] = e_k1d_i / volume * cc
            ek_errs[:, t] = e_k1d_i_err / volume * cc

            if debug:
                total_energy = np.nansum(udata ** 2) * dx * dy * dz / 2
                total_energy_iso = np.trapz(e_freq1d_i[:, 0], freq1d_i[:, 0])
                print('total energy in real space: ', total_energy)
                print('total energy in Fourier space: ', np.nansum(ek_nd) * dfx * dfy * dfz)
                print(
                    'total energy in Fourier space (isotropic approximation), kmin, kmax: ', total_energy_iso,
                    k1d_i[0, 0],
                    k1d_i[-1, 0])
                print('volume', volume)
        k1ds = k1d_i[:, 0]
        return eks, ek_errs, k1ds

    dim, duration = len(udata), udata.shape[-1]

    e_ks, ks = get_energy_spectrum_nd_ver2(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz,
                                           window=window, correct_signal_loss=correct_signal_loss)
    e_k, e_k_err, kk = convert_nd_spec_to_1d(e_ks, ks, nkout=nkout, cc=cc, udata=udata, debug=debug)

    if notebook:
        from tqdm import tqdm as tqdm
    return e_k, e_k_err, kk


def get_1d_energy_spectrum(udata, k='kx', x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None,
                           window=None, correct_signal_loss=True, debug=False, verbose=True):
    """
    Returns 1D energy spectrum from velocity field data

    Parameters
    ----------
    udata: nd array
    k: str, default: 'kx'
        string to specify the direction along which the given velocity field is Fourier-transformed
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of available window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    eiis: numpy array
        eiis[0] = E11, eiis[1] = E22
        ... 1D energy spectra with argument k="k" (kx by default)
    eii_errs: numpy array:
        eiis[0] = E11_error, eiis[1] = E22_error
    k: 1d numpy array
        Wavenumber with shape (number of data points, )
        ... Unlike get_energy_spectrum(...), this method NEVER outputs the wavenumber array with shape (number of data points, duration)
    """
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim, duration = len(udata), udata.shape[-1]
    if dim == 2:
        ux, uy = udata[0, y0:y1, x0:x1, :], udata[1, y0:y1, x0:x1, :]
        height, width, duration = ux.shape
        udata_tmp = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        ux, uy, uz = udata[0, y0:y1, x0:x1, z0:z1, :], udata[1, y0:y1, x0:x1, z0:z1, :], udata[2, y0:y1, x0:x1, z0:z1,
                                                                                         :]
        height, width, depth, duration = ux.shape
        udata_tmp = udata[:, y0:y1, x0:x1, z0:z1, :]
    else:
        raise ValueError('... Error: Invalid dimension is given. Use 2 or 3 for the number of spatial dimensions. ')

    if k == 'kx':
        ax_ind = 1  # axis number to take 1D DFT
        n = width
        d = dx
        if dim == 2:
            ax_ind_for_avg = 0  # axis number(s) to take statistics (along y)
        elif dim == 3:
            ax_ind_for_avg = (0, 2)  # axis number(s) to take statistics  (along y and z)
    elif k == 'ky':
        ax_ind = 0  # axis number to take 1D DFT
        n = height
        d = dy
        if dim == 2:
            ax_ind_for_avg = 1  # axis number(s) to take statistics  (along x)
        elif dim == 3:
            ax_ind_for_avg = (1, 2)  # axis number(s) to take statistics  (along x and z)
    elif k == 'kz':
        ax_ind = 2  # axis number to take 1D DFT
        n = depth
        d = dz
        ax_ind_for_avg = (0, 1)  # axis number(s) along which statistics is computed  (along x and y)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=d))
    deltaf = freq[1] - freq[0]
    n_samples = ux.shape[ax_ind]

    # Apply a hamming window to get lean FFT spectrum for aperiodic signals
    if window is not None:
        duration = udata.shape[-1]
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1)
            udata_tapered = np.empty_like(udata_tmp)
            for i in range(dim):
                udata_tapered[i, ...] = udata_tmp[i, ...] * windows
            ux, uy = udata_tapered
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0,
                                        z1=z1)
            udata_tapered = np.empty_like(udata_tmp)
            for i in range(dim):
                udata_tapered[i, ...] = udata_tmp[i, ...] * windows
            ux, uy, uz = udata_tapered

    if correct_signal_loss:
        if window is not None:
            if dim == 2:
                xx, yy = get_equally_spaced_grid(udata, spacing=dx)
                window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
                signal_intensity_loss = np.nanmean(window_arr)
            elif dim == 3:
                xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
                window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
                signal_intensity_loss = np.nanmean(window_arr)
        else:
            signal_intensity_loss = 1.
    # E11
    ux_k = np.fft.fftshift(np.fft.fft(ux, axis=ax_ind))
    if verbose or debug:
        print('deltaf, nsamples, d', deltaf, n_samples, d)
        print('dx, 1/deltaf/n', d, 1 / deltaf / n_samples)
        print('Parseval (ux)', np.nansum(ux ** 2) / (np.nansum(np.abs(ux_k * np.conj(ux_k))) / n_samples))
    ux_k /= 2 * np.pi * deltaf * n_samples  # convert to spectral density (Power per wavenumber)
    e11_nd = np.abs(ux_k * np.conj(ux_k)) * 2  # e11 is defined as twice as the square of the 1D FT of u1
    e11 = np.nanmean(e11_nd, axis=ax_ind_for_avg) * (2 * np.pi * deltaf) ** (dim - 1)
    e11_err = np.nanstd(e11_nd, axis=ax_ind_for_avg) / np.sqrt(np.product(e11_nd.shape[ax_ind_for_avg]))

    # E22
    uy_k = np.fft.fftshift(np.fft.fft(uy, axis=ax_ind))
    if verbose or debug:
        print('Parseval (uy)', np.nansum(uy ** 2) / (np.nansum(np.abs(uy_k * np.conj(uy_k))) / n_samples))
    uy_k /= 2 * np.pi * deltaf * n_samples  # convert to spectral density
    e22_nd = np.abs(uy_k * np.conj(uy_k)) * 2 * (2 * np.pi * deltaf) ** (
                dim - 1)  # e22 is defined as twice as the square of the 1D FT of u2
    e22 = np.nanmean(e22_nd, axis=ax_ind_for_avg)
    e22_err = np.nanstd(e22_nd, axis=ax_ind_for_avg) / np.sqrt(np.product(e22_nd.shape[ax_ind_for_avg]))

    # Get an array for wavenumber
    k = np.fft.fftfreq(n, d=d) * 2 * np.pi  # shape=(n, duration)
    k = np.fft.fftshift(k)

    deltak = k[1] - k[0]
    if dim == 3:
        # E33
        uz_k = np.fft.fftshift(np.fft.fft(uz, axis=ax_ind))
        uz_k /= 2 * np.pi * deltaf * n_samples  # convert to spectral density
        e33_nd = np.abs(uz_k * np.conj(uz_k)) * 2 * (2 * np.pi * deltaf) ** (
                    dim - 1)  # e33 is defined as twice as the square of the 1D FT of u3
        e33 = np.nanmean(e33_nd, axis=ax_ind_for_avg)
        e33_err = np.nanstd(e33_nd, axis=ax_ind_for_avg) / np.sqrt(np.product(e33_nd.shape[ax_ind_for_avg]))

        # Must divide by 2pi because np.fft.fft() performs in the frequency space (NOT angular frequency space)
        eiis, eii_errs = np.array([e11, e22, e33]), np.array([e11_err, e22_err, e33_err])
    elif dim == 2:
        # Must divide by 2pi^2 because np.fft.fft() performs in the frequency space (NOT angular frequency space)
        eiis, eii_errs = np.array([e11, e22]), np.array([e11_err, e22_err])
    else:
        raise ValueError('... 1d spectrum: Check the dimension of udata! It must be 2 or 3!')

    # Windowing causes the loss of the signal (energy.)
    # ... This compensates for the loss.
    if correct_signal_loss:
        for i in range(dim):
            eiis[i] /= signal_intensity_loss
            eii_errs[i] /= signal_intensity_loss
    if debug:
        print(
            'get_1d_energy_spectrum(): debug is set True. It will check the property: <ui ui>=\int_0^\infty Eii = 0.5 \int_{-infty}^\infty Eii ')
        for i in range(dim):
            ui2_tavg = np.nanmean(udata[i, ...] ** 2, axis=tuple(range(dim)))[0]
            k_i, eiis_i = clean_data_interp1d(eiis[i, ..., 0], k)
            integral_i = np.trapz(eiis_i[..., 0], x=k_i[:, 0], axis=0)
            print('...Frame0: <u%d squared> / integral of E_%d%d: ' % (i + 1, i + 1, i + 1), ui2_tavg / integral_i)
    return eiis, eii_errs, k


def get_dissipation_spectrum(udata, nu, x0=0, x1=None, y0=0, y1=None,
                             z0=0, z1=None, dx=None, dy=None, dz=None, nkout=None,
                             window='flattop', correct_signal_loss=True, notebook=True):
    """
    Returns dissipation spectrum D(k) = 2 nu k^2 E(k) where E(k) is the 1d energy spectrum

    Parameters
    ----------
    udata: nd array
    epsilon: nd array or float, default: None
        dissipation rate used for scaling energy spectrum
        If not given, it uses the values estimated using the rate-of-strain tensor
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of available window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    D_k: numpy array
        Dissipation spectrum with shape (number of data points, duration)
    D_k_err: numpy array
        Dissipation spectrum error with shape (number of data points, duration)
    k1d: numpy array
        Wavenumber with shape (number of data points, duration)

    """
    e_k, e_k_err, k1d = get_energy_spectrum(udata, x0=x0, x1=x1,
                                            y0=y0, y1=y1, z0=z0, z1=z1,
                                            dx=dx, dy=dy, dz=dz, nkout=nkout, window=window,
                                            correct_signal_loss=correct_signal_loss,
                                            notebook=notebook)
    # Plot dissipation spectrum
    D_k, D_k_err = 2 * nu * e_k * (k1d ** 2), 2 * nu * e_k_err * (k1d ** 2)
    return D_k, D_k_err, k1d


def get_rescaled_energy_spectrum(udata, epsilon=10 ** 5, nu=1.0034, x0=0, x1=None,
                                 y0=0, y1=None, z0=0, z1=None,
                                 dx=1, dy=1, dz=1, nkout=None,
                                 window=None, correct_signal_loss=True, notebook=True):
    """
    Returns SCALED energy spectrum E(k), its error, and wavenumber.
        - E(k) is sometimes called the 3D energy spectrum since it involves 3D FT of a velocity field.
        - ALWAYS greater than E11(k) and E22(k) which involve 1D FT of the velocity field along a specific direction.
        - Returns wavenumber with shape (# of data points, duration) instead of (# of data points, ).
        ... This seems redundant; however, values in the wavenumber array at a given time may fluctuate in principle
        due to a method how it computes the histogram (spectrum.) It should never fluctuate with the current method but
        is subject to a change by altering the method of determining the histogram. Therefore, this method outputs
        the wavenumber array with shape (# of data points, duration).

    Parameters
    ----------
    udata: nd array
    epsilon: nd array or float, default: None
        dissipation rate used for scaling energy spectrum
        If not given, it uses the values estimated using the rate-of-strain tensor
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of available window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    e_k_norm: numpy array
        Scaled energy spectrum with shape (number of data points, duration)
    e_k_err_norm: numpy array
        Scaled energy spectrum with shape (number of data points, duration)
    k_norm: numpy array
        Scaled energy spectrum with shape (number of data points, duration)
    """

    # get energy spectrum
    e_k, e_k_err, kk = get_energy_spectrum(udata, x0=x0, x1=x1,
                                           y0=y0, y1=y1, z0=z0, z1=z1,
                                           dx=dx, dy=dy, dz=dz, nkout=nkout, window=window, correct_signal_loss=True,
                                           notebook=notebook)

    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** (0.25)  # mm
    # print 'dissipation rate, Kolmogorov scale: ', epsilon, eta

    # # Subtlety: E(k) and E11(k) is not the same. E(k)=C epsilon^(2/3)k^(-5/3), E11(k)=C1 epsilon^(2/3)k^(-5/3)
    # # In iso, homo, turbulence, C1 = 18/55 C. (Pope 6.242)
    # c = 1.6
    # c1 = 18. / 55. * c

    k_norm = kk * eta
    e_k_norm = e_k[...] / ((epsilon * nu ** 5.) ** (0.25))
    e_k_err_norm = e_k_err[...] / ((epsilon * nu ** 5.) ** (0.25))

    return e_k_norm, e_k_err_norm, k_norm


def get_1d_rescaled_energy_spectrum(udata, epsilon=None, nu=1.0034, x0=0, x1=None,
                                    y0=0, y1=None, z0=0, z1=None,
                                    dx=1, dy=1, dz=1, notebook=True, window='flattop', correct_signal_loss=True):
    """
    Returns SCALED 1D energy spectra (E11 and E22)
    ... Applies the flattop window function by default
    ... Uses dissipation rate estimated by the rate-of-strain tensor unless given]


    Parameters
    ----------
    udata: nd array
    epsilon: nd array or float, default: None
        dissipation rate used for scaling energy spectrum
        If not given, it uses the values estimated using the rate-of-strain tensor
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of available window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.

    Returns
    -------
    eii_arr_norm: nd array
        Scaled spectral densities (Scaled E11 and Scaled E22)
    eii_err_arr_norm: nd array
        Scaled errors for (Scaled E11 and Scaled E22)
    k_norm: nd array
        Scaled wavenumber
    """
    dim = len(udata)
    duration = udata.shape[-1]
    # get energy spectrum
    eii_arr, eii_err_arr, k1d = get_1d_energy_spectrum(udata, x0=x0, x1=x1,
                                                       y0=y0, y1=y1, z0=z0, z1=z1,
                                                       dx=dx, dy=dy, dz=dz,
                                                       window=window, correct_signal_loss=correct_signal_loss)
    if epsilon is None:
        epsilon = get_epsilon_using_sij(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz)

    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** 0.25  # mm
    # print 'dissipation rate, Kolmogorov scale: ', epsilon, eta

    shape = (len(k1d), duration)
    k_norm = np.empty(shape)
    eii_arr_norm = np.empty_like(eii_arr)
    eii_err_arr_norm = np.empty_like(eii_err_arr)

    for t in range(duration):
        try:
            k_norm[:, t] = k1d * eta[t]
        except:
            k_norm[:, t] = k1d * eta

    for i in range(dim):
        eii_arr_norm = eii_arr[...] / ((epsilon * nu ** 5.) ** 0.25)
        eii_err_arr_norm = eii_err_arr[...] / ((epsilon * nu ** 5.) ** 0.25)

    return eii_arr_norm, eii_err_arr_norm, k_norm


def get_rescaled_dissipation_spectrum(udata, epsilon=10 ** 5, nu=1.0034, x0=0, x1=None,
                                      y0=0, y1=None, z0=0, z1=None,
                                      dx=1, dy=1, dz=1, nkout=None, notebook=True):
    """
    Return rescaled dissipation spectra
    D(k)/(u_eta^3) vs k * eta
    ... convention: k =  2pi/ L

    Parameters
    ----------
    udata: nd array
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise


    Returns
    -------
    D_k_norm: nd array
        Scaled dissipation spectrum values with shape (# of data points, duration)
    D_k_err_norm: nd array
        Scaled dissipation spectrum error with shape (# of data points, duration)
    k_norm: nd array
        Scaled wavenumber with shape (# of data points, duration)
    """
    # get dissipation spectrum
    D_k, D_k_err, k1d = get_dissipation_spectrum(udata, nu=nu, x0=x0, x1=x1,
                                                 y0=y0, y1=y1, z0=z0, z1=z1,
                                                 dx=dx, dy=dy, dz=dz, nkout=nkout, notebook=notebook)

    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** 0.25  # mm
    u_eta = (nu * epsilon) ** 0.25
    print('dissipation rate, Kolmogorov scale: ', epsilon, eta)

    k_norm = k1d * eta
    D_k_norm = D_k[...] / (u_eta ** 3)
    D_k_err_norm = D_k_err[...] / (u_eta ** 3)
    return D_k_norm, D_k_err_norm, k_norm


def scale_energy_spectrum(e_k, kk, epsilon=10 ** 5, nu=1.0034, e_k_err=None):
    """
    Scales raw energy spectrum by given dissipation rate and viscosity

    Parameters
    ----------
    e_k: numpy array
        spectral energy density
    kk: numpy array
        wavenumber
    epsilon: numpy array or float
        dissipation rate used for scaling. It could be 1D numpy array or float.
    nu numpy array or float
        viscosity used for scaling. It could be 1D numpy array or float.
    e_k_err: numpy array
        error of dissipation rate used for scaling. It could be 1D numpy array or float.
    Returns
    -------
    e_k_norm, k_norm if e_k_err is not given
    e_k_norm, e_k_err_norm, k_norm if e_k_err is given
    """
    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** (0.25)  # mm
    # print 'dissipation rate, Kolmogorov scale: ', epsilon, eta

    try:
        k_norm = kk * eta
    except:  # if eta is a numpy array because time-varying epsilon, k_norm will be a 2d array
        kk_ = np.repeat(kk[..., np.newaxis], eta.shape[-1], axis=1)
        k_norm = (kk_ * eta)
    e_k_norm = e_k[...] / ((epsilon * nu ** 5.) ** (0.25))
    if e_k_err is not None:
        e_k_err_norm = e_k_err[...] / ((epsilon * nu ** 5.) ** (0.25))
        return e_k_norm, e_k_err_norm, k_norm
    else:
        return e_k_norm, k_norm


def get_large_scale_vel_field(udata, kmax, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, window=None,
                              dx=None, dy=None, dz=None):
    """
    Returns a velocity field which satisfies k = sqrt(kx^2 + ky^2) < kmax in the original  field (udata)

    Parameters
    ----------
    udata: nd array
    ... velocity field data with shape (# of components, physical dimensions (width x height x depth), duration)
    kmax: float
    ... value of k below which spectrum is kept. i.e. cutoff k for the low-pass filter
    x0: int
    ... index used to specify a region of a  field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    x1: int
    ... index used to specify a region of a  field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    y0: int
    ... index used to specify a region of a  field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    y1: int
    ... index used to specify a region of a  field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    z0: int
    ... index used to specify a region of a  field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    z1: int
    ... index used to specify a region of a  field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    window: 2d/3d nd array
    ... window used to make input data periodic to surpress spectral leakage in FFT
    dx: float
    ... x spacing. used to compute k
    dy: float
    ... y spacing. used to compute k
    dz: float
    ... z spacing. used to compute k

    Returns
    -------
    udata_ifft: nd array
    ... low-pass filtered velocity field data
    coords: nd array
    ... if dim == 2, returns np.asarray([xgrid, ygrid])
    ... if dim == 3, returns np.asarray([xgrid, ygrid, zgrid])
    """
    if dx is None or dy is None:
        print('ERROR: dx or dy is not provided! dx is grid spacing in real space.')
        print('... k grid will be computed based on this spacing! Please provide.')
        raise ValueError
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    dim = len(udata)
    udata = fix_udata_shape(udata)
    if dim == 2:
        xx, yy = get_equally_spaced_grid(udata, spacing=dx)
        coords = np.asarray([xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]])
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        if dz is None:
            print('ERROR: dz is not provided! dx is grid spacing in real space.')
            print('... k grid will be computed based on this spacing! Please provide.')
            raise ValueError
        xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
        coords = np.asarray([xx[y0:y1, x0:x1, z0:z1], yy[y0:y1, x0:x1, z0:z1], zz[y0:y1, x0:x1, z0:z1]])
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    n_samples = 1
    for d in range(dim):
        n_samples *= udata.shape[d + 1]

    # Apply a window to suppress spectral leaking
    if window is not None:
        duration = udata.shape[-1]
        if dim == 2:
            windows = get_window_radial(xx, yy, wtype=window, duration=duration)
            udata_tapered = udata * windows
        elif dim == 3:
            windows = get_window_radial(xx, yy, zz=zz, wtype=window, duration=duration)
            udata_tapered = udata * windows
        else:
            raise ValueError('... dimension of udata must be 2 or 3!')
        ukdata = np.fft.fftn(udata_tapered, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))
    else:
        ukdata = np.fft.fftn(udata, axes=list(range(1, dim + 1)))
        ukdata = np.fft.fftshift(ukdata, axes=list(range(1, dim + 1)))

    if dim == 2:
        ncomp, height, width, duration = udata.shape
        # k space grid
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dy)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi  # Convert inverse length into wavenumber
        ks = np.asarray([kxx, kyy])
    elif dim == 3:
        ncomp, height, width, depth, duration = udata.shape
        # k space grid
        kx = np.fft.fftfreq(width, d=dx)
        ky = np.fft.fftfreq(height, d=dy)
        kz = np.fft.fftfreq(depth, d=dz)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
        ks = np.asarray([kxx, kyy, kzz])

    # Make radial k array
    kr = np.zeros_like(kxx)
    for i in range(dim):
        kr += ks[i] ** 2
    kr = np.sqrt(kr)
    # Make a mask such that kr > kmax
    mask = kr > kmax
    print('# of Masked Elements / total: %d / %d = %.4f' % (np.sum(mask), n_samples, np.sum(mask) / float(n_samples)))
    # Let uk array into a masked array
    ukdata = ma.asarray(ukdata)

    mask = np.repeat(mask[..., np.newaxis], duration, axis=dim)
    mask = np.repeat(mask[np.newaxis, ...], dim, axis=0)
    ukdata.mask = mask
    # Make all elements where kr > kmax = 0 (Low pass filter)
    ukdata = ukdata.filled(fill_value=0)

    # Inverse FT
    ukdata = np.fft.ifftshift(ukdata, axes=list(range(1, dim + 1)))
    udata_ifft = np.fft.ifftn(ukdata, axes=list(range(1, dim + 1))).real  # Use only a real part

    return udata_ifft, coords


########## DISSIPATION RATE ##########

def get_epsilon_using_sij(udata, dx=None, dy=None, dz=None, nu=1.004,
                          x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None):
    """
    Returns the dissipation rate computed using the rate-of-strain tensor

    sij: numpy array with shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    ... idea is... sij[spacial coordinates, time, tensor indices]
        e.g.-  sij(x, y, t) can be accessed by sij[y, x, t, i, j]
    ... sij = d ui / dxj
    Parameters
    ----------
    udata: nd array
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z

    Returns
    -------
    epsilon: numpy array
        dissipation rate
    """
    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, t0:t1]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        udata = udata[:, y0:y1, x0:x1, z0:z1, t0:t1]

    if dx is None:
        raise ValueError('... dx is None. Provide int or float.')
    if dy is None:
        raise ValueError('... dy is None. Provide int or float.')
    if dim == 3:
        if dz is None:
            raise ValueError('... dz is None. Provide int or float.')

    duidxj = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz)
    if dim == 3:
        sij, tij = decompose_duidxj(duidxj)
        epsilon_spatial = 2. * nu * np.nansum(sij ** 2, axis=(4, 5))
        epsilon = np.nanmean(epsilon_spatial, axis=tuple(range(dim)))
    elif dim == 2:
        # Estimate epsilon from 2D data (assuming isotropy)
        epsilon_0 = np.nanmean(duidxj[..., 0, 0] ** 2, axis=tuple(range(dim)))
        epsilon_1 = np.nanmean(duidxj[..., 0, 1] ** 2,
                               axis=tuple(range(dim)))  # Fixed from duidxj[..., 0, 0] ** 2 12/10/19 Takumi
        epsilon_2 = np.nanmean(duidxj[..., 0, 1] * duidxj[..., 1, 0], axis=tuple(range(dim)))
        epsilon = 6. * nu * (epsilon_0 + epsilon_1 + epsilon_2)  # Hinze, 1975, eq. 3-98
    return epsilon


def get_epsilon_iso(udata, x=None, y=None, lambda_f=None, lambda_g=None, nu=1.004,
                    x0=0, x1=None, y0=0, y1=None, **kwargs):
    """
    Return dissipation rate (epsilon) computed by isotropic formula involving Taylor microscale
    Isotropic formula:
    ... epsilon = 30. * nu * u2_irms / (lambda_f ** 2)
    ... epsilon = 15. * nu * u2_irms / (lambda_g ** 2)

    How it works:
    ... if Taylor microscales (lambdas) are not provided, then it attempts to compute them from udata.
    ... If either or both of lambda_f and lambda_g are provided, then it uses them.

    Parameters
    ----------
    udata: nd array, a velocity field (dim, nrows, ncols, nstacks, duration)
    x: 2d array, x coordinates- get x from get_udata_from_path(dpath, return_xy=True)
    y: 2d array, y coordinates- get y from get_udata_from_path(dpath, return_xy=True)
    lambda_f: float/1d array, longitudinal two-point correlation function
    lambda_g: float/1d array, transverse two-point correlation function
    nu: float, viscosity, default: 1.004
    x0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    kwargs: dict, optional
    ... passed to get_two_point_vel_corr_iso() if lambda_f and lambda_g are not provided

    Returns
    -------
    epsilon: float/1d array with shape (duration,)
    ... dissipation rate
    """
    udata = fix_udata_shape(udata)
    udata = udata[:, y0:y1, x0:x1, :]
    dim = len(udata)
    u2_irms = 2. / dim * get_spatial_avg_energy(udata)[0]

    # if both of lambda_g and lambda_f are provided, use lambdaf over lambdag
    if lambda_f is None and lambda_g is None:
        print('... Both of Taylor microscales, lambda_f, lambda_g, were not provided!')
        print('... Compute lambdas from scratch. One must provide x and y.')
        if x is None or y is None:
            raise ValueError('... x and y were not provided! Exiting...')
        else:
            autocorrs = get_two_point_vel_corr_iso(udata, x, y, return_rij=False, **kwargs)
            r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran = autocorrs
            lambda_f, lambda_g = get_taylor_microscales(r_long, f_long, r_tran, g_tran)
            epsilon = 15. * nu * u2_irms / (lambda_g ** 2)

    elif lambda_f is not None and lambda_g is None:
        epsilon = 30. * nu * u2_irms / (lambda_f ** 2)
    else:
        epsilon = 15. * nu * u2_irms / (lambda_g ** 2)
    return epsilon


def get_epsilon_using_diss_spectrum(udata, nu=1.0034, x0=0, x1=None,
                                    y0=0, y1=None, z0=0, z1=None,
                                    dx=1, dy=1, dz=1, nkout=None, notebook=True):
    """
    Returns dissipation rate computed by integrated the dissipation specrtrum
    ... must have a fully resolved spectrum to yield a reasonable result

    Parameters
    ----------
    udata: nd array
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise

    Returns
    -------
    epsilon: numpy array
        dissipation rate
    """
    # get dissipation spectrum
    D_k, D_k_err, k1d = get_dissipation_spectrum(udata, nu=nu, x0=x0, x1=x1,
                                                 y0=y0, y1=y1, z0=z0, z1=z1,
                                                 dx=dx, dy=dy, dz=dz, nkout=nkout, notebook=notebook)
    ## Convention of k: k=2pi/L. Many literatures use 1/L. So be careful. -Takumi

    duration = D_k.shape[-1]
    epsilon = np.empty(duration)
    for t in range(duration):
        # Divide the integrated result by (2*np.pi)^3 because the literature use the convention k=1/L NOT 2pi/L
        # Recall D ~ k^2dk. this is why there are (2pi)^3
        epsilon[t] = np.trapz(D_k[:, t], k1d[:, t])
    return epsilon

def get_epsilon_using_struc_func_old(rrs, Dxxs, epsilon_guess=100000, r0=1.0, r1=10.0, p=2, method='Nelder-Mead'):
    """
    [DEPRICATED]
    Returns the values of estimated dissipation rate using a long. structure function

    Parameters
    ----------
    rrs: numpy array
        separation length for long. transverse function
    Dxxs: numpy array
        values of long. transverse function
    epsilon_guess: numpy array
        initial guess for dissipation rate
    r0: float
        The scaled structure function is expected to have a plateau [r0, r1]
    r1: float
        The scaled structure function is expected to have a plateau [r0, r1]
    p: float/int
        order of structure function
        ... smaller order is better due to intermittency
    method: str, default: 'Nelder-Mead'
        method used to find the minimum of the test function

    Returns
    -------
    epsilons: numpy array
        estimated dissipation rate
    """
    print('... DEPRICATED- use get_dissipation_rate_using_struc_func(dll, r_dll, c=2.1, p=2, n=5)')

    def find_nearest(array, value, option='normal'):
        """
        Find an element and its index closest to 'value' in 'array'

        Parameters
        ----------
        array
        value

        Returns
        -------
        idx: index of the array where the closest value to 'value' is stored in 'array'
        array[idx]: value closest to 'value' in 'array'
        """
        # get the nearest value such that the element in the array is LESS than the specified 'value'
        if option == 'less':
            array_new = copy.copy(array)
            array_new[array_new > value] = np.nan
            idx = np.nanargmin(np.abs(array_new - value))
            return idx, array_new[idx]
        # get the nearest value such that the element in the array is GREATER than the specified 'value'
        if option == 'greater':
            array_new = copy.copy(array)
            array_new[array_new < value] = np.nan
            idx = np.nanargmin(np.abs(array_new - value))
            return idx, array_new[idx]
        else:
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

    def func(epsilon, rr, Dxx, r0, r1, p, c=2.0):
        """
        Test function to be minimized to estimate dissipation rate from the longitudinal structure function
        Parameters
        ----------
        epsilon
        rr
        Dxx
        r0
        r1
        p
        c

        Returns
        -------

        """
        ind0 = find_nearest(rr, r0)[0]
        ind1 = find_nearest(rr, r1)[0]
        residue = np.nansum((Dxx[ind0:ind1] - c * (rr[ind0:ind1] * epsilon) ** (p / 3.)) ** 2)
        return residue

    if len(rrs.shape) != 0:
        duration = rrs.shape[-1]
    else:
        duration = 1
    epsilons = np.empty(duration)
    for t in range(duration):
        result = minimize(func, epsilon_guess, (rrs[..., t], Dxxs[..., t], r0, r1, p), method=method)
        if result.success:
            epsilons[t] = result.x
        else:
            epsilons[t] = np.nan
    return epsilons


def get_epsilon_using_struc_func(dll, r_dll, c=2.1, p=2., n=5):
    """
    Estimates dissipation rate based on the plateau of the rescaled structure funciton
    ... Structure function D = c(epsilon r)^{p/3} in the inertial subrage, ignoring the intermittency correction
        Let f be f(r)=(D/c)^{3/p} / r.
        Then, f(r)= epsilon where r is in the inertial subrange. This is essentially the value at the plateau.
        ... This function grabs the n largest values of f(r), and considers its average as a dissipation rate.

    Parameters
    ----------
    dll: nd array with shape (m, duration)
        ... the p-th order structure function
        ... one of the outputs of get_structure_function()
    r_dll: nd array with shape (m, duration)
        ... separation length for the structure function (correlation function of the velocity difference)
    c: float, default:2.1 (applicable to p=2)
        ... Kolmogorov constant for the p-th order structure function
        ... c=2.1 (p=2, experiment), c=-4/5 (p=3, theory)
    p: float, order of the structure function
        ... the order of the structure function
    n: int
        ... number of structure peaks of f(r) used to estimate the dissipation rate

    Returns
    -------
    epsilons: 1d array with shape (duration, ), dissipation rate (mean of the n largest values of f(r))
    epsilon_stds: 1d array with shape (duration, ), std of the dissipation rate (std of the n largest values of f(r))
    """
    if p != 2:
        print(
            '... WARNING: Make sure to supply the appropirate Kolmogorov constant for the p-th order structure funciton: D=c(epsilon r)^(p/3)')
    if p == 3:
        c = -0.8
        print('... c=-0.8 according to Karman-Howarth eq. Ignoring user-supplied value for c...')
    dll = np.asarray(dll)
    if len(dll.shape) == 1: dll = dll.reshape((len(dll), 1))
    eps = (dll / c) ** (3. / p) / r_dll  # the plateau value of this function is equal to dissipation rate
    epsilon_cands = np.partition(eps, -n, axis=0)[-n:, :]
    epsilons = np.nanmean(epsilon_cands, axis=0)
    epsilon_stds = np.nanstd(epsilon_cands, axis=0)

    return epsilons, epsilon_stds


# def get_epsilon_from_spectrum(udata, dx=1., dy=1.,
#                               x0=0, x1=None, y0=0, y1=None,
#                               window=None,
#                               epsilon_range=(2, 7), n=51,
#                               nu=1.004, kMaxInd=None,
#                               plot=False, t0=0
#                               ):
#     """
#     Estimates dissipation rate by comparing the 1D spectrum to the master curve in Saddoughi&Veeravalli 1994
#     (WILL BE DELETED)
#     Parameters
#     ----------
#     udata
#     dx
#     dy
#     x0
#     x1
#     y0
#     y1
#     epsilon_range
#     n
#     nu
#     kMaxInd:
#     plotResults
#
#     Returns
#     -------
#
#     """
#     udata = fix_udata_shape(udata)
#     duration = udata.shape[-1]
#     epsilons = np.empty(duration)
#
#     e11_sv, keta_sv = get_rescaled_energy_spectrum_saddoughi()
#     g = interpolate.interp1d(keta_sv, e11_sv, bounds_error=False, fill_value=np.nan)
#     eiis, eiierr, k1 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, x1=x1, y0=y0, y1=y1, verbose=False, window=window)
#     m = k1.shape[0]
#     # RESAMPLE 1D DFT result
#     k1_rs, e11rs = resample2d(k1[:kMaxInd], eiis[0, :kMaxInd, :], n=m, mode='log')
#     epsilons2test = np.logspace(epsilon_range[0], epsilon_range[1], n)
#     residuals = np.zeros((n, duration))
#
#     for i, epsilon2test in enumerate(epsilons2test):
#         try:
#             e11r, k1eta = scale_energy_spectrum(e11rs, k1_rs, epsilon=epsilon2test, nu=nu)
#             measuredLog = np.log10(e11r)
#             measuredRefLog = np.swapaxes(np.tile(np.log10(g(k1eta)), (duration, 1)), 0, 1)
#             residuals[i, :] = np.nansum(np.abs(measuredLog - measuredRefLog), axis=0)
#         except:
#             residuals[i, :] = np.nan
#     epsilons = epsilons2test[np.nanargmin(residuals, axis=0)]
#
#     if plot:
#         from matplotlib.patches import Polygon
#         epsilon = epsilons[t0]
#         eta = compute_kolmogorov_lengthscale_simple(epsilon, nu)
#         e11rs_t = e11rs[:, t0]
#         #         e11_avg = np.nanmean(e11rs, axis=-1)
#         fig, ax11 = graph.plot(epsilons2test, residuals[:, t0], subplot=121)
#         epsilon_min, epsilon_max = 10 ** (np.log10(epsilon) * 0.9), 10 ** (np.log10(epsilon) * 1.1)
#         graph.tosemilogx(ax11)
#         e11r, k1eta = scale_energy_spectrum(e11rs_t, k1_rs, epsilon=epsilon, nu=nu)
#         fig, ax12 = graph.plot(k1eta, e11r, subplot=122)
#         e11rUpper, k1etaUpper = scale_energy_spectrum(e11rs_t, k1_rs,
#                                                           epsilon=epsilon_min,
#                                                           nu=nu)
#         e11rLower, k1etaLower = scale_energy_spectrum(e11rs_t, k1_rs,
#                                                           epsilon=epsilon_max,
#                                                           nu=nu)
#         polygon1 = Polygon(list(zip(k1etaUpper, e11rUpper)) + list(zip(k1etaLower[::-1], e11rLower[::-1])))
#         polygon1.set_alpha(0.5)
#         ax12.add_patch(polygon1)
#         graph.plot_saddoughi(ax=ax12)
#         graph.tologlog(ax12)
#         graph.axvband(ax11, epsilon_min, epsilon_max, color='C0')
#         graph.axvline(ax12, k1_rs[-1] * eta)
#         graph.labelaxes(ax11, '$\epsilon~(mm^2/s^3)$',
#                         'Residuals \n$\sum | \log_{10}{\\tilde{E}_{11}(\kappa \eta)} - \log_{10}{\\tilde{E}_{11}^{Ref}(\kappa \eta)}|$')
#     return epsilons

def get_epsilon_from_1d_spectrum(udata, k='kx', dx=1., dy=1., c1=0.491, r=(1., np.inf), **kwargs):
    """
    Returns a dissipation rate from a 1D energy spectrum
    How it works:
        Since
            E11=c1 epsilon^(2/3) k^(-5/3) in the inertial subrange,
        then
            epsilon = (E11/ (c1 k^(-5/3)))^(3/2)

    Parameters
    ----------
    udata: nd array with shape (dim, nrows, ncols, duration)
    k: str, 'kx' or 'ky', the direction along which the 1D spectrum is computed
    dx: float, spacing along x
    dy: float, spacing along y
    c1: kolmogorov constant for E11=c1 epsilon^(2/3) k^(-5/3)
    r: tuple, (r0, r1)
        ... (r0, r1) defines a inertial subrange
    kwargs: keyword arguments passed to get_1d_energy_spectrum()

    Returns
    -------
    epsilons: 1d array with shape (duration, )
    epsilon_stds: 1d array with shape (duration, ), std of the dissipation rate (std of the n largest values of f(r))
    """
    kmin, kmax = 2 * np.pi / r[1], 2 * np.pi / r[0]
    udata = fix_udata_shape(udata)
    eii, eii_err, k11 = get_1d_energy_spectrum(udata, k=k, dx=dx, dy=dy, **kwargs)
    e11 = eii[0]
    epsilons = np.empty(udata.shape[-1])
    epsilon_stds = np.empty(udata.shape[-1])

    indMin, indMax = find_nearest(k11, kmin)[0], find_nearest(k11, kmax)[0]
    for t in range(udata.shape[-1]):
        epsilon_ = (e11[..., t] / (c1 * k11 ** (-5 / 3.))) ** 1.5
        epsilons[t] = np.nanmean(epsilon_[indMin:indMax])
        epsilon_stds[t] = np.nanstd(epsilon_[indMin:indMax])
    return epsilons, epsilon_stds


def get_epsilon_local_using_sij(udata, dx=None, dy=None, dz=None, nu=1.004,
                                x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None, xx=None, yy=None):
    """
    Returns the local dissipation rate computed using the rate-of-strain tensor
    ... epsilon(x, y, z, t) = 2 * nu * sij(x, y, z, t) sij(x, y, z, t)
        sij: numpy array with shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
        ... sij[spacial coordinates, time, tensor indices]
            e.g.-  sij(x, y, t) can be accessed by sij[y, x, t, i, j]
        ... sij = d ui / dxj
    ... To compute epsilon, it uses a formula from Hinze (1975)
        ... epsilon_spatial = 6. * nu * (epsilon_0 + epsilon_1 + epsilon_2)  # Hinze, 1975, eq. 3-98
    Parameters
    ----------
    udata: nd array
    nu: flaot, kinematic viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    xx: 2d/3d array, x-coordinate of the data
    yy: 2d/3d array, y-coordinate of the data
    zz: 2d/3d array, z-coordinate of the data

    Returns
    -------
    epsilon_spatial: numpy array
        ... dissipation rate with shape (nrows, ncols, (nstacks), duration)
    """
    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, t0:t1]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        udata = udata[:, y0:y1, x0:x1, z0:z1, t0:t1]

    if dx is None:
        raise ValueError('... dx is None. Provide int or float.')
    if dy is None:
        raise ValueError('... dy is None. Provide int or float.')
    if dim == 3:
        if dz is None:
            raise ValueError('... dz is None. Provide int or float.')

    duidxj = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz, )
    if dim == 3:
        sij, tij = decompose_duidxj(duidxj)
        epsilon_spatial = 2. * nu * np.nansum(sij ** 2, axis=(4, 5))
    elif dim == 2:
        # Estimate epsilon from 2D data (assuming isotropy)
        epsilon_0 = duidxj[..., 0, 0] ** 2
        epsilon_1 = duidxj[..., 0, 1] ** 2
        epsilon_2 = duidxj[..., 0, 1] * duidxj[..., 1, 0]
        epsilon_spatial = 6. * nu * (epsilon_0 + epsilon_1 + epsilon_2)  # Hinze, 1975, eq. 3-98
    return epsilon_spatial


########## advanced analysis ##########
## Spatial autocorrelation functions
def compute_spatial_autocorr(ui, x, y, roll_axis=1, n_bins=None, x0=0, x1=None, y0=0, y1=None,
                             t0=None, t1=None, coarse=1.0, coarse2=0.2, notebook=True):
    """
    [DEPRICATED] Use get_two_point_vel_corr() instead.

    Compute spatial autocorrelation (two-point correlation) function of 2D velocity field using np.roll
    Spatial autocorrelation function:
        f = <u_j(\vec{x}) u_j(\vec{x}+r\hat{x_i})> / <u_j(\vec{x})^2>
    where velocity vector u = (u1, u2).
    If i = j, f is called longitudinal autocorrelation function.
    Otherwise, f is called transverse autocorrelation function.

    Example:
        u = ux # shape(height, width, duration)
        x_, y_  = range(u.shape[1]), range(u.shape[0])
        x, y = np.meshgrid(x_, y_)

        # LONGITUDINAL AUTOCORR FUNC
        rrs, corrs, corr_errs = compute_spatial_autocorr(u, x, y, roll_axis=1)  # roll_axis is i in the description above

        # TRANSVERSE AUTOCORR FUNC
        rrs, corrs, corr_errs = compute_spatial_autocorr(u, x, y, roll_axis=0)  # roll_axis is i in the description above


    Parameters
    ----------
    ui: numpy array, one-component velocity field i.e. shape: (height, width, duration)
    x: numpy array, 2d grid
    y: numpy array, 2d grid
    roll_axis: int, axis index to compute correlation... 0 for y-axis, 1 for x-axis
    n_bins: int, number of bins to compute statistics
    x0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int, index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    coarse: float (0, 1], This parameter determines the bin width (int(1. / coarse)) of the spatial autocorrelation function.
    coarse2: float (0, 1], This parameter is used to control the number of data points to compute a correlation value at a particular distance.
        ... 1: all data points are used
        ... 0.5: only a half of data points are used
    Returns
    -------
    rr: 2d numpy array, (distance, time)
    corr: 2d numpy array, (autocorrelation values, time)
    corr_err: 2d numpy array, (std of autocorrelation values, time)
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    # Array sorting
    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return arr1, arr2

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    if x0 is None:  # if None, use the whole space
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:  # if None, use the whole space
        x1 = ui.shape[1]
    elif x1 < 0:
        x1 = ui.shape[1] - x1
    if y1 is None:
        y1 = ui.shape[0]
    elif y1 < 0:
        y1 = ui.shape[0] - y1
    if t0 is None:
        t0 = 0
    if t1 is None:
        t1 = ui.shape[-1]
    elif t1 < 0:
        t1 = ui.shape[-1] - t1

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [nrows, ncolumns]
    # Number of bins- if this is too small, correlation length would be overestimated. Keep it around ncolumns
    if n_bins is None:
        n_bins = int(max(limits) * coarse)

    # Use a portion of data
    y_grid, x_grid = y[y0:y1, x0:x1], x[y0:y1, x0:x1]

    # Initialization
    rrs, corrs, corr_errs = np.empty((n_bins, t1 - t0)), np.empty((n_bins, t1 - t0)), np.empty((n_bins, t1 - t0))

    for t in tqdm(list(range(t0, t1)), desc='autocorr. time'):
        # Call velocity field at time t as uu
        uu = ui[y0:y1, x0:x1, t]
        roll_indices = list(range(0, limits[roll_axis], int(1. / coarse)))
        m = len(roll_indices)  # number of r to be considered
        n = int(x_grid.size * coarse2)

        # uu2_norm = np.nanmean(ui[y0:y1, x0:x1, ...] ** 2, axis=(0, 1))  # mean square velocity (avg over space)
        uu2_norm = np.nanmean(uu ** 2)  # mean square velocity (avg over space)

        # Fisrt, store distance and the product in an array, then compute the stats later
        rr = np.empty((n, m))
        corr = np.empty((n, m))

        if np.isnan(uu2_norm) or np.isinf(uu2_norm) or uu2_norm == 0:
            print('compute_spatial_autocorr: uu2_norm is invalid, %d / %d' % (t, t1 - t0))
            for i in range(int(coarse * limits[roll_axis])):
                x_grid_rolled, y_grid_rolled = np.roll(x_grid, i, axis=roll_axis), np.roll(y_grid, i, axis=roll_axis)
                r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)
                rr[:, i] = r_grid.flatten()[:n]
            # flatten arrays to feed to binned_statistic
            rr = rr.flatten()
            rr_means, rr_edges, binnumber = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
            rr_binwidth = (rr_edges[1] - rr_edges[0])
            rr_ = rr_edges[1:] - rr_binwidth / 2
            # rr = sorted(rr_)

            rrs[:, t - t0] = rr
            corrs[:, t - t0] = np.asarray([np.nan for i in range(n_bins)])
            corr_errs[:, t - t0] = np.asarray([np.nan for i in range(n_bins)])
        else:
            # for i in tqdm(range(int(coarse * limits[roll_axis])), desc='computing correlation'):
            for i in range(int(coarse * limits[roll_axis])):
                uu_rolled = np.roll(uu, i, axis=roll_axis)
                x_grid_rolled, y_grid_rolled = np.roll(x_grid, i, axis=roll_axis), np.roll(y_grid, i, axis=roll_axis)
                r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)
                corr_uu = uu * uu_rolled / uu2_norm  # correlation values
                # try:
                #     corr_uu = uu * uu_rolled / uu2_norm[t]  # correlation values
                #     rr[:, i] = r_grid.flatten()[:n]
                #     corr[:, i] = corr_uu.flatten()[:n]
                # except:
                #     rr[:, i] = r_grid.flatten()[:n]
                #     corr[:, i] = corr_uu.flatten()[:n]
                rr[:, i] = r_grid.flatten()[:n]
                corr[:, i] = corr_uu.flatten()[:n]
            # flatten arrays to feed to binned_statistic
            rr, corr = rr.flatten(), corr.flatten()

            # make sure rr and corr do not contain nans
            mask = get_mask_for_nan_and_inf(corr)
            mask = ~mask

            rr, corr = rr[mask], corr[mask]

            # get a histogram
            rr_means, rr_edges, binnumber = binned_statistic(rr, rr, statistic='mean', bins=n_bins, )
            corr_, _, _ = binned_statistic(rr, corr, statistic='mean', bins=n_bins)
            corr_err, _, _ = binned_statistic(rr, corr, statistic='std', bins=n_bins)
            counts, _, _ = binned_statistic(rr, corr, statistic='count', bins=n_bins)

            # One may use rr_means or the middle point of each bin for plotting
            # Default is to use the middle point
            rr_binwidth = (rr_edges[1] - rr_edges[0])
            rr_ = rr_edges[1:] - rr_binwidth / 2

            # Sort arrays
            rr, corr = sort2arr(rr_, corr_)
            rr, corr_err = sort2arr(rr_, corr_err)

            # MAKE SURE f(r=0)=g(r=0)=1
            # IF coarse2 is not 1.0, the autocorrelation functions at r=0 may take values other than 1.
            # This is due to using an inadequate normalization factor.
            # As a consequence, this messes up determining Taylor microscale etc.
            # One can fix this properly by computing the normalizing factor using the same undersampled ensemble.
            # BUT it is not worth the effort because one just needs to scale the correlation values here.
            corr /= corr[0]
            corr_err /= corr[0] * np.sqrt(counts)

            # Insert to a big array
            # rrs[0, t] = 0
            # corrs[0, t] = 1.0
            # corr_errs[0, t] = 0
            rrs[:, t - t0] = rr
            corrs[:, t - t0] = corr
            corr_errs[:, t - t0] = corr_err

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, corrs, corr_errs


def compute_spatial_autocorr3d(ui, x, y, z, roll_axis=1, n_bins=None,
                               x0=None, x1=None, y0=None, y1=None,
                               z0=None, z1=None,
                               t0=None, t1=None,
                               coarse=1.0, coarse2=0.2,
                               periodic=False, Lx=None, Ly=None, Lz=None,
                               notebook=True):
    """
    [DEPRICATED] Use get_two_point_vel_corr instead.
    
    Compute spatial autocorrelation function of 2+1 velocity field using np.roll()
    Spatial autocorrelation function:
        f = <u_j(\vec{x}) u_j(\vec{x}+r\hat{x_i})> / <u_j(\vec{x})^2>
    where velocity vector u = (u1, u2).
    If i = j, f is called longitudinal autocorrelation function.
    Otherwise, f is called transverse autocorrelation function.

    Example:
        u = ux # shape(height, width, duration)
        x_, y_  = range(u.shape[1]), range(u.shape[0])
        x, y = np.meshgrid(x_, y_)

        # LONGITUDINAL AUTOCORR FUNC
        rrs, corrs, corr_errs = compute_spatial_autocorr(u, x, y, roll_axis=1)  # roll_axis is i in the description above

        # TRANSVERSE AUTOCORR FUNC
        rrs, corrs, corr_errs = compute_spatial_autocorr(u, x, y, roll_axis=0)  # roll_axis is i in the description above


    Parameters
    ----------
    ui: numpy array, 3 + 1 scalar field. i.e. shape: (height, width, duration)
    x: numpy array, 3d grid
    y: numpy array, 3d grid
    z: numpy array, 3d grid
    roll_axis: int, axis index to compute correlation... 0 for y-axis, 1 for x-axis, 2 for z-axis
    n_bins: int, number of bins to compute statistics
    x0: int, index to specify a portion of data in which autocorrelation funciton is computed. Will use data u[y0:y1, x0:x1, z0:z1, t0:t1].
    x1: int, index to specify a portion of data in which autocorrelation funciton is computed.
    y0: int, index to specify a portion of data in which autocorrelation funciton is computed.
    y1: int, index to specify a portion of data in which autocorrelation funciton is computed.
    z0: int, index to specify a portion of data in which autocorrelation funciton is computed.
    z1: int, index to specify a portion of data in which autocorrelation funciton is computed.
    t0: int, index to specify a portion of data in which autocorrelation funciton is computed.
    t1: int, index to specify a portion of data in which autocorrelation funciton is computed.
    coarse: float (0, 1], Process coarse * possible data points. This is an option to output coarse results.
    coarse2: float (0, 1], Rolls matrix
    Returns
    -------
    rr: 2d numpy array, (distance, time)
    corr: 2d numpy array, (autocorrelation values, time)
    corr_err: 2d numpy array, (std of autocorrelation values, time)
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')

    # Array sorting
    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return arr1, arr2

    if x0 is None:  # if None, use the whole space
        x0, y0 = 0, 0
        x1, y1 = ui.shape[1], ui.shape[0]
        z0, z1 = 0, ui.shape[2]
    if t0 is None:
        t0 = 0
    if t1 is None:
        t1 = ui.shape[3]

    # Some useful numbers for processing
    nrows, ncolumns, nsteps = y1 - y0, x1 - x0, z1 - z0
    limits = [ncolumns, nrows, nsteps]
    # Number of bins- if this is too small, correlation length would be overestimated. Keep it around ncolumns
    if n_bins is None:
        n_bins = int(max(limits) * coarse) - 1
    else:
        n_bins = n_bins - 1

    if Lx is None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        Lx = xmax - xmin
    if Ly is None:
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        Ly = ymax - ymin
    if Lz is None:
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        Lz = zmax - zmin

    # Use a portion of data
    z_grid, y_grid, x_grid = z[y0:y1, x0:x1, z0:z1], y[y0:y1, x0:x1, z0:z1], x[y0:y1, x0:x1, z0:z1]

    # Initialization
    rrs, corrs, corr_errs = np.zeros((n_bins, t1 - t0)), np.ones((n_bins, t1 - t0)), np.zeros((n_bins, t1 - t0))

    roll_indices = list(range(0, limits[roll_axis], int(1. / coarse)))
    m = len(roll_indices)
    n = int(np.ceil(x_grid.size / int(1 / coarse2)))
    print('(No of r at which correlation is computed, No of samples used to compute stats) = (%d, %d)' % (m, n))
    for t in tqdm(list(range(t0, t1)), desc='autocorr. 3d time'):
        # Call velocity field at time t as uu
        uu = ui[y0:y1, x0:x1, z0:z1, t]

        uu2_norm = np.nanmean(ui[y0:y1, x0:x1, z0:z1, ...] ** 2, axis=(0, 1, 2))  # mean square velocity

        # Initialization
        # rr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))
        # corr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))

        rr = np.empty((n, m))
        corr = np.empty((n, m))

        for j, i in enumerate(tqdm(roll_indices, desc='computing correlation')):
            uu_rolled = np.roll(uu, i, axis=roll_axis)
            x_grid_rolled, y_grid_rolled, z_grid_rolled = np.roll(x_grid, i, axis=roll_axis), \
                                                          np.roll(y_grid, i, axis=roll_axis), \
                                                          np.roll(z_grid, i, axis=roll_axis)
            rx = x_grid_rolled - x_grid
            ry = y_grid_rolled - y_grid
            rz = z_grid_rolled - z_grid
            if periodic:
                rx[rx > Lx / 2] = Lx - rx[rx > Lx / 2]
                rx[rx < -Lx / 2] = Lx + rx[rx < -Lx / 2]
                ry[ry > Ly / 2] = Ly - ry[ry > Ly / 2]
                ry[ry < -Ly / 2] = Ly + ry[ry < -Ly / 2]
                rz[rz > Lz / 2] = Lz - rz[rz > Lz / 2]
                rz[rz < -Lz / 2] = Lz + rz[rz < -Lz / 2]
            r_grid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)  # use unsigned distance to take statistics
            corr_uu = uu * uu_rolled / uu2_norm[t]  # two-pt correlation
            # indices = list(range(n))
            rr[:, j] = r_grid.flatten()[::int(1 / coarse2)]
            corr[:, j] = corr_uu.flatten()[::int(1 / coarse2)]

        # flatten arrays to feed to binned_statistic
        rr, corr = rr.flatten(), corr.flatten()

        # get a histogram
        # rr_, _, _ = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
        rr_means, rr_edges, binnumber = binned_statistic(rr, rr, statistic='mean', bins=n_bins - 1)
        rr_binwidth = (rr_edges[1] - rr_edges[0])
        rr_ = rr_edges[1:] - rr_binwidth / 2
        corr_, _, _ = binned_statistic(rr, corr, statistic='mean', bins=n_bins - 1)
        corr_err, _, _ = binned_statistic(rr, corr, statistic='std', bins=n_bins - 1)

        # # Sort arrays
        # rr, corr = sort2arr(rr_, corr_)
        # rr, corr_err = sort2arr(rr_, corr_err)
        #
        # # Insert to a big array
        # rrs[:, t] = rr
        # corrs[:, t] = corr
        # corr_errs[:, t] = corr_err

        rrs[0, t], corrs[0, t], corr_errs[0, t] = 0, 1., 0
        _, corrs[1:, t] = sort2arr(rr_, corr_)
        rrs[1:, t], corr_errs[1:, t] = sort2arr(rr_, corr_err)

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, corrs, corr_errs


def get_two_point_vel_corr_roll(ui, x, y, z=None, roll_axis=1, n_bins=None,
                                x0=None, x1=None, y0=None, y1=None,
                                z0=None, z1=None,
                                t0=None, t1=None,
                                coarse=1.0, coarse2=0.2,
                                periodic=False, Lx=None, Ly=None, Lz=None,
                                notebook=True):
    """
    Returns a two-point velocity correlation function <ui(x) ui(x+r)> / < |ui(x)|^2 >\
    ... Redirects to compute_spatial_autocorr2d OR compute_spatial_autocorr3d!
        ... Spatial autocorrelation function and two-pt vel corr function are the same things.
    ... This function is primarily used to compute the longitudnal/transverse two-pt vel correlation functions
        ... f(r) =  <u1(x) u1(x+r \hat{x_1})> / < |u1(x)|^2 >\
            ... ui = udata[0, ...], roll_axis=1
        ... g(r) =  <u1(x) u1(x+r \hat{x_2})> / < |u1(x)|^2 >\
            ... ui = udata[1, ...], roll_axis=1
    ... One can deduce Taylor microscale, energy spectrum from f(r) and g(r)
    ... One can compute Rii with the outputs of this function.
        (1) R11 = f(r) * < ux^2 >
        (2) R22 = g(r) * < uy^2 >
            ... This g(r) can be obtained by (ui = udata[1, ...], roll_axis=1)
        (3) R33 = g(r) * < uy^2 >
            ... This g(r) can be obtained by (ui = udata[2, ...], roll_axis=1)
        ... Important quantities to check are one-dimensional spectra Eii.
            ... Eii = 2 x Fourier Transform of(Rii)
            ... This may not yield the cleanest spectra; however, one should always check
                the following equality
                ... < ux^2 > = \int_0^\infty E11(\kappa1) d\kappa1
                ... < uy^2 > = \int_0^\infty E22(\kappa1) d\kappa1
                ... < uz^2 > = \int_0^\infty E33(\kappa1) d\kappa1
                i.e. (Turbulent) Energy = 0.5  \int_0^\infty Eii(\kappa1) d\kappa1

    Parameters
    ----------
    ui: nd array, ux, uy or uz
    ... e.g. udata[0, ...] for ux
    x: 2d/3d grid, x coordinate of a field
    y: 2d/3d grid, y coordinate of a field
    z: 2d/3d grid, y coordinate of a field
    roll_axis: int, direction of a displacement vector between two points
        ... 0: y, 1: x, 2:z
    n_bins: int, nuber of bins used to compute the two-point statistics
    x0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    x1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    y0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    y1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    z0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    z1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    t0: int, index used to specify a temporal region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    t1: int, index used to specify a temporal region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    coarse: float, (0, 1], default: 1
        the first parameter to save computation time related sampling frequency
        ... The higher "coarse" is, it samples more possible data points.
        ... If "coarse" == 1, it samples all possible data points.
        ... If "coarse" == 0.5, it samples only a half of possible data points. Could overestimate Taylor microscale.
    coarse2: float, (0, 1], default: 0.2
        the second parameter to save computation time related to making a histogram
        ... Determines how many sampled data points to be used to make a histogram
        ... If "coarse" == 1, it uses all data points to make a histogram.
        ... If "coarse" == 0.5, it uses only a half of data points to make a histogram
    periodic: bool, default False
        ... turn this on to correctly account for the periodicty of the field
    Lx: float, size of the periodic box in the x-direction (not index)
    Ly: float, size of the periodic box in the y-direction (not index)
    Lz: float, size of the periodic box in the z-direction (not index)
    notebook: bool, if True, it uses tqdm_notebook instead of tqdm

    Returns
    -------
    rr: 2d numpy array, distance (distance, time)
    corr: 2d numpy array, (two-point correlation values, time)
    corr_err: 2d numpy array, (std of two-point correlation, time)
    """
    if z1 is None:
        rrs, corrs, corr_errs = compute_spatial_autocorr(ui, x, y, roll_axis=roll_axis, n_bins=n_bins,
                                                         x0=x0, x1=x1, y0=y0, y1=y1,
                                                         t0=t0, t1=t1,
                                                         coarse=coarse, coarse2=coarse2,
                                                         notebook=notebook)
    else:
        rrs, corrs, corr_errs = compute_spatial_autocorr3d(ui, x, y, z=z, roll_axis=roll_axis, n_bins=n_bins,
                                                           x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                                           t0=t0, t1=t1,
                                                           coarse=coarse, coarse2=coarse2,
                                                           periodic=periodic, Lx=Lx, Ly=Ly, Lz=Lz,
                                                           notebook=notebook)

    return rrs, corrs, corr_errs


def get_two_point_vel_corr(udata, x, y, z=None,
                           x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                           nd=10 ** 3, nr=70, rmax=None, notebook=True,
                           periodic=False):
    """
    
    Returns the normalized two-point velocity tatistics (rrs, fs, f_errs, rrs, gs, g_errs)

    ... f(r): longitudinal two-pt  stat
    ... g(r): transverse two-pt  stat
    ... In order for f(r) and g(r) to have a meaning, the flow MUST be isotropic
    ... Algorithm:
        1. Choose two points A and B. Let r be the displacement vector (x_A - x_B)
        2. Create a right-handed orthogonal basis (\hat{r}, n) or (\hat{r}, n1, n2)
        3. f(r) = < (u(A) \cdot r) u(B) \cdot r)> /   < |u(A) \cdot r|^2>
           g(r) = < (u(A) \cdot n) u(B) \cdot n)> /   < |u(A) \cdot r|^2>
    ... The v-field must be isotropic for this result to make sense.


    Parameters
    ----------
    udata: nd array, a velocity field
    x: 2/3d array, x-coordinates of the velocity field
    y: 2/3d array, y-coordinates of the velocity field
    z: 2/3d array, z-coordinates of the velocity field
    x0: int, index used to specify a spatial region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    x1: int, index used to specify a spatial region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    y0: int, index used to specify a spatial region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    y1: int, index used to specify a spatial region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    z0: int, index used to specify a spatial region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    z1: int, index used to specify a spatial region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    t0: int, index used to specify a temporal region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    t1: int, index used to specify a temporal region in which the statistics is computed udata[:, x0:x1, y0:y1, z0:z1, t0:t1]
    nd: int, number of bins in the histogram- the bin width is roughly rmax/nd
    nr: int, number of pairs of points used to compute the two-point correlation function at a particular distance r
    rmax: float, maximum distance to compute the two-point correlation function
    notebook: bool, if True, tqdm_notebook is used instead of tqdm
    periodic: bool, if True, the velocity field is assumed to be periodic

    Returns
    -------
    autocorrs: tuple
    ... autocorrs = (rrs, fs, f_errs, rrs, gs, g_errs)
    ... rrs: 2d array, radial distances with shape (nd, duration)
    ... fs: 2d array, longitudinal velocity correlation funciton (nd, duration)
    ... f_errs: 2d array, longitudinal velocity correlation funciton error (nd, duration)
    ... rrs: 2d array, radial distances with shape (nd, duration)
    ...... same as the first return value (this is returned for consistency)
    ... gs: 2d array, transverse velocity correlation funcitonwith shape (nd, duration)
    ... g_errs: 2d array, transverse velocity correlation funciton error with shape (nd, duration)
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    # def get_rotation_matrix_between_two_vectors(a, b):
    #     """
    #     Returns a 3D rotation matrix R that rotates a unit vector onto a unit vector of b
    #     """
    #     a, b = vec.norm(a), vec.norm(b)
    #     v = vec.cross(a, b)
    #     s = vec.mag1(v)
    #     c = vec.dot(a, b)
    #
    #     A = np.asarray([[0, -v[2], v[1]],
    #                     [v[2], 0, -v[0]],
    #                     [-v[1], v[0], 0]])
    #     I = np.asarray([[1, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 1]])
    #     R = I + A + np.matmul(A, A) * (1 - c) / s ** 2
    #     return R
    def get_rotation_matrix_between_two_vectors(a, b):
        """
        Returns a 3D rotation matrix R that rotates a unit vector of "a" onto a unit vector of "b"
        """
        a, b = vec.norm(a), vec.norm(b)
        if all(a == b):
            return np.identity(3)
        elif all(a == -b):
            # When a and b are complete opposite to each other, there is no unique rotation matrix in 3D!
            ## Also, note that -np.identity(3) is not unitary.
            R = np.identity(3)
            if len(np.argwhere(a)) == 1:
                cond1 = np.where(v != 0)[0][0]
                cond2 = np.where(v == 0)[0][0]
                R[:, cond1] *= -1
                R[:, cond2] *= -1
            else:
                cond1 = np.argmax(np.abs(v))
                cond2 = [ind for ind in [0, 1, 2] if ind != cond1][0]
                R[:, cond1] *= -1
                R[:, cond2] *= -1
            return R
        else:
            v = vec.cross(a, b)
            s = vec.mag1(v)
            c = vec.dot(a, b)
            A = np.asarray([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            I = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])
            R = I + A + np.matmul(A, A) * (1 - c) / s ** 2
            return R

    udata = fix_udata_shape(udata)
    dim = len(udata)

    if x1 is None:
        x1 = udata[0, ...].shape[1]
    if y1 is None:
        y1 = udata[0, ...].shape[0]
    if x1 < 0:
        x1 = udata[0, ...].shape[1] - x1
    if y1 < 0:
        y1 = udata[0, ...].shape[0] - y1

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
    if dim == 3:
        if z1 is None:
            z1 = udata[0, ...].shape[2]
        nsteps = z1 - z0
        limits = [ncolumns, nrows, nsteps]
    if t1 is None:
        t1 = udata.shape[-1]

    # Number of bins- if this is too small, correlation length would be overestimated.
    # Keep it around ncolumns
    if nr is None:
        nr = max(limits)

    # Use a portion of data
    if dim == 3:
        z_grid, y_grid, x_grid = z[y0:y1, x0:x1, z0:z1], y[y0:y1, x0:x1, z0:z1], x[y0:y1, x0:x1, z0:z1]
        dx, dy, dz = np.abs(x_grid[0, 1, 0] - x_grid[0, 0, 0]), \
                     np.abs(y_grid[1, 0] - y_grid[0, 0, 0]), \
                     np.abs(y_grid[0, 0, 1] - z_grid[0, 0, 0])

    elif dim == 2:
        y_grid, x_grid = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
        dx, dy = np.abs(x_grid[0, 1] - x_grid[0, 0]), np.abs(y_grid[1, 0] - y_grid[0, 0])
        nt = 1

    # Initialization
    rrs, fs, f_errs = np.zeros((nr, t1 - t0)), np.ones((nr, t1 - t0)), np.zeros((nr, t1 - t0))
    rrs, gs, g_errs = np.zeros((nr, t1 - t0)), np.ones((nr, t1 - t0)), np.zeros((nr, t1 - t0))
    is_R1_reasonable = False

    if periodic:
        prefactor = 0.5
    else:
        prefactor = 1.

    if dim == 2:
        xmin, xmax, ymin, ymax = np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)
        width, height = xmax - xmin, ymax - ymin
        if rmax is None: rmax = min([width, height]) * prefactor
        rs_ = np.linspace(dx * 2, rmax, nr)
    elif dim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(x_grid), np.max(x_grid), \
                                             np.min(y_grid), np.max(y_grid), \
                                             np.min(z_grid), np.max(z_grid)
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin
        if rmax is None: rmax = min([width, height, depth]) * prefactor
        rs_ = np.linspace(dx * 2, rmax, nr)
    rs = np.empty(nd)
    fs_ = np.empty(nd)
    gs_ = np.empty(nd)
    denominators_f = np.empty(nd)
    denominators_g = np.empty(nd)

    uirms = get_characteristic_velocity(udata)
    for t in tqdm(list(range(t0, t1)), desc='two-pt vel corr: time'):
        for i, r in enumerate(tqdm(rs_, desc='two-pt vel corr: r-loop')):
            for j in range(nd):
                if dim == 2:
                    while not is_R1_reasonable:
                        # Randomly pick a point in space, call it R0
                        X0, Y0 = np.random.random() * width + xmin, np.random.random() * height + ymin
                        X0_ind, _ = find_nearest(x_grid[0, :], X0)
                        Y0_ind, _ = find_nearest(y_grid[:, 0], Y0)
                        R0 = np.asarray([x_grid[0, X0_ind], y_grid[Y0_ind, 0]])
                        # Randomly pick another point in space, call it R1
                        theta = 2 * np.pi * np.random.random()
                        X1, Y1 = X0 + r * np.cos(theta), Y0 + r * np.sin(theta)
                        if periodic:
                            if X1 > xmax:
                                X1 = - width
                            elif X1 < xmin:
                                X1 += width
                            if Y1 > ymax:
                                Y1 = - height
                            elif Y1 < ymin:
                                Y1 += height
                            break
                        else:
                            is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin
                    X1_ind, _ = find_nearest(x_grid[0, :], X1)
                    Y1_ind, _ = find_nearest(y_grid[:, 0], Y1)
                    R1 = np.asarray([x_grid[0, X1_ind], y_grid[Y1_ind, 0]])
                elif dim == 3:
                    while not is_R1_reasonable:
                        # Randomly pick a point in space, call it R0
                        X0, Y0, Z0 = np.random.random() * width + xmin, np.random.random() * height + ymin, np.random.random() * depth + zmin
                        X0_ind, _ = find_nearest(x_grid[0, :, 0], X0)
                        Y0_ind, _ = find_nearest(y_grid[:, 0, 0], Y0)
                        Z0_ind, _ = find_nearest(z_grid[0, 0, :], Z0)
                        R0 = np.asarray([x_grid[0, X0_ind, 0], y_grid[Y0_ind, 0, 0], z_grid[0, 0, Z0_ind]])
                        # Randomly pick another point in space, call it R1
                        theta = 2 * np.pi * np.random.random()
                        phi = 2 * np.pi * np.random.random()
                        X1, Y1, Z1 = X0 + r * np.sin(theta) * np.cos(phi), \
                                     Y0 + r * np.sin(theta) * np.sin(phi), \
                                     Z0 + r * np.cos(theta)
                        is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin and Z1 < zmax and Z1 > zmin
                    X1_ind, _ = find_nearest(x_grid[0, :, 0], X1)
                    Y1_ind, _ = find_nearest(y_grid[:, 0, 0], Y1)
                    Z1_ind, _ = find_nearest(z_grid[0, 0, :], Z1)
                    R1 = np.asarray([x_grid[0, X1_ind, 0], y_grid[Y1_ind, 0, 0], z_grid[0, 0, Z1_ind]])

                R01 = R1 - R0
                if periodic:
                    if R01[0] > width / 2:
                        R01[0] -= width
                    elif R01[0] < -width / 2:
                        R01[0] += width
                    if R01[1] > height / 2:
                        R01[1] -= height
                    elif R01[1] < -height / 2:
                        R01[1] += height
                    if dim == 3:
                        if R01[2] > depth / 2:
                            R01[2] -= depth
                        elif R01[2] < -depth / 2:
                            R01[2] += depth

                basis = vec.get_an_orthonormal_basis(dim, v1=R01)
                # CRUCIAL: make sure to use the same convention to define the direction of the transverse vector
                basis = vec.apply_right_handedness(basis)

                rs[j] = vec.mag1(R01)
                # denominator = vec.dot(udata[:, Y0_ind, X0_ind, t], basis[:, 0]) ** 2 # perhaps i should use u'
                # denominators = uirms[t]**2

                if dim == 2:
                    # denominator = vec.dot(udata[:, Y0_ind, X0_ind, t], basis[:, 1])  ** 2
                    fs_[j] = vec.dot(udata[:, Y0_ind, X0_ind, t], basis[:, 0]) * vec.dot(udata[:, Y1_ind, X1_ind, t],
                                                                                         basis[:, 0])
                    gs_[j] = vec.dot(udata[:, Y0_ind, X0_ind, t], basis[:, 1]) * vec.dot(udata[:, Y1_ind, X1_ind, t],
                                                                                         basis[:, 1])
                    denominators_f[j] = vec.dot(udata[:, Y0_ind, X0_ind, t], basis[:, 0]) ** 2
                    denominators_g[j] = vec.dot(udata[:, Y0_ind, X0_ind, t], basis[:, 1]) ** 2
                elif dim == 3:
                    if t == t0:
                        R01_0 = copy.deepcopy(R01)
                        basis_0 = copy.deepcopy(basis)
                        # Rts = vec.get_perp_vectors_3d(R01, n=nt)
                    else:
                        R = get_rotation_matrix_between_two_vectors(R01_0, R01)
                        basis[:, 1] = np.matmul(R, basis_0[:, 1])
                        basis[:, 2] = np.matmul(R, basis_0[:, 2])
                    fs_[j] = vec.dot(udata[:, Y0_ind, X0_ind, Z0_ind, t], basis[:, 0]) * vec.dot(
                        udata[:, Y1_ind, X1_ind, Z1_ind, t], basis[:, 0])
                    gs_[j] = vec.dot(udata[:, Y0_ind, X0_ind, Z0_ind, t], basis[:, 1]) * vec.dot(
                        udata[:, Y1_ind, X1_ind, Z0_ind, t], basis[:, 1])
                    # for k, Rt in enumerate(Rts):
                    #     gs_[j * nt + k] = vec.dot(udata[:, Y0_ind, X0_ind, Z0_ind, t], Rt) * vec.dot(udata[:, Y1_ind, X1_ind, Z0_ind, t], Rt) / denominator

                    denominators_f[j] = vec.dot(udata[:, Y0_ind, X0_ind, Z0_ind, t], basis[:, 0]) ** 2
                    denominators_g[j] = vec.dot(udata[:, Y0_ind, X0_ind, Z0_ind, t], basis[:, 1]) ** 2
                is_R1_reasonable = False

            rrs[i, t] = np.nanmean(rs)
            fs[i, t] = np.nanmean(fs_) / np.nanmean(denominators_f)
            f_errs[i, t] = fs[i, t] * np.sqrt(
                (np.nanstd(fs_) / np.nanmean(fs_)) ** 2 + (np.nanstd(denominators_f) / np.nanmean(denominators_f)) ** 2)
            gs[i, t] = np.nanmean(gs_) / np.nanmean(denominators_f)
            g_errs[i, t] = gs[i, t] * np.sqrt(
                (np.nanstd(gs_) / np.nanmean(gs_)) ** 2 + (np.nanstd(denominators_g) / np.nanmean(denominators_g)) ** 2)
    autocorrs = (rrs, fs, f_errs, rrs, gs, g_errs)

    if notebook:
        from tqdm import tqdm as tqdm
    return autocorrs


def get_two_point_vel_corr_iso(udata, x, y, z=None, time=None, n_bins=None,
                               x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=None, t1=None,
                               coarse=1.0, coarse2=0.2, notebook=True, return_rij=False):
    """
    Returns two-point velocity autocorrelation tensor, and autocorrelation functions.
    Uses the x-component of velocity. (CAUTION required for unisotropic flows)
    ... Pope Eq. 6.44

    Parameters
    ----------
    udata: 5D or 4D numpy array, 5D if the no. of spatial dimensions is 3. 4D if the no. of spatial dimensions is 2.
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)
    x: 2d/3d grid, x coordinate of a field
    y: 2d/3d grid, y coordinate of a field
    z: 2d/3d grid, y coordinate of a field
    roll_axis: int, direction of a displacement vector between two points
        ... 0: y, 1: x, 2:z
    n_bins: int, nuber of bins used to compute the two-point statistics
    x0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    x1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    y0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    y1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    z0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    z1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    t0: int, index used to specify a temporal region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    t1: int, index used to specify a temporal region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    coarse: float, (0, 1], default: 1
        the first parameter to save computation time related sampling frequency
        ... The higher "coarse" is, it samples more possible data points.
        ... If "coarse" == 1, it samples all possible data points.
        ... If "coarse" == 0.5, it samples only a half of possible data points. Could overestimate Taylor microscale.
    coarse2: float, (0, 1], default: 0.2
        the second parameter to save computation time related to making a histogram
        ... Determines how many sampled data points to be used to make a histogram
        ... If "coarse" == 1, it uses all data points to make a histogram.
        ... If "coarse" == 0.5, it uses only a half of data points to make a histogram
    notebook: bool, if True, it uses tqdm_notebook instead of tqdm to display a progress bar
    return_rij: bool,
        ... If True, it returns a funciton (the autocorrealtion tensor Rij with aarguments (i, j, r, t, udata))
           AND autocorrs = (r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran)

    Returns
    -------
    autocorrs: tuple, (r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran)
        ... r_long: 2D numpy array, distance for f_long, shape: (nbins, duration)
        ... f_long: longitudinal autocorrelation values, shape: (nbins, duration)
        ... f_err_long: longitudinal autocorrelation errors, shape: (nbins, duration)
        ... r_tran: 2D numpy array, distance for g_tran, shape: (nbins, duration)
        ... g_tran: transverse autocorrelation values, shape: (nbins, duration)
        ... g_err_tran: transverse autocorrelation errors, shape: (nbins, duration)
    rij: funciton with arguments (r, t, i, j), autocorrelation tensor ,optional
        ... Rij (\vec{r} , t) = <u_j(\vec{x}) u_j(\vec{x}+\vec{r})> / <u_j(\vec{x})^2>
        ... If system is homogeneous and isotropic,
                        Rij (\vec{r} , t) = u_rms^2 [g(r,t) delta_ij + {f(r,t) - g(r,t)} r_i * r_j / r^2]
            where f, and g are long. and transverse autocorrelation functions.
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    dim = len(udata)
    if dim == 2:
        height, width, duration = udata[0].shape
        ux, uy = udata[0], udata[1]
    elif dim == 3:
        height, width, depth, duration = udata[0].shape
        ux, uy, uz = udata[0], udata[1], udata[2]

    print('Compute two-point velocity autocorrelation')
    if dim == 2:
        r_long, f_long, f_err_long = compute_spatial_autocorr(ux, x, y, roll_axis=1, n_bins=n_bins, x0=x0, x1=x1,
                                                              y0=y0, y1=y1, t0=t0, t1=t1,
                                                              coarse=coarse, coarse2=coarse2, notebook=notebook)
        r_tran, g_tran, g_err_tran = compute_spatial_autocorr(ux, x, y, roll_axis=0, n_bins=n_bins, x0=x0, x1=x1,
                                                              y0=y0, y1=y1, t0=t0, t1=t1,
                                                              coarse=coarse, coarse2=coarse2, notebook=notebook)
    elif dim == 3:
        r_long, f_long, f_err_long = compute_spatial_autocorr3d(ux, x, y, z, roll_axis=1, n_bins=n_bins, x0=x0, x1=x1,
                                                                y0=y0, y1=y1, z0=z0, z1=z1,
                                                                coarse=coarse, coarse2=coarse2, notebook=notebook)
        r_tran, g_tran, g_err_tran = compute_spatial_autocorr3d(ux, x, y, z, roll_axis=0, n_bins=n_bins, x0=x0, x1=x1,
                                                                y0=y0, y1=y1, z0=z0, z1=z1,
                                                                coarse=coarse, coarse2=coarse2, notebook=notebook)
    # Return autocorrelation values and rs
    autocorrs = (r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran)

    if notebook:
        from tqdm import tqdm as tqdm

    if return_rij:
        print('Compute two-point velocity autocorrelation tensor Rij.')
        # Make long./trans. autocorrelation functions
        ## make an additional list to use 2d interpolation: i.e. supply autocorrelation values as a function of (r, t)
        time_list = []
        for t in time:
            time_list += [t] * len(r_long[:, 0])
        # 2d interpolation of long./trans. autocorrelation functions
        print('... 2D interpolation to define long./trans. autocorrelation function (this may take a while)')
        f = interp2d(r_long.flatten(), time_list, f_long.flatten())
        g = interp2d(r_tran.flatten(), time_list, g_tran.flatten())

        # Define Rij(r, t) as a function.
        def two_pt_velocity_autocorrelation_tensor(i, j, r, t, udata):
            dim = len(r)
            u2_avg = np.nanmean(udata ** 2, axis=tuple(range(dim + 1)))  # ui_rms(t)
            if dim == 2:
                x, y = r[0], r[1]
            elif dim == 3:
                x, y, z = r[0], r[1], r[2]
            r2_norm = np.zeros_like(x)
            for k in range(dim):
                r2_norm += r[k] ** 2
            r_norm = np.sqrt(r2_norm)
            Rij_value = u2_avg[t] * (
                    g(r_norm, t) * kronecker_delta(i, j) + (f(r_norm, t) - g(r_norm, t)) * r[i] * r[j] / (
                    r_norm ** 2))
            return Rij_value

        print(
            '... Returning two-point velocity autocorrelation tensor Rij(r, t). Arguments: i, j, r, t. Pope Eq. 6.44.')
        return two_pt_velocity_autocorrelation_tensor, autocorrs
    else:
        return autocorrs


def get_autocorr_functions(r_long, f_long, r_tran, g_tran, time):
    """
    Return interpolated functions using the outputs of get_two_point_vel_corr_iso()
    ... arguments: r, t

    Parameters
    ----------
    r_long: numpy array
        output of get_two_point_vel_corr_iso()
    f_long: numpy array
        output of get_two_point_vel_corr_iso()
    r_tran: numpy array
        output of get_two_point_vel_corr_iso()
    g_tran: numpy array
        output of get_two_point_vel_corr_iso()
    time: numpy array
        time corresponding to the given autocorrelation functions


    Returns
    -------
    f, g: long./trans. autocorrelation functions with argument (distance, time)
    """
    time_list = []
    for t in time:
        time_list += [t] * len(r_long[:, 0])
    # 2d interpolation of long./trans. autocorrelation functions
    f = interp2d(r_long.flatten(), time_list, f_long.flatten())
    g = interp2d(r_tran.flatten(), time_list, g_tran.flatten())
    return f, g


def get_autocorr_functions_int_list(r_long, f_long, r_tran, g_tran):
    """
    Returns lists of INTERPOLATED autocorrelation functions
    ... an element of the list is a spline interpolating function at an instant in time

    Parameters
    ----------
    r_long: numpy array with shape (n_bins, duration)
        ... distance for the long. autocorrelation function
        output of get_two_point_vel_corr_iso()
    f_long: numpy array
        ... long. autocorrelation function
        output of get_two_point_vel_corr_iso()
    r_tran: numpy array
        ... distance for the trans. autocorrelation function
        output of get_two_point_vel_corr_iso()
    g_tran: numpy array
        ... trans. autocorrelation function
        output of get_two_point_vel_corr_iso()

    Returns
    -------
    fs: list, length = duration = f_long.shape[-1]
        list of interpolated longitudinal structure functions
    gs: list, length = duration = g_long.shape[-1]
        list of interpolated transverse structure functions
    """
    n, duration = r_long.shape
    data = [r_long, f_long, r_tran, g_tran]
    # Remove nans if necessary
    for i, datum in enumerate(data):
        if ~np.isnan(data[i]).any():
            data[i] = data[i][~np.isnan(data[i])]
    # interpolate data (3rd order spline)
    fs, gs = [], []
    for t in range(duration):
        # if r_long contains nans, UnivariateSpline fails. so clean this up.
        r_long_tmp, f_long_tmp = remove_nans_for_array_pair(r_long[:, t], f_long[:, t])
        r_tran_tmp, g_tran_tmp = remove_nans_for_array_pair(r_tran[:, t], g_tran[:, t])

        # Make sure that f(r=0, t)=g(r=0,t)=1
        f_long_tmp /= f_long_tmp[0]
        g_tran_tmp /= g_tran_tmp[0]

        # Make autocorrelation functions even
        r_long_tmp = np.concatenate((-np.flip(r_long_tmp, axis=0)[:-1], r_long_tmp))
        f_long_tmp = np.concatenate((np.flip(f_long_tmp, axis=0)[:-1], f_long_tmp))
        r_tran_tmp = np.concatenate((-np.flip(r_tran_tmp, axis=0)[:-1], r_tran_tmp))
        g_tran_tmp = np.concatenate((np.flip(g_tran_tmp, axis=0)[:-1], g_tran_tmp))

        # Interpolate
        f_spl = UnivariateSpline(r_long_tmp, f_long_tmp, s=0, k=3)  # longitudinal autocorrelation func.
        g_spl = UnivariateSpline(r_tran_tmp, g_tran_tmp, s=0, k=3)  # transverse autocorrelation func.

        fs.append(f_spl)
        gs.append(g_spl)
    return fs, gs


def get_autocorrelation_tensor_iso(r_long, f_long, r_tran, g_tran, time):
    """
    Returns an autocorrelation tensor, assuming isotropy of the flow
    ... not recommended for practical use due to long convergence time

    Parameters
    ----------
    r_long: numpy array
        output of get_two_point_vel_corr_iso()
    f_long: numpy array
        output of get_two_point_vel_corr_iso()
    r_tran: numpy array
        output of get_two_point_vel_corr_iso()
    g_tran: numpy array
        output of get_two_point_vel_corr_iso()
    time: numpy array
        time corresponding to the given autocorrelation functions

    Returns
    -------
    rij: autocorrelation tensor
        technically a function with arguments: i, j, r, t, udata (tensor indices, separation distance, time index, udata)
    """
    f, g = get_autocorr_functions(r_long, f_long, r_tran, g_tran, time)

    # Define Rij(r, t) as a function.
    def two_pt_velocity_autocorrelation_tensor(i, j, r, t, udata):
        dim = len(r)
        u2_avg = np.nanmean(udata ** 2, axis=tuple(range(dim + 1)))  # spatial average
        if dim == 2:
            x, y = r[0], r[1]
        elif dim == 3:
            x, y, z = r[0], r[1], r[2]
        r2_norm = np.zeros_like(x)
        for k in range(dim):
            r2_norm += r[k] ** 2
        r_norm = np.sqrt(r2_norm)
        rij_value = u2_avg[t] * (
                g(r_norm, t) * kronecker_delta(i, j) + (f(r_norm, t) - g(r_norm, t)) * r[i] * r[j] / (
                r_norm ** 2))
        return rij_value

    rij = two_pt_velocity_autocorrelation_tensor
    return rij


# STRUCTURE FUNCTION

## OLD CODE USING np.roll
### ... Turns out it is only valid for the second-order struc. func.

def get_structure_function_long(udata, x, y, z=None, p=2, roll_axis=1, n_bins=None, nu=1.004, u='ux',
                                x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                                coarse=1.0, coarse2=0.2, notebook=True):
    """
    DEPRECATED! Use get_structure_function()

    Structure tensor Dij is essentially the covariance of the two-point velocity difference
    There is one-to-one correspondence between Dij and Rij. (Pope 6.36)
    This method returns the LONGITUDINAL STRUCTURE FUNCTION.
    If p=2, this returns D_LL.
    ... Returns rrs, Dxxs, Dxx_errs, rrs_scaled, Dxxs_scaled, Dxx_errs_scaled
    Parameters
    ----------
    udata
    x: numpy array
        x-coordinate of the spatial grid corresponding to the given udata
        ... it does not have to ve equally spaced
    y: numpy array
        y-coordinate of the spatial grid corresponding to the given udata
        ... it does not have to ve equally spaced
    z: numpy array
        z-coordinate of the spatial grid corresponding to the given udata
        ... it does not have to ve equally spaced
    p: float or int
        order of structure function
    roll_axis: int
        "u" and "roll_axis" determines whether this method returns the longitudinal or transverse structure function
        If you want longitudinal, match "u" and "roll_axis". e.g.- u='ux' and roll_axis=1, u='uy' and roll_axis=0
        If you want transverse, do not match "u" and "roll_axis". e.g.- u='ux' and roll_axis=0, u='uy' and roll_axis=1
    n_bins: int, default=None
        number of bins used to take a histogram of velocity difference
    nu: float
        viscosity
    u: str, default='ux'
        velocity component used to compute the structure functions. Choices are 'ux', 'uy', 'uz'
    x0: int, default: 0
        Specified the region of udata to compute the structure function.
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    x1: int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    y0: int, default: 0
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    y1 int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    z0: int, default: 0
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    z1 int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    t0: int, default: 0
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    t1 int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    coarse: float, (0, 1], default: 1
        the first parameter to save computation time related sampling frequency
        ... The higher "coarse" is, it samples more possible data points.
        ... If "coarse" == 1, it samples all possible data points.
        ... If "coarse" == 0.5, it samples only a half of possible data points. Could overestimate Taylor microscale.
    coarse2: float, (0, 1], default: 0.2
        the second parameter to save computation time related to making a histogram
        ... Determines how many sampled data points to be used to make a histogram
        ... If "coarse" == 1, it uses all data points to make a histogram.
        ... If "coarse" == 0.5, it uses only a half of data points to make a histogram
    notebook: bool
        ... if True, it uses tqdm.tqdm_notebook instead of tqdm.tqdm

    Returns
    -------
    rrs: numpy array
        two-point separation distance for the structure function
    Dxxs: numpy array
        values of the structure function
    Dxx_errs: numpy array
        error of the structure function
    rrs_scaled: numpy array
        two-point separation distance for the SCALED structure function
    Dxxs_scaled: numpy array
        values of the SCALED structure function
    Dxx_errs_scaled: numpy array
        error of the SCALED structure function
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')

    # Array sorting
    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and Sorted arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return arr1, arr2

    dim = len(udata)
    if u == 'ux':
        ui = udata[0, ...]
    elif u == 'uy':
        ui = udata[1, ...]
    elif u == 'uz':
        ui = udata[2, ...]

    if x1 is None:
        x1 = ui.shape[1]
    if y1 is None:
        y1 = ui.shape[0]
    if x1 < 0:
        x1 = ui.shape[1] - x1
    if y1 < 0:
        y1 = ui.shape[0] - y1

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
    if dim == 3:
        if z1 is None:
            z1 = ui.shape[2]
        nsteps = z1 - z0
        limits = [ncolumns, nrows, nsteps]
    if t1 is None:
        t1 = ui.shape[-1]

    # Number of bins- if this is too small, correlation length would be overestimated. Keep it around ncolumns
    if n_bins is None:
        n_bins = int(max(limits) * coarse)

    # Use a portion of data
    if dim == 3:
        z_grid, y_grid, x_grid = z[y0:y1, x0:x1, z0:z1], y[y0:y1, x0:x1, z0:z1], x[y0:y1, x0:x1, z0:z1]
        ui = ui[y0:y1, x0:x1, z0:z1, :]
    elif dim == 2:
        y_grid, x_grid = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
        ui = ui[y0:y1, x0:x1, :]

    # Initialization
    rrs, Dxxs, Dxx_errs = np.zeros((n_bins, t1 - t0)), np.ones((n_bins, t1 - t0)), np.zeros((n_bins, t1 - t0))

    for t in tqdm(list(range(t0, t1)), desc='struc. func. time'):
        # Call velocity field at time t as uu
        uu = ui[..., t]

        # Initialization
        ## m: number of rolls it tries. coarse is a parameter to sample different rs evenly
        #### coarse=1: Compute DLL(r,t) for all possible r. if coarse=0.5, it samples only a half of possible rs.
        ## n: number of data points from which DLL statistics is computed.
        #### coarse2=1: use all data points. (e.g. for 1024*1024 grid, use 1024*1024*coarse2 data points)
        roll_indices = list(range(0, limits[roll_axis], int(1. / coarse)))
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        rr = np.empty((n, m))
        Dxx = np.empty((n, m))

        for j, i in enumerate(roll_indices):
            # for i in range(int(coarse * limits[roll_axis])):
            if dim == 3:
                x_grid_rolled, y_grid_rolled, z_grid_rolled = np.roll(x_grid, i, axis=roll_axis), \
                                                              np.roll(y_grid, i, axis=roll_axis), \
                                                              np.roll(z_grid, i, axis=roll_axis)
                r_grid = np.sqrt(
                    (x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2 + (z_grid_rolled - z_grid) ** 2)
            elif dim == 2:
                x_grid_rolled, y_grid_rolled = np.roll(x_grid, -i, axis=roll_axis), np.roll(y_grid, -i, axis=roll_axis)
                r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)

            uu_rolled = np.roll(uu, -i, axis=roll_axis)
            Dxx_raw = (uu_rolled - uu) ** p
            rr[:, j] = r_grid.flatten()[:n]
            Dxx[:, j] = Dxx_raw.flatten()[:n]

        # flatten arrays to feed to binned_statistic
        rr_flatten, Dxx_flatten = rr.flatten(), Dxx.flatten()
        # Nans are not handled very well in binned_statistic
        # Get rid of nans from rr_flatten and Dxx_flatten
        mask = ~np.isnan(Dxx_flatten)
        rr_flatten, Dxx_flatten = rr_flatten[mask], Dxx_flatten[mask]

        # get a histogram
        # rr_, _, _ = binned_statistic(rr_raw, rr_raw, statistic='mean', bins=n_bins)
        rr_means, rr_edges, binnumber = binned_statistic(rr_flatten, rr_flatten, statistic='mean', bins=n_bins)
        Dxx_, _, _ = binned_statistic(rr_flatten, Dxx_flatten, statistic='mean', bins=n_bins)
        Dxx_err_, _, _ = binned_statistic(rr_flatten, Dxx_flatten, statistic='std', bins=n_bins)
        rr_binwidth = (rr_edges[1] - rr_edges[0])
        rr_ = rr_edges[1:] - rr_binwidth / 2.

        # This is faster?
        _, Dxxs[:, t] = sort2arr(rr_, Dxx_)
        rrs[:, t], Dxx_errs[:, t] = sort2arr(rr_, Dxx_err_)

    # Also return scaled results
    if dim == 3:
        dx, dy, dz = x[0, 1, 0] - x[0, 0, 0], y[1, 0, 0] - y[0, 0, 0], z[0, 0, 1] - z[0, 0, 0]
    elif dim == 2:
        dx, dy = x[0, 1] - x[0, 0], y[1, 0] - y[0, 0]
        dz = None
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu, t0=t0, t1=t1)
    eta = (nu ** 3 / epsilon) ** 0.25
    rrs_scaled = rrs / eta
    Dxxs_scaled = Dxxs / ((epsilon * rrs) ** (float(p) / 3.))
    Dxx_errs_scaled = Dxx_errs / ((epsilon * rrs) ** (float(p) / 3.))

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, Dxxs, Dxx_errs, rrs_scaled, Dxxs_scaled, Dxx_errs_scaled


def get_structure_function_roll(udata, x, y, z=None, indices=('x', 'x'), roll_axis=1, n_bins=None, nu=1.004,
                                x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                                coarse=1.0, coarse2=0.2, notebook=True):
    """
    DEPRECATED! Use get_structure_function()
    A method to compute a generalized structure function

    Structure tensor Dij...k is essentially the generalized variance of the two-point velocity difference
    This method returns the STRUCTURE FUNCTION D_{ij...k}(r x_m) where the tensor indices are "indices"
    and the subscript of x_m, m, is expressed as roll_axis.

    If indices=(0, 0) & roll_axis=1, this returns D_{xx}(r \hat{x}).  # longitudinal, 2nd order
    If indices=(1, 1) & roll_axis=0, this returns D_{yy}(r \hat{y}).  # longitudinal, 2nd order
    If indices=(1, 1) & roll_axis=0, this returns D_{yy}(r \hat{x}).  # transverse, 2nd order
    If indices=(0, 0, 0) & roll_axis=1, this returns D_{xxx}(r \hat{x}).  # longitudinal, 3nd order
    If indices=(0, 1, 0) & roll_axis=1, this returns D_{xyx}(r \hat{x}).  # (1, 2, 1)-component of the 3rd order structure function tensor

    ... Returns rrs, Dijks, Dijk_errs, rrs_scaled, Dijks_scaled, Dijk_errs_scaled

    Parameters
    ----------
    udata
    x: numpy array
        x-coordinate of the spatial grid corresponding to the given udata
        ... it does not have to ve equally spaced
    y: numpy array
        y-coordinate of the spatial grid corresponding to the given udata
        ... it does not have to ve equally spaced
    z: numpy array
        z-coordinate of the spatial grid corresponding to the given udata
        ... it does not have to ve equally spaced
    indices: array-like, default: (1, 1)
        ... tensor indices of the structure function tensor: D_{i,j,..., k}
        ... indices=(1,1) corresponds to D_{xx}.
        ... For a 3D spatial field, x, y, z corresponds to 1, 0, 2 respectively. Recall udata.shape = (height, width, depth, duration)
        ... For a 2D spatial field, x, y corresponds to 1, 0 respectively. Recall udata.shape = (height, width, duration)
    roll_axis: int
        "u" and "roll_axis" determines whether this method returns the longitudinal or transverse structure function
        If you want longitudinal, match "u" and "roll_axis". e.g.- u='ux' and roll_axis=1, u='uy' and roll_axis=0
        If you want transverse, do not match "u" and "roll_axis". e.g.- u='ux' and roll_axis=0, u='uy' and roll_axis=1
    n_bins: int, default=None
        number of bins used to take a histogram of velocity difference
    nu: float
        viscosity
    u: str, default='ux'
        velocity component used to compute the structure functions. Choices are 'ux', 'uy', 'uz'
    x0: int, default: 0
        Specified the region of udata to compute the structure function.
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    x1: int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    y0: int, default: 0
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    y1 int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    z0: int, default: 0
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    z1 int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    t0: int, default: 0
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    t1 int, default: None
        This method uses only udata[y0:y1, x0:x1, z0:z1, t0:t1].
    coarse: float, (0, 1], default: 1
        the first parameter to save computation time related to sampling frequency
        ... The higher "coarse" is, it samples more possible data points.
        ... If "coarse" == 1, it samples all possible data points.
        ... If "coarse" == 0.5, it samples only a half of possible data points.
    coarse2: float, (0, 1], default: 0.2
        the second parameter to save computation time related to making a histogram
        ... Determines how many sampled data points to be used to make a histogram
        ... If "coarse" == 1, it uses all data points to make a histogram.
        ... If "coarse" == 0.5, it uses only a half of data points to make a histogram
    notebook: bool
        ... if True, it uses tqdm.tqdm_notebook instead of tqdm.tqdm

    Returns
    -------
    rrs: numpy array
        two-point separation distance for the structure function
    Dxxs: numpy array
        values of the structure function
    Dxx_errs: numpy array
        error of the structure function
    rrs_scaled: numpy array
        two-point separation distance for the SCALED structure function
    Dxxs_scaled: numpy array
        values of the SCALED structure function
    Dxx_errs_scaled: numpy array
        error of the SCALED structure function
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    # Array sorting
    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and Sorted arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return arr1, arr2

    # tensor indices
    def fix_tensor_indices(tensor_indices):
        """
        Ensures that the given "tensor_indices" is a tuple of integers which correspond to the right directions

        Parameters
        ----------
        tensor_indices: tuple of integers or strings

        Returns
        -------
        tensor_indices_int: tuple of integers
        """
        fixer = {'x': 0, 'y': 1, 'z': 2, 0: 0, 1: 1, 2: 2}
        tensor_indices_int = list(fixer[index] for index in tensor_indices)
        return tensor_indices_int

    def translate_tensor_indices(tensor_indices):
        """
        Returns the name of the axis that tensor indices refer to

        Parameters
        ----------
        tensor_indices: tuple of integers or strings

        Returns
        -------
        tensor_indices_str: str
        """
        translator = {'x': 'x', 'y': 'y', 'z': 'z', 0: 'x', 1: 'y', 2: 'z'}
        tensor_indices_str = ''
        tensor_indices_str = tensor_indices_str.join([translator[index] for index in tensor_indices])
        return tensor_indices_str

    def translate_roll_axis(roll_axis):
        """
        Returns the name of axis corresponding to "roll_axis"
        Parameters
        ----------
        roll_axis: int

        Returns
        -------
        str, name of the axis corresponding to "roll_axis"

        """
        translator = {1: 'x', 0: 'y', 2: 'z'}
        return translator[roll_axis]

    dim = len(udata)

    if x1 is None:
        x1 = udata[0, ...].shape[1]
    if y1 is None:
        y1 = udata[0, ...].shape[0]
    if x1 < 0:
        x1 = udata[0, ...].shape[1] - x1
    if y1 < 0:
        y1 = udata[0, ...].shape[0] - y1

    tensor_indices = fix_tensor_indices(indices)
    tensor_indices_str = translate_tensor_indices(indices)
    roll_axis_str = translate_roll_axis(roll_axis)
    print('... Computing D_{%s}(r \hat{%s})' % (tensor_indices_str, roll_axis_str))

    # order of structure function tensor: p
    p = float(len(indices))

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
    if dim == 3:
        if z1 is None:
            z1 = udata[0, ...].shape[2]
        nsteps = z1 - z0
        limits = [ncolumns, nrows, nsteps]
    if t1 is None:
        t1 = udata[0, ...].shape[-1]

    # Number of bins- if this is too small, correlation length would be overestimated. Keep it around ncolumns
    if n_bins is None:
        n_bins = int(max(limits) * coarse)

    # Use a portion of data
    if dim == 3:
        z_grid, y_grid, x_grid = z[y0:y1, x0:x1, z0:z1], y[y0:y1, x0:x1, z0:z1], x[y0:y1, x0:x1, z0:z1]
        udatai = udata[:, y0:y1, x0:x1, z0:z1, :]
    elif dim == 2:
        y_grid, x_grid = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
        udatai = udata[:, y0:y1, x0:x1, :]

    # Initialization
    rrs, Dijks, Dijk_errs = np.zeros((n_bins, t1 - t0)), np.ones((n_bins, t1 - t0)), np.zeros((n_bins, t1 - t0))

    for t in tqdm(list(range(t0, t1)), desc='struc. func. time'):
        # Initialization
        ## m: number of rolls it tries. coarse is a parameter to sample different rs evenly
        #### coarse=1: Compute DLL(r,t) for all possible r. if coarse=0.5, it samples only a half of possible rs.
        ## n: number of data points from which DLL statistics is computed.
        #### coarse2=1: use all data points. (e.g. for 1024*1024 grid, use 1024*1024*coarse2 data points)
        roll_indices = list(range(0, limits[roll_axis], int(1. / coarse)))
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        rr = np.empty((n, m))
        Dijk = np.empty((n, m))

        for j, i in enumerate(roll_indices):
            # for i in range(int(coarse * limits[roll_axis])):
            if dim == 3:
                x_grid_rolled, y_grid_rolled, z_grid_rolled = np.roll(x_grid, i, axis=roll_axis), \
                                                              np.roll(y_grid, i, axis=roll_axis), \
                                                              np.roll(z_grid, i, axis=roll_axis)
                r_grid = np.sqrt(
                    (x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2 + (z_grid_rolled - z_grid) ** 2)
            elif dim == 2:
                x_grid_rolled, y_grid_rolled = np.roll(x_grid, i, axis=roll_axis), np.roll(y_grid, i, axis=roll_axis)
                r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)

            Dijk_raw = np.ones_like(udata[0, ..., t])
            for tensor_index in tensor_indices:
                uu = udatai[tensor_index, ..., t]
                uu_rolled = np.roll(uu, i, axis=roll_axis)
                Dijk_raw *= (uu_rolled - uu)
            rr[:, j] = r_grid.flatten()[:n]
            Dijk[:, j] = Dijk_raw.flatten()[:n]

        # flatten arrays to feed to binned_statistic
        rr_flatten, Dijk_flatten = rr.flatten(), Dijk.flatten()
        # Nans are not handled very well in binned_statistic
        # Get rid of nans from rr_flatten and Dxx_flatten
        mask = ~np.isnan(Dijk_flatten)
        rr_flatten, Dijk_flatten = rr_flatten[mask], Dijk_flatten[mask]

        # get a histogram
        # rr_, _, _ = binned_statistic(rr_raw, rr_raw, statistic='mean', bins=n_bins)
        rr_means, rr_edges, binnumber = binned_statistic(rr_flatten, rr_flatten, statistic='mean', bins=n_bins)
        Dijk_, _, _ = binned_statistic(rr_flatten, Dijk_flatten, statistic='mean', bins=n_bins)
        Dijk_err_, _, _ = binned_statistic(rr_flatten, Dijk_flatten, statistic='std', bins=n_bins)
        rr_binwidth = (rr_edges[1] - rr_edges[0])
        rr_ = rr_edges[1:] - rr_binwidth / 2.

        # Sort results
        _, Dijks[:, t] = sort2arr(rr_, Dijk_)
        rrs[:, t], Dijk_errs[:, t] = sort2arr(rr_, Dijk_err_)

    # Also return scaled results
    if dim == 3:
        dx, dy, dz = x[0, 1, 0] - x[0, 0, 0], y[1, 0, 0] - y[0, 0, 0], z[0, 0, 1] - z[0, 0, 0]
    elif dim == 2:
        dx, dy = x[0, 1] - x[0, 0], y[1, 0] - y[0, 0]
        dz = None
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu, t0=t0, t1=t1)
    eta = (nu ** 3 / epsilon) ** 0.25
    rrs_scaled = rrs / eta
    Dijks_scaled = Dijks / ((epsilon * rrs) ** (float(p) / 3.))
    Dijk_errs_scaled = Dijk_errs / ((epsilon * rrs) ** (float(p) / 3.))

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, Dijks, Dijk_errs, rrs_scaled, Dijks_scaled, Dijk_errs_scaled


# NEW CODE with improved performance for the higher-order structure functions
## Instead of np.roll, the code below computes the statistics of the velocity difference at two arbitrary chosen points
def get_structure_function(udata, x, y, z=None, nu=1.004,
                           x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                           t0=0, t1=None,
                           p=2, nr=None, nd=1000, mode='long',
                           time_thd=10.,
                           spacing='log',
                           notebook=True):
    """
    Compute the structure function of the velocity field.

    Parameters
    ----------
    udata: ndarray, 2/3D velocity field
    x: ndarray, x-coordinates of the velocity field
    y: ndarray, y-coordinates of the velocity field
    z: ndarray, z-coordinates of the velocity field
    nu: float, kinematic viscosity
    x0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    x1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    y0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    y1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    z0: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    z1: int, index used to specify a spatial region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    t0: int, index used to specify a temporal region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    t1: int, index used to specify a temporal region in which the statistics is computed [y0:y1, x0:x1, z0:z1, t0:t1]
    p: int, order of the structure function
    nr: int, number of points (distances) at which the structure function is evaluated
    nd: int, number of pairs considered to compute the structure function value at a given distance
    mode: str, 'long' or 'trans'
        ... Specify whether longitudinal or transverse correlations is evaluated
    time_thd: float, threshold for the computation time (in sec) to determine a correlation value at a given distance r
        ... default: 10. sec
    spacing: str, 'log' or 'linear'
        ... Specify whether the considered distance r is evenly spaced logarithmically or linearly
    notebook: bool, if True, it will use tqdm_notebook instead of tqdm to display a progress bar

    Returns
    -------
    rrs: 2d array, distances at which the structure function is evaluated, shape: (nr, duration)
    Dijks: 2d array, structure function values at the distances rrs, shape: (nr, duration)
    Dijk_errs: 2d array, standard errors of the structure function values at the distances rrs, shape: (nr, duration)
    rrs_scaled: 2d array, rrs / eta, eta: Kolmogorov length scale estimated from the rate-of-strain tensor
    Dijks_scaled: 2d array, Dijks / (epsilon * rrs ** (p/3)), epsilon: Kolmogorov length scale estimated from the rate-of-strain tensor
    Dijk_errs_scaled: 2d array, Dijk_errs / (epsilon * rrs ** (p/3)), epsilon: Kolmogorov length scale estimated from the rate-of-strain tensor
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    # Array sorting
    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and Sorted arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return arr1, arr2

    def find_nearest(array, value, option='normal'):
        """
        Find an element and its index closest to 'value' in 'array'
        Parameters
        ----------
        array
        value

        Returns
        -------
        idx: index of the array where the closest value to 'value' is stored in 'array'
        array[idx]: value closest to 'value' in 'array'

        """
        # get the nearest value such that the element in the array is LESS than the specified 'value'
        if option == 'less':
            array_new = copy.copy(array)
            array_new[array_new > value] = np.nan
            idx = np.nanargmin(np.abs(array_new - value))
            return idx, array_new[idx]
        # get the nearest value such that the element in the array is GREATER than the specified 'value'
        if option == 'greater':
            array_new = copy.copy(array)
            array_new[array_new < value] = np.nan
            idx = np.nanargmin(np.abs(array_new - value))
            return idx, array_new[idx]
        else:
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

    def mag1(X):
        '''Calculate the length of an array of vectors, keeping the last dimension
        index.'''
        return np.sqrt((np.asarray(X) ** 2).sum(-1))[..., np.newaxis]

    def dot(X, Y):
        '''Calculate the dot product of two arrays of vectors.'''
        return (np.asarray(X) * Y).sum(-1)

    def norm(X):
        '''Computes a normalized version of an array of vectors.'''
        # The norma of the null vector is fixed to be (1,0,0) this is consistent
        # with othe function of this module.
        X = np.asarray(X)
        return X / mag1(X)

    # def rotate_vector_cart(X, alpha, beta=None, gamma=None):
    #     '''Rotates a vector in the Cartesian coordinates by theta (and phi) in 2D (3D)'''
    #
    #     from scipy.spatial.transform import Rotation
    #     X_ = np.asarray(X)
    #
    #     dim = X.shape[0]
    #
    #     if dim == 2:
    #         X_ = np.append(X_, 0)
    #         r = Rotation.from_euler('z', alpha, degrees=True)
    #         return r.apply(X_)[:dim]
    #     elif dim == 3:
    #         if beta is None or gamma is None:
    #             print('... ERROR: angles are not provided!')
    #             print('... One requires three Euler angles to specify a rotation in 3D.')
    #             print(
    #                 '... 1. Rotate by alpha about z \n    2. Rotate by alpha about y \n    3. Rotate by alpha about x')
    #
    #         r = Rotation.from_euler('zyx', [alpha, beta, gamma], degrees=True)
    #         return r.apply(X_)

    dim = len(udata)

    if x1 is None:
        x1 = udata[0, ...].shape[1]
    if y1 is None:
        y1 = udata[0, ...].shape[0]
    if x1 < 0:
        x1 = udata[0, ...].shape[1] - x1
    if y1 < 0:
        y1 = udata[0, ...].shape[0] - y1

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
    if dim == 3:
        if z1 is None:
            z1 = udata[0, ...].shape[2]
        nsteps = z1 - z0
        limits = [ncolumns, nrows, nsteps]
    if t1 is None:
        t1 = udata[0, ...].shape[-1]

    # Number of bins- if this is too small, correlation length would be overestimated. Keep it around ncolumns
    if nr is None:
        nr = max(limits)

    # Use a portion of data
    if dim == 3:
        z_grid, y_grid, x_grid = z[y0:y1, x0:x1, z0:z1], y[y0:y1, x0:x1, z0:z1], x[y0:y1, x0:x1, z0:z1]
        dx, dy, dz = np.abs(x_grid[0, 1, 0] - x_grid[0, 0, 0]), \
                     np.abs(y_grid[1, 0] - y_grid[0, 0, 0]), \
                     np.abs(y_grid[0, 0, 1] - z_grid[0, 0, 0])

    elif dim == 2:
        y_grid, x_grid = y[y0:y1, x0:x1], x[y0:y1, x0:x1]
        dx, dy = np.abs(x_grid[0, 1] - x_grid[0, 0]), np.abs(y_grid[1, 0] - y_grid[0, 0])

    # Initialization
    rrs, Dijks, Dijk_errs = np.zeros((nr, t1 - t0)), np.ones((nr, t1 - t0)), np.zeros((nr, t1 - t0))
    is_R1_reasonable = False

    if dim == 2:
        xmin, xmax, ymin, ymax = np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)
        width, height = xmax - xmin, ymax - ymin
        if spacing == 'log':
            rs_ = np.logspace(np.log10(dx), np.log10(min([width, height])), num=nr)
        else:
            rs_ = np.linspace(dx * 2, min([width, height]) * 1, nr)
    elif dim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid), np.min(
            z_grid), np.max(z_grid)
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin
        if spacing == 'log':
            rs_ = np.logspace(np.log10(dx), np.log10(min([width, height, depth])), num=nr)
        else:
            rs_ = np.linspace(dx, min([width, height, depth]), nr)
    rs = np.empty(nd)
    Dijks_ = np.empty(nd)
    for t in tqdm(list(range(t0, t1)), desc='struc. func. time'):
        for i, r in enumerate(tqdm(rs_, desc='struc. func. r-loop')):
            time0 = time_mod.time()
            for j in range(nd):
                time1 = time_mod.time()
                if (time1 - time0) > time_thd:
                    print('... Terminating for long computation time \n'
                          '(r, number of iterations performed before termination / target) = (%f, %d / %d)' % (
                              r, j, nd))
                    break

                if dim == 2:
                    while not is_R1_reasonable:
                        # Randomly pick a point in space, call it R0
                        X0, Y0 = np.random.random() * width + xmin, np.random.random() * height + ymin
                        X0_ind, _ = find_nearest(x_grid[0, :], X0)
                        Y0_ind, _ = find_nearest(y_grid[:, 0], Y0)
                        R0 = np.asarray([x_grid[0, X0_ind], y_grid[Y0_ind, 0]])
                        # Randomly pick another point in space, call it R1
                        theta = 2 * np.pi * np.random.random()
                        X1, Y1 = X0 + r * np.cos(theta), Y0 + r * np.sin(theta)
                        R1 = np.asarray([X1, Y1])
                        is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin
                        if is_R1_reasonable:
                            X1_ind, _ = find_nearest(x_grid[0, :], X1)
                            Y1_ind, _ = find_nearest(y_grid[:, 0], Y1)
                            R1 = np.asarray([x_grid[0, X1_ind], y_grid[Y1_ind, 0]])
                            if all(R0 == R1):
                                is_R1_reasonable = False
                elif dim == 3:
                    while not is_R1_reasonable:
                        # Randomly pick a point in space, call it R0
                        X0, Y0, Z0 = np.random.random() * width + xmin, np.random.random() * height + ymin, np.random.random() * depth + zmin
                        R0 = np.asarray([X0, Y0, Z0])
                        X0_ind, _ = find_nearest(x_grid[0, :, 0], X0)
                        Y0_ind, _ = find_nearest(y_grid[:, 0, 0], Y0)
                        Z0_ind, _ = find_nearest(z_grid[0, 0, :], Z0)
                        R0 = np.asarray([x_grid[0, X0_ind, 0], y_grid[Y0_ind, 0, 0], z_grid[Z0_ind, 0, 0]])
                        # Randomly pick another point in space, call it R1
                        theta = 2 * np.pi * np.random.random()
                        phi = 2 * np.pi * np.random.random()
                        X1, Y1, Z1 = X0 + r * np.cos(theta) * np.cos(phi), Y0 + r * np.cos(theta) * np.sin(
                            phi), Z0 + r * np.sin(phi)
                        R1 = np.asarray([X1, Y1, Z1])
                        is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin and Z1 < zmax and Z1 > zmin
                        if is_R1_reasonable:
                            X1_ind, _ = find_nearest(x_grid[0, :, 0], X1)
                            Y1_ind, _ = find_nearest(y_grid[:, 0, 0], Y1)
                            Z1_ind, _ = find_nearest(z_grid[0, 0, :], Z1)
                            R1 = np.asarray([x_grid[0, X1_ind, 0], y_grid[Y1_ind, 0, 0], z_grid[0, 0, Z1_ind]])
                            if all(R0 == R1):
                                is_R1_reasonable = False
                R01 = R1 - R0

                basis = vec.get_an_orthonormal_basis(dim, v1=R01)
                # CRUCIAL: make sure to use the same convention to define the direction of the transverse vector
                basis = vec.apply_right_handedness(basis)
                if mode == 'long':
                    RR = basis[:, 0]  # a normalized longitudinal vector
                elif mode == 'trans':
                    # In 3D, transverse strcture function is ill-defined!
                    if dim == 3:
                        print('transverse direction is not well-defined in 3D! Aborting')
                        sys.exit(1)
                    else:
                        RR = basis[:, 1]  # a normalized transverse vector
                rs[j] = mag1(R01)
                udiff = udata[:, Y1_ind, X1_ind, t] - udata[:, Y0_ind, X0_ind, t]
                udiff = udiff[..., 0]
                Dijks_[j] = dot(udiff, norm(RR)) ** p
                is_R1_reasonable = False
            rrs[i, t] = np.nanmean(rs)
            Dijks[i, t] = np.nanmean(Dijks_)
            Dijk_errs[i, t] = np.nanstd(Dijks_)

    # Also return the rescaled quantities
    if dim == 3:
        dx, dy, dz = x[0, 1, 0] - x[0, 0, 0], y[1, 0, 0] - y[0, 0, 0], z[0, 0, 1] - z[0, 0, 0]
    elif dim == 2:
        dx, dy = x[0, 1] - x[0, 0], y[1, 0] - y[0, 0]
        dz = None
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu, t0=t0, t1=t1)
    eta = (nu ** 3 / epsilon) ** 0.25
    rrs_scaled = rrs / eta
    Dijks_scaled = Dijks / ((epsilon * rrs) ** (float(p) / 3.))
    Dijk_errs_scaled = Dijk_errs / ((epsilon * rrs) ** (float(p) / 3.))

    print('mean epsilon, mean eta: ', np.nanmean(epsilon), np.nanmean(eta))
    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, Dijks, Dijk_errs, rrs_scaled, Dijks_scaled, Dijk_errs_scaled


def scale_raw_structure_funciton_long(rrs, Dxxs, epsilon, Dxx_errs=None, nu=1.004, p=2):
    """
    Returns the scaled structure functions when raw structure function data are given
    ... This allows users to scale the structure functions with epsilon and nu input by the users.

    Parameters
    ----------
    rrs: numpy array
        separation between two points, output of get_two_point_vel_corr_iso()
    Dxxs: numpy array
        longitudinal structure function, output of get_two_point_vel_corr_iso()
    Dxx_errs: numpy array
        error of longitudinal structure function, output of get_two_point_vel_corr_iso()
    epsilon: numpy array or float
        dissipation rate
    nu: float
        viscosity
    p: int/float
        order of the structure function

    Returns
    -------
    rrs_s: numpy array
        Scaled r
    Dxxs_s: numpy array
        Scaled structure function
    Dxx_errs_s: numpy array (optional)
        Scaled DLL error
    """
    if type(epsilon) == list:
        epsilon = np.asarray(epsilon)

    eta = compute_kolmogorov_lengthscale_simple(epsilon, nu)
    if type(epsilon) == np.ndarray:
        Dxxs_s, Dxx_errs_s, rrs_s = np.empty_like(Dxxs), np.empty_like(Dxxs), np.empty_like(Dxxs)
        for t in list(range(len(epsilon))):
            Dxxs_s[:, t] = Dxxs[:, t] / (epsilon[t] * rrs[:, t]) ** (p / 3.)
            if Dxx_errs is not None:
                Dxx_errs_s[:, t] = Dxx_errs[:, t] / (epsilon[t] * rrs[:, t]) ** (p / 3.)
            rrs_s[:, t] = rrs[:, t] / eta[t]
    else:
        Dxxs_s = Dxxs / (epsilon * rrs) ** (p / 3.)
        if Dxx_errs is not None:
            Dxx_errs_s = Dxx_errs / (epsilon * rrs) ** (p / 3.)
        rrs_s = rrs / eta

    if Dxx_errs is not None:
        return rrs_s, Dxxs_s, Dxx_errs_s
    else:
        return rrs_s, Dxxs_s,

########## Length scales ##########
## TAYLOR MICROSCALES ##
# Taylor microscales 1: using autocorrelation functions
### DEFAULT ###
def remove_nans_for_array_pair(arr1, arr2):
    """
    remove nans or infs for a pair of 1D arrays, and returns the compressed arrays with the same length
    e.g.: arr1 = [0, 0.1, 0.2, np.nan, 0.4, 0.5], arr2 = [-1.2, 19.2. 155., np.inf, 0.1]
        -> compressed_arr1 = [0, 0.1, 0.2, 0.5], compressed_arr2 = [-1.2, 19.2., 0.1]

    Parameters
    ----------
    arr1: numpy array
        ... may include np.nan and np.inf
    arr2: numpy array
        ... may include np.nan and np.inf
    Returns
    -------
    compressed_arr1: numpy array
        ... without any np
    compressed_arr2: numpy array
    """

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    mask1 = get_mask_for_nan_and_inf(arr1)
    mask2 = get_mask_for_nan_and_inf(arr2)
    mask = ~(~mask1 * ~mask2)

    arr1 = ma.array(arr1, mask=mask)
    arr2 = ma.array(arr2, mask=mask)
    compressed_arr1 = arr1.compressed()
    compressed_arr2 = arr2.compressed()
    return compressed_arr1, compressed_arr2


def get_taylor_microscales(r_long, f_long, r_tran, g_tran, residual_thd=0.015, deg=2, return_err=False, npt_min=4):
    """
    Returns Taylor microscales as the curvature of the autocorrelation functions at r=0
    ... Algorithm:
    ...     (1) Polynomial fit (cubic) the long./trans. autocorr functions
    ...     (2) Evaluate its second derivate at r=0.
    ...     (3) Relate that value to the taylor microscale
    ... Feed the results from get_two_point_vel_corr_iso().

    Parameters
    ----------
    r_long: numpy 2d array with shape (no. of elements, duration)
        ... r for longitudinal autoorrelation values
    f_long: numpy 2d array with shape (no. of elements, duration)
        ... longitudinal autoorrelation values
    r_tran: numpy 2d array with shape (no. of elements, duration)
        ... r for longitudinal autoorrelation values
    g_tran: numpy 2d array with shape (no. of elements, duration)
        ... longitudinal autoorrelation values
    residual_thd: float
        ... threshold for the residuals of the polynomial fit
    deg: int, default 2, degree of the polynomial fit
    return_err, bool, default False, whether to return the error of the polynomial fit
    npt_min: int, default 4, minimum number of points for the polynomial fit to compute Taylor's microscale

    Returns
    -------
    lambda_f: numpy 2d array with shape (duration, )
        ... Longitudinal Taylor microscale
    lambda_g: numpy 2d array with shape (duration, )
        ... Transverse Taylor microscale
    lambda_err_f: numpy 2d array with shape (duration, ), optional
    lambda_err_g: numpy 2d array with shape (duration, ), optional
    """

    def compute_lambda_from_autocorr_func(r, g, deg=2):
        """
        Parameters
        ----------
        r: numpy 1d array
        ... r for longitudinal autoorrelation function
        g: numpy 1d array
        ... longitudinal autoorrelation values
        n: degree of polynomial fit, default: 3

        Returns
        -------
        lambdag: taylor microscale
        """
        import warnings
        warnings.simplefilter('ignore', np.RankWarning)

        # Make autocorrelation function an even function. This helps computing the second derivative robustly
        r_tmp = np.concatenate((-np.flip(r, axis=0)[:-1], r))
        g_tmp = np.concatenate((np.flip(g, axis=0)[:-1], g))

        z, residual, rank, singular_values, rcond = np.polyfit(r_tmp, g_tmp, deg, full=True)
        p = np.poly1d(z)  # poly class instance
        p_der2 = np.polyder(p, m=2)  # take the second derivative
        lambdag = (-p_der2(0) / 2.) ** -0.5
        return lambdag, residual

    n, duration = r_long.shape
    # data = [r_long, f_long, r_tran, g_tran]
    # # Remove nans if necessary
    # for i, datum in enumerate(data):
    #     if ~np.isnan(data[i]).any():
    #         data[i] = data[i][~np.isnan(data[i])]

    # initialize
    lambda_f, lambda_g, lambda_err_f, lambda_err_g = [], [], [], []
    for t in range(duration):

        # if r_long contains nans, polynomial fitting fails. so clean it up.
        r_long_tmp, f_long_tmp = remove_nans_for_array_pair(r_long[:, t], f_long[:, t])
        r_tran_tmp, g_tran_tmp = remove_nans_for_array_pair(r_tran[:, t], g_tran[:, t])

        if len(f_long_tmp) == 0 or len(g_tran_tmp) == 0:
            lambda_f.append(np.nan)
            lambda_g.append(np.nan)
            lambda_err_f.append(np.nan)
            lambda_err_g.append(np.nan)
        else:
            # Make sure that f(r=0, t)=g(r=0,t)=1
            f_long_tmp /= f_long_tmp[0]
            g_tran_tmp /= g_tran_tmp[0]

            lambdafs, lambdags = [], []
            residuals_f, residuals_g = [], []
            # Extract lambda from autocorrelation functions (long. and trans.)
            for n in range(npt_min, len(f_long_tmp)):
                lambda_f_tmp, residual_f = compute_lambda_from_autocorr_func(r_long_tmp[:n], f_long_tmp[:n], deg=deg)
                if len(residual_f) == 0:
                    residual_f = 0
                else:
                    residual_f = residual_f[0]

                if residual_f < residual_thd:
                    lambdafs.append(lambda_f_tmp)
                    residuals_f.append(residual_f)
                else:
                    break
            for n in range(npt_min, len(g_tran_tmp)):
                lambda_g_tmp, residual_g = compute_lambda_from_autocorr_func(r_tran_tmp[:n], g_tran_tmp[:n], deg=deg)
                if len(residual_g) == 0:
                    residual_g = 0
                else:
                    residual_g = residual_g[0]
                if residual_g < residual_thd:
                    lambdags.append(lambda_g_tmp)
                    residuals_g.append(residual_g)
                else:
                    break
            # Insert lambdas to lists
            lambda_f.append(np.nanmean(lambdafs))
            lambda_g.append(np.nanmean(lambdags))
            lambda_err_f.append(np.nanstd(lambdafs))
            lambda_err_g.append(np.nanstd(lambdags))

    if not return_err:
        return np.asarray(lambda_f), np.asarray(lambda_g)
    else:
        return np.asarray(lambda_f), np.asarray(lambda_g), np.asarray(lambda_err_f), np.asarray(lambda_err_g)


# Taylor microscales 2: isotropic formula. Must know epsilon beforehand
def get_taylor_microscales_iso(udata, epsilon, nu=1.004):
    """
    Return Taylor microscales computed by isotropic formulae: lambda_g_iso = (15 nu * u_irms^2 / epsilon) ^ 0.5

    Parameters
    ----------
    udata: nd array
    epsilon: float or array with the same length as udata
    nu: float
        viscoty

    Returns
    -------
    lambda_f_iso: numpy array
        longitudinal Taylor microscale
    lambda_g_iso: numpy array
        transverse Taylor microscale
    """
    u_irms = get_characteristic_velocity(udata)
    lambda_g_iso = np.sqrt(15. * nu * u_irms ** 2 / epsilon)
    lambda_f_iso = np.sqrt(30. * nu * u_irms ** 2 / epsilon)
    return lambda_f_iso, lambda_g_iso


# def get_taylor_microscale_heatmaps(udata, dx, dy, dz=None):
#     dim = udata.shape[0]
#     duidxj = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz)
#
#     uu2 = np.zeros_like(udata[0])
#     for d in range(dim):
#         uu2 += udata[d, ...] ** 2
#     uu2 /= float(dim)
#
#     lambda_f = np.sqrt(2 * uu2 / duidxj[..., 0, 0] ** 2)
#     lambda_g = np.sqrt(2 * uu2 / duidxj[..., 0, 1] ** 2)
#     return lambda_f, lambda_g
#


## INTEGRAL SCALES ##
# Integral scales 1: using autocorrelation functions
### DEFAULT ###
def get_integral_scales(r_long, f_long, r_tran, g_tran, method='trapz'):
    """
    Returns integral scales computed by using autocorrelation functions
    ... To have a meaningful result, the autocorrelation functions must be fully resolved.

    Parameters
    ----------
    r_long: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    f_long numpy array
        ... output of get_two_point_vel_corr_iso(...)
    r_tran numpy array
        ... output of get_two_point_vel_corr_iso(...)
    g_tran numpy array
        ... output of get_two_point_vel_corr_iso(...)
    method: string, default: 'trapz'
        ... integration method, choose from quadrature or trapezoidal

    Returns
    -------
    L11: numpy array
        longitudinal integral length scale
    L22: numpy array
        transverse integral length scale
    """
    n, duration = r_long.shape
    data = [r_long, f_long, r_tran, g_tran]
    # Remove nans if necessary
    for i, datum in enumerate(data):
        if ~np.isnan(data[i]).any():
            data[i] = data[i][~np.isnan(data[i])]
    # interpolate data (3rd order spline)
    L11, L22 = [], []
    for t in range(duration):
        # hide np.nans from the arrays
        cond1, cond2 = ~np.isnan(f_long[:, t]), ~np.isnan(r_long[:, t])
        mask = cond1 * cond2
        # quadrature
        if method == 'quad':
            rmin, rmax = np.nanmin(r_long), np.nanmax(r_long)
            f_spl = UnivariateSpline(r_long[mask, t], f_long[mask, t] / f_long[0, t], s=0,
                                     k=3)  # longitudinal autocorrelation func.
            g_spl = UnivariateSpline(r_tran[mask, t], g_tran[mask, t] / g_tran[0, t], s=0,
                                     k=3)  # transverse autocorrelation func.
            L11.append(integrate.quad(lambda r: f_spl(r), rmin, rmax)[0])
            L22.append(integrate.quad(lambda r: g_spl(r), rmin, rmax)[0])
        # trapezoidal
        elif method == 'trapz':
            L11.append(np.trapz(f_long[mask, t], r_long[mask, t]))
            L22.append(np.trapz(g_tran[mask, t], r_tran[mask, t]))
    return L11, L22


# Integral scales 2: using autocorrelation tensor. Should be equivalent to get_integral_scales()
def get_integral_scales_using_rij(udata, Rij, rmax, n=100):
    """
    [DEPRICATED- computing the autocorrelation tensor could be time-consuming]
    Use autocorrelation tensor, Rij to calculate integral length scale

    Parameters
    ----------
    udata
    Rij: numpy array
        two-point velocity autocorrelation tensor
        ... can be obtained by get_two_point_vel_corr_iso(...) but may take a long time
    rmax: float
    n: int
        The higher n is, the integrand function becomes more smooth

    Returns
    -------
    L11: nd array
        ... longitudinal integral length scale
    L22: nd array
        ... transverse integral length scale
    """
    x = np.linspace(0, rmax, n)
    y = [0] * n
    z = [0] * n
    rs = np.stack((x, y, z))

    duration = udata.shape[-1]
    L11, L22 = [], []
    for t in range(duration):
        integrand11 = Rij(0, 0, rs, t, udata) / Rij(0, 0, (0.0000001, 0, 0), t, udata)
        integrand22 = Rij(1, 1, rs, t, udata) / Rij(1, 1, (0.0000001, 0, 0), t, udata)
        L11.append(np.trapz(integrand11[1:], x[1:]))
        L22.append(np.trapz(integrand22[1:], x[1:]))
    return L11, L22


# Integral scales 3: isotropic, using E(k). Must know a full 1d spectrum
def get_integral_scales_iso_spec(udata, e_k, k):
    """
    Integral scale defined through energy spectrum.
    ... Assumes isotropy and a full 1D energy spectrum. Pope 6.260.
    ... To have a meaningful result, the energy spectrum must be fully resolved.

    Parameters
    ----------
    udata
    e_k: numpy array
        output of get_energy_spectrum()
    k: numpy array
        output of get_energy_spectrum()

    Returns
    -------
    L_iso_spec: 1d array
    """

    duration = udata.shape[-1]
    u2_irms = get_characteristic_velocity(udata) ** 2
    L_iso_spec = []
    for t in range(duration):
        try:
            L_iso_spec.append(np.pi / (2. * u2_irms[t]) * np.trapz(e_k[1:, t] / k[1:, t], k[1:, t]))
        except:
            L_iso_spec.append(np.pi / (2. * u2_irms[t]) * np.trapz(e_k[1:, t] / k[1:], k[1:]))
    return np.asarray(L_iso_spec)


# Integral scales 4: characteristic size of Large-eddies
def get_integral_scale_large_eddy(udata, epsilon):
    """
    dissipation rate (epsilon) is known to be proportional to u'^3 / L.

    Some just define an integral scale L as u'^3 / epsilon.
    This method just returns this ratio. It is often interpreted as the characteristic scale of large eddies.

    Parameters
    ----------
    udata: numpy array
    epsilon: numpy array / float
        dissipation rate

    Returns
    -------
    L: numpy array
        integral length scale (characterisic size of large eddies)
    """
    u_irms = get_characteristic_velocity(udata)
    L = u_irms ** 3 / epsilon
    return L


def get_integral_velocity_scale(udata):
    """
    Return integral velocity scale- identical to u' (characteristic velocity)
    .. See get_characteristic_velocity() for more details

    Parameters
    ----------
    udata: numpy array

    Returns
    -------
    u_irms: See get_characteristic_velocity()
    """
    return get_characteristic_velocity(udata)


## KOLMOGOROV SCALES ##
### DEFAULT ###
def get_kolmogorov_scale(udata, dx, dy, dz=None, nu=1.004):
    """
    Returns kolmogorov LENGTH scale
    ... estimates dissipation rate from the rate-of-strain tensor
    ... eta = (nu ** 3 / epsilon) ** 0.25

    Parameters
    ----------
    udata: numpy array, a velocity field
    dx: float
        data spacing in x
    dy: float
        data spacing in y
    dz: float, optional
        data spacing in z
    nu: float
        viscosity

    Returns
    -------
    eta: numpy array
        kolmogorov length scale
    """
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    eta = (nu ** 3 / epsilon) ** 0.25
    return eta


########## ALL SCALES (LENGTH, VELOCITY, TIME)  ##########
def get_integral_scales_all(udata, dx, dy, dz=None, nu=1.004):
    """
    Returns integral scales of turbulence
    ... L = u_irms ** 3 / epsilon
    ... U = u_irms
    ... T = L/U #
    Parameters
    ----------
    udata: numpy array
    dx: float
        data spacing in x
    dy: float
        data spacing in y
    dz: float
        data spacing in z
    nu: float
        viscosity

    Returns
    -------
    L_le: 1d array, integral length scale: u_irms ** 3 / epsilon
    u_L: 1d array: characteristic velocity, characteristic velocity
    tau_L: 1d array: large eddy turnover time, L_le / u_L
    """
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    energy_avg = get_spatial_avg_energy(udata)[0]
    L_le = get_integral_scale_large_eddy(udata, epsilon)
    u_L = energy_avg ** 0.5
    tau_L = L_le / u_L
    return L_le, u_L, tau_L


def get_taylor_microscales_all(udata, r_long, f_long, r_tran, g_tran):
    """
    Returns Taylor microscales (Length, Velocity, Time) based on autocorrelation functions
    ... More time-consuming than get_taylor_microscales_all_iso() but more basic
    ...... the time-consuming part is the estimation of the autocorrelation functions: r_long, f_long, r_tran, g_tran

    Parameters
    ----------
    udata: numpy array
        velocity field array
    r_long: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    f_long: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    r_tran: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    g_tran: numpy array
        ... output of get_two_point_vel_corr_iso(...)

    Returns
    -------
    lambda_f: 1d array
        Taylor microscale (length)
    u_lambda: 1d array
        Taylor microscale (velocity)
    tau_lambda: 1d array
        Taylor microscale (time)
    """
    lambda_f, lambda_g = get_taylor_microscales(r_long, f_long, r_tran, g_tran)
    u_lambda = get_characteristic_velocity(udata)  # u_irms = u_lambda
    tau_lambda = lambda_g / u_lambda  # other way to define the time scale is through temporal autocorrelation
    return lambda_g, u_lambda, tau_lambda


def get_taylor_microscales_all_iso(udata, epsilon, nu=1.004):
    """
    Returns Taylor microscales using isotropic formula based on udata and dissipation rate
    ... Quick but only applicable to isotropic turbulence
    ... a dissipation rate must be accurately estimated to have a meaningful result

    Parameters
    ----------
    udata: numpy array
        velocity field array
    epsilon: numpy array
        disspation rate
    nu: numpy array
        viscosity

    Returns
    -------
    lambda_f_iso: 1d array
        Taylor microscale (length)
    u_lambda_iso: 1d array
        Taylor microscale (velocity)
    tau_lambda_iso: 1d array
        Taylor microscale (time)
    """
    lambda_f_iso, lambda_g_iso = get_taylor_microscales_iso(udata, epsilon, nu=nu)
    u_lambda_iso = get_characteristic_velocity(udata)  # u_irms = u_lambda
    tau_lambda_iso = lambda_g_iso / u_lambda_iso  # other way to define the time scale is through temporal autocorrelation
    return lambda_g_iso, u_lambda_iso, tau_lambda_iso


def get_kolmogorov_scales_all(udata, dx, dy, dz, nu=1.004):
    """
    Returns Kolmogorov scales (Length, Velocity, Time) based on udata and dissipation rate

    Parameters
    ----------
    udata: numpy array
    dx: float
        data spacing in x
    dy: float
        data spacing in y
    dz: float
        data spacing in z
    nu: float
        viscosity

    Returns
    -------
    eta: 1d array
    u_eta: 1d array
    tau_eta: 1d array
    """
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    eta = (nu ** 3 / epsilon) ** 0.25
    u_eta = (nu * epsilon) ** 0.25
    tau_eta = (nu / epsilon) ** 0.5
    return eta, u_eta, tau_eta


########## REYNOLDS NUMBERS ##########
def get_turbulence_re(udata, dx, dy, dz=None, nu=1.004):
    """
    Returns turbulence reynolds number (Pope 6.59)
    ... Integral Reynolds number

    Parameters
    ----------
    udata: numpy array
    dx: float
        data spacing in x
    dy: float
        data spacing in y
    dz: float
        data spacing in z
    nu: float
        viscosity

    Returns
    -------
    Re_L: numpy array
        Turbulence Reynolds number
    """
    L, u_L, tau_L = get_integral_scales_all(udata, dx, dy, dz, nu=nu)
    Re_L = u_L * L / nu
    return Re_L


def get_taylor_re(udata, r_long, f_long, r_tran, g_tran, nu=1.004):
    """
    Returns Taylor reynolds number (Pope 6.63) from autocorrelation functions

    Parameters
    ----------
    udata: numpy array
        velocity field array
    r_long: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    f_long: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    r_tran: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    g_tran: numpy array
        ... output of get_two_point_vel_corr_iso(...)
    nu: float, default: 1.004 (water, in mm^2/s)
        viscosity

    Returns
    -------
    Re_lambda: array
        Taylor Reynolds number
    """
    lambda_g, u_irms, tau_lambda = get_taylor_microscales_all(udata, r_long, f_long, r_tran, g_tran)
    Re_lambda = u_irms * lambda_g / nu
    return Re_lambda


def get_taylor_re_iso(udata, epsilon, nu=1.004):
    """
    Returns Taylor reynolds number (Pope 6.63) using isotropic formula

    Parameters
    ----------
    udata: numpy array
        velocity field array
    epsilon: numpy array
        disspation rate
    nu: numpy array
        viscosity

    Returns
    -------
    Re_lambda_iso: array
        Taylor Reynolds number
    """

    lambda_g_iso, u_irms_iso, tau_lambda_iso = get_taylor_microscales_all_iso(udata, epsilon, nu=nu)
    Re_lambda_iso = u_irms_iso * lambda_g_iso / nu
    return Re_lambda_iso


# SKEWNESS AND KUTROSIS
def get_skewness(udata, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None):
    """
    Computes skewness of a given velocity field
    ... DO NOT USE scipy.stats.skew()- which computes skewness in the language of probability
    ... i.e. probabilistic skewness is defined through its probabilistic moments (X-E[X])^n
    ... In turbulence, skewnesss is defined using the DERIVATIVES of turbulent velocity
    ... In turbulence, skewness is approximately -0.4 according according to Kambe

    Parameters
    ----------
    udata: nd array, a velocity field
    x0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    x1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    y0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    y1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    z0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    z1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    t0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    t1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]

    Returns
    -------
    skewness: 1d array
    """
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim = udata.shape[0]

    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, t0:t1]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
            udata = udata[:, y0:y1, x0:x1, z0:z1, t0:t1]
    if dim == 2:
        duxdx = np.gradient(udata[0], axis=1)
        duydy = np.gradient(udata[1], axis=0)
        m3 = np.nanmean(duxdx ** 3 + duydy ** 3, axis=(0, 1))
        m2 = np.nanmean(duxdx ** 2 + duydy ** 2, axis=(0, 1))
    elif dim == 3:
        duxdx = np.gradient(udata[0], axis=1)
        duydy = np.gradient(udata[1], axis=0)
        duzdz = np.gradient(udata[2], axis=2)
        m3 = np.nanmean(duxdx ** 3 + duydy ** 3 + duzdz ** 3, axis=(0, 1, 2))
        m2 = np.nanmean(duxdx ** 2 + duydy ** 2 + duzdz ** 2, axis=(0, 1, 2))
    skewness = m3 / m2 ** 1.5
    return skewness


def get_kurtosis(udata, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None):
    """
    Computes kurtosis of a given velocity field
    ... DO NOT USE scipy.stats.kurtosis()- which computes skewness in the language of probability
    ... i.e. probabilistic skewness is defined through its probabilistic moments (X-E[X])^n
    ... In turbulence, kurotsis is defined using the DERIVATIVES of turbulent velocity
    ... In turbulence, kurtosis is approximately 7.2 according to Kambe

    Parameters
    ----------
    udata: nd array, a velocity field
    x0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    x1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    y0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    y1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    z0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    z1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    t0: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]
    t1: int, index to specify the region of interest in udata; udata[:, y0:y1, x0:x1, (z0:z1), t0:t1]

    Returns
    -------
    kurtosis: 1d array
    """
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, t0:t1]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
            udata = udata[:, y0:y1, x0:x1, z0:z1, t0:t1]
    if dim == 2:
        duxdux = np.gradient(udata[0], axis=1)
        duyduy = np.gradient(udata[1], axis=0)
        m4 = np.nanmean(duxdux ** 4 + duyduy ** 4, axis=(0, 1))
        m2 = np.nanmean(duxdux ** 2 + duyduy ** 2, axis=(0, 1))
    elif dim == 3:
        duxdux = np.gradient(udata[0], axis=1)
        duyduy = np.gradient(udata[1], axis=0)
        duzduz = np.gradient(udata[2], axis=2)
        m4 = np.nanmean(duxdux ** 4 + duyduy ** 4 + duzduz ** 4, axis=(0, 1, 2))
        m2 = np.nanmean(duxdux ** 2 + duyduy ** 2 + duzduz ** 2, axis=(0, 1, 2))
    kurtosis = m4 / m2 ** 2
    return kurtosis


########## Sample velocity fields ##########
def rankine_vortex_2d(xx, yy, x0=0, y0=0, gamma=1., a=1.):
    """
    Reutrns a 2D velocity field with a single Rankine vortex at (x0, y0)

    Parameters
    ----------
    xx: numpy array
        x-coordinate, 2d grid
    yy: numpy array
        y-coordinate, 2d grid
    x0: float
        x-coordinate of the position of the rankine vortex
    y0: float
        y-coordinate of the position of the rankine vortex
    gamma: float
        circulation of the rankine vortex
    a: float
        core radius of the rankine vortex

    Returns
    -------
    udata: nd array, velocity field
        ... udata[0] = ux
        ... udata[1] = uy

    """
    rr, phi = cart2pol(xx - x0, yy - y0)

    cond = rr <= a
    ux, uy = np.empty_like(rr), np.empty_like(rr)
    # r <= a
    ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
    # r > a
    ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
    uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

    udata = np.stack((ux, uy))

    return udata


def rankine_vortex_line_3d(xx, yy, zz,
                           x0=0, y0=0, z0=0,
                           gamma=1., a=1.,
                           ux0=0, uy0=0, uz0=0, axis=2):
    """
    Reutrns a 3D velocity field with a Rankine vortex filament at (x0, y0, z)

    Parameters
    ----------
    xx: 3d numpy grid
    yy: 3d numpy grid
    zz: 3d numpy grid
    x0: float, location of Rankine vortex filament
    y0: float, location of Rankine vortex filament
    gamma: float, circulation
    a: float, core radius
    uz0: float, constant velocity component in the z-direction

    Returns
    -------
    udata: nd array, velocity field
        ... udata[0] = ux
        ... udata[1] = uy
        ... udata[2] = uz
    """
    # rr, theta, phi = cart2sph(xx - x0, yy - y0, zz - z0)
    # rr, phi = cart2pol(xx - x0, yy - y0)
    # get_time_avg_energy
    # cond = rr < a
    # ux, uy, uz = np.empty_like(rr), np.empty_like(rr), np.empty_like(rr)

    if axis == 2:  # vortex filament is along the z-axis: (x, y, z) = (x0, y0, z)
        rr, phi = cart2pol(xx - x0, yy - y0)
        cond = rr < a
        ux, uy, uz = np.empty_like(rr), np.empty_like(rr), np.empty_like(rr)

        ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
        uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
        ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
        uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])
        uz = np.ones_like(uz) * uz0
    elif axis == 0:  # vortex filament is along the y-axis: (x, y, z) = (x0, y, z0)
        rr, phi = cart2pol(zz - z0, xx - x0)
        cond = rr < a
        ux, uy, uz = np.empty_like(rr), np.empty_like(rr), np.empty_like(rr)

        uz[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
        ux[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
        uz[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
        ux[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

        uy = np.ones_like(uy) * uy0
    elif axis == 1:  # vortex filament is along the x-axis: (x, y, z) = (x, y0, z0)
        rr, phi = cart2pol(yy - y0, zz - z0)
        cond = rr < a
        ux, uy, uz = np.empty_like(rr), np.empty_like(rr), np.empty_like(rr)

        uy[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
        uz[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
        uy[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
        uz[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

        ux = np.ones_like(ux) * ux0
    udata = np.stack((ux, uy, uz))

    return udata


def lamb_oseen_vortex_2d(xx, yy, x0=0, y0=0, gamma=1., a0=1., nu=1., t=0):
    """
    Return a 2D velocity field with a single Rankine vortex at (x0, y0)

    Parameters
    ----------
    xx: numpy array
        x-coordinate, 2d grid
    yy: numpy array
        y-coordinate, 2d grid
    x0: float
        x-coordinate of the position of the rankine vortex
    y0: float
        y-coordinate of the position of the rankine vortex
    gamma: float
        circulation of the rankine vortex
    a: float
        core radius of the rankine vortex

    Returns
    -------
    udata: nd array, velocity field
        ... udata[0] = ux
        ... udata[1] = uy
    """
    rr, phi = cart2pol(xx - x0, yy - y0)

    a = np.sqrt(a0 ** 2 + 4 * nu * t)

    utheta = gamma / (2 * np.pi * rr) * (1 - np.exp(- rr ** 2 / a ** 2))
    ux, uy = - utheta * np.sin(phi), utheta * np.cos(phi)

    udata = np.stack((ux, uy))
    return udata


def lamb_oseen_vortex_line_3d(xx, yy, zz, x0=0, y0=0, z0=0, gamma=1., a0=1., nu=1., t=0,
                              ux0=0, uy0=0, uz0=0, axis=2):
    """
    Returns a velocity field (udata with shape (3, nrows, ncols, nstacks) induced by
    a Lamb-Oseen voretx line of a diameter a0 and circulation gamma
    ... Lambd-Oseen vortex core grows as \sqrt{a0**2 + 4 \nu t}
    ... Azimuthal velocity: utheta = gamma / (2 * np.pi * rr) * (1 - np.exp(- rr ** 2 / a ** 2))
    ... an orientation of a vortex can be specified
    ... a rectilinear stream can be added along the x-/y-/z-direction

    Parameters
    ----------
    xx: 3d array, x-coordinate
    yy: 3d array, y-coordinate
    zz: 3d array, y-coordinate
    x0: float, x-coordinate of the point where the vortex line penetrates
    y0: float, y-coordinate of the point where the vortex line penetrates
    z0: float, z-coordinate of the point where the vortex line penetrates
    gamma: float, circulation
    a0: float, core radius
    nu: float, kinematic viscosity
    t: float, time,
        ... Lambd-Oseen vortex core grows as \sqrt{a0**2 + 4 \nu t}
    ux0: float, a stream velocity, default = 0
    uy0: float, a stream velocity, default = 0
    uz0: float, a stream velocity, default = 0
    axis: int, axis of the symmetry,  default = 2 (z-axis)
        ... Choose from [0, 1, 2] corresponding to (y, x, z)

    Returns
    -------
    udata: 4d array, a velocity field (3, nrows, ncols, nstacks)
    """
    a = np.sqrt(a0 ** 2 + 4 * nu * t)
    if axis == 2:  # vortex filament is along the z-axis: (x, y, z) = (x0, y0, z)
        rr, phi = cart2pol(xx - x0, yy - y0)
        utheta = gamma / (2 * np.pi * rr) * (1 - np.exp(- rr ** 2 / a ** 2))
        ux, uy = - utheta * np.sin(phi), utheta * np.cos(phi)
        uz = np.ones_like(ux) * uz0

    elif axis == 0:  # vortex filament is along the y-axis: (x, y, z) = (x0, y, z0)
        rr, phi = cart2pol(zz - z0, xx - x0)
        utheta = gamma / (2 * np.pi * rr) * (1 - np.exp(- rr ** 2 / a ** 2))
        uz, ux = - utheta * np.sin(phi), utheta * np.cos(phi)
        uy = np.ones_like(ux) * uy0
    elif axis == 1:  # vortex filament is along the x-axis: (x, y, z) = (x, y0, z0)
        rr, phi = cart2pol(yy - y0, zz - z0)
        utheta = gamma / (2 * np.pi * rr) * (1 - np.exp(- rr ** 2 / a ** 2))
        uy, uz = - utheta * np.sin(phi), utheta * np.cos(phi)
        ux = np.ones_like(uy) * ux0
    udata = np.stack((ux, uy, uz))
    return udata


def get_unidirectional_flow(xx, yy, t=np.asarray([0, 1]), U=10, c=0,
                            decay=None, decay_scale=1.):
    """
    Returns udata of a unidirectional flow. The forwarding direciton is described by a tangent vector t.

    Parameters
    ----------
    xx: 2d array, x coordinates
    yy: 2d array, y coordinates
    t: 1d array with two elements specifying a direction of the flow (y, x)
    U: float, velocity magnitude
    c: float, a parameter relevant for decay=='linear' or 'exponential
        ... This parameter determines where the field starts to decrease either linearly or exponentially
        ... e.g. c = 100, decay=='linear', t=(0, 1)
        ...... This makes a field with a constant flow up to xx<c (c and xx must share the units)
        ...... Then, the field linearly decays to zero at max(xx)
    decay: str, options: ["linear", "exponential"] default: None
        ... If given, the field decays to zero at the edge of the field
    decay_scale: float, a relevant parameter for an exponentially decaying field, default: 1.
        ... The higher, decay_scale is, the flow decays faster to zero.
        ... udata = U * np.exp(- decay_scale * dd / np.max(dd))
        ... If given, the field decays to zero

    Returns
    -------
    udata: 3d numpy array of a unidirectional field

    """
    theta = np.arctan2(t[0], t[1])
    ux = np.ones_like(xx) * U * np.cos(theta)
    uy = np.ones_like(xx) * U * np.sin(theta)
    udata = np.stack((ux, uy))

    dd = np.abs((xx + np.tan(theta) * yy + c)) / np.sqrt(1 + np.tan(theta) ** 2)
    if decay in ['lin', 'linear']:
        udata = udata * (1 - dd / np.max(dd))
    elif decay in ['exp', 'exponential']:
        udata = udata * np.exp(- decay_scale * dd / np.max(dd))
    else:
        pass
    return udata


def get_sample_turb_field_3d(return_coord=True):
    """
    Returns udata=(ux, uy, uz) of a slice of isotropic, homogeneous turbulence data (DNS, JHTD)

    Parameters
    ----------
    return_coord: bool, default: True
        ... If True, it returns a positional grid (xxx, yyy, zzz)

    Returns
    -------
    udata: numpy array
        velocity field
    xx: numpy array
        x-coordinate of 3D grid
    yy: numpy array
        y-coordinate of 3D grid
    zz: numpy array
        z-coordinate of 3D grid
    """
    # get module location
    mod_loc = os.path.abspath(__file__)
    pdir, filename = os.path.split(mod_loc)
    datapath = os.path.join(pdir, 'reference_data/isoturb_slice2.h5')
    data = h5py.File(datapath, 'r')

    keys = list(data.keys())
    keys_u = [key for key in keys if 'u' in key]
    keys_u = natural_sort(keys_u)
    duration = len(keys_u)
    depth, height, width, ncomp = data[keys_u[0]].shape
    udata = np.empty((ncomp, height, width, depth, duration))

    Lx, Ly, Lz = 2 * np.pi, 2 * np.pi, 2 * np.pi
    dx = dy = dz = Lx / 1024

    for t in range(duration):
        udata_tmp = data[keys_u[t]]
        udata_tmp = np.swapaxes(udata_tmp, 0, 3)
        udata[..., t] = udata_tmp
    data.close()

    if return_coord:
        x, y, z = list(range(width)), list(range(height)), list(range(depth))
        xx, yy, zz = np.meshgrid(y, x, z)
        xx, yy, zz = xx * dx, yy * dy, zz * dz
        return udata, xx, yy, zz
    else:
        return udata


def fftnoise2d(f):
    """
    Returns a inverse FT of a given noise after multiplying random phase

    Parameters
    ----------
    f: 2d array, frequency of noise

    Returns
    -------
    noise = np.fft.ifftn(f*random_phase).real, 2d array

    """
    f = np.array(f, dtype='complex')
    phases = np.random.rand(f.shape[0], f.shape[1]) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f *= phases
    noise = np.fft.ifftn(f).real
    return noise


def band_limited_noise2d(min_freq=None, max_freq=None, nsamples=1024, samplerate=1., exponent=-5. / 3.):
    """
    Generates a 2D array of noise with a specific spectral distribution function (k^exponent)
    ... the output noise could be bandlimited [min_freq, max_freq]
    ... the output noise could have a simple frequency dependence
    ......exponent=0: White, exponent=-1, Pink, exponent=-5/3: Kolmogorov, etc.

    Parameters
    ----------
    min_freq: float, the bandwidth of the noise is [min_freq, max_freq]
    max_freq: float, the bandwidth of the noise is [min_freq, max_freq]
    nsamples: int, number of samples in 1D (length of the field)
    samplerate: float, spacing of the field
    exponent: float, frequency dependence of the noise... f^exponent
        ... This should be [-7.5, 0]

    Returns
    -------
    noise: 2d array with shape (nsamples, nsamples)
    """
    print(
        'band_limited_noise2d():: this function may give a noise with a different f-dependence \n The deviation depends on nsamples, samplerate, therefore, is difficult to generalize. \n It should work for nsamples=1024, samplerate=1, exponent=[-7.5, 0]')
    xi = [-7.5, -7.24482759, -6.98965517, -6.73448276, -6.47931034,
          -6.22413793, -5.96896552, -5.7137931, -5.45862069, -5.20344828,
          -4.94827586, -4.69310345, -4.43793103, -4.18275862, -3.92758621,
          -3.67241379, -3.41724138, -3.16206897, -2.90689655, -2.65172414,
          -2.39655172, -2.14137931, -1.8862069, -1.63103448, -1.37586207,
          -1.12068966, -0.86551724, -0.61034483, -0.35517241, -0.1]
    xo = [-13.20247402982382,
          -13.063170624394223,
          -12.852061722924377,
          -12.421652771499275,
          -11.899878602381872,
          -11.424663757138987,
          -10.978062348717065,
          -10.377409187875005,
          -9.897140931980784,
          -9.467254228316564,
          -8.858265356432794,
          -8.229197868702832,
          -7.784149181687997,
          -7.3709346625639505,
          -6.8789520539171685,
          -6.359913480896892,
          -5.875116275624759,
          -5.282040594199156,
          -4.875876629088813,
          -4.252993866292538,
          -3.7986857552691404,
          -3.3289036673593535,
          -2.7531143296776937,
          -2.282607200597624,
          -1.6773103401587297,
          -1.3106515274523698,
          -0.8380738853233445,
          -0.4410159650357907,
          -0.16807885045855095,
          -0.03076051153313276]
    g = interpolate.interp1d(xo, xi)
    try:
        exponent_adjusted = g(exponent)
    except ValueError:
        exponent_adjusted = exponent
        print('Warning: the given exponent may not work. Confirm the power spectrum.')

    freqX = np.abs(np.fft.fftfreq(nsamples, 1 / samplerate))
    freqY = np.abs(np.fft.fftfreq(nsamples, 1 / samplerate))
    freqXX, freqYY = np.meshgrid(freqX, freqY)
    freqs = np.sqrt(freqXX ** 2 + freqYY ** 2)

    if min_freq is None and max_freq is None:
        f = np.ones((nsamples, nsamples))
        f *= freqs ** (exponent_adjusted)
        f[np.where(freqs == 0)] = 0
    #         print(np.isinf(freqs))
    #         f[np.logical_or(np.where(freqs==0), np.isinf(freqs))] = 0
    else:
        f = np.zeros(nsamples)
        if min_freq is not None and max_freq is not None:
            keep = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
        elif min_freq is not None and max_freq is None:
            keep = freqs >= min_freq
        elif min_freq is None and max_freq is not None:
            keep = freqs <= max_freq

        f[keep] = 1
        f *= freqs ** exponent
        f[np.where(freqs == 0)] = 0
    #         f[np.logical_or(np.where(freqs==0), np.isinf(freqs))]=0 # spectral rep of noise
    return fftnoise2d(f)


def generate_sample_field(L, n=201, exponent=-5. / 3., mag=1e4, return_xy=True):
    """
    Returns a square v-field which has the energy spectrum with a given exponent
    ... 1. It prepares a Fourier Transform of the resulting velocitiy field
        2. Add random phase to the array
        3. Inverse Fourier Transform the prepared field
        4. Multilply "mag"
    ... Known issues: this algorithm fails for exponents outside [-7.5, -0.1]

    Parameters
    ----------
    L: float, size of a field
    n: int, length of the array
    exponent: float, [-7.5, -0.1] seems to work.
    mag: float, this factor is multiplied to the resulting v-field in the end
    return_xy: bool, if True, returns x and y arrays

    Returns
    -------
    udata: 2d array with shape (n, n)
    xx: 2d array with shape (n, n), optional, x-coordinates of the array
    yy: 2d array with shape (n, n), optional, y-coordinates of the array
    """
    x, y = np.linspace(-L, L, n), np.linspace(-L, L, n)
    xx, yy = np.meshgrid(x, y)
    dx, dy = get_grid_spacing(xx, yy)
    freqs = np.fft.fftfreq(n, dx)
    # fmin, fmax = np.min(freqs[int((n + 1) / 2):]), np.max(freqs)

    ux = band_limited_noise2d(nsamples=n, samplerate=n, exponent=exponent)
    uy = band_limited_noise2d(nsamples=n, samplerate=n, exponent=exponent)

    ux, uy = ux - np.nanmean(ux), uy - np.nanmean(uy)
    udata = np.empty((2,) + xx.shape)
    udata[0, :, :] = ux * mag
    udata[1, :, :] = uy * mag
    if return_xy:
        return udata, xx, yy
    else:
        return udata


########## turbulence related stuff  ##########
def get_rescaled_energy_spectrum_saddoughi():
    """
    Returns values to plot rescaled energy spectrum from Saddoughi (1992)

    Parameters
    ----------

    Returns
    -------
    e: numpy array
        spectral energy density: energy stored between [k, k+dk)
    k: numpy array
        wavenumber
    """
    k = np.asarray(
        [1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    e = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])
    return e, k


def get_energy_spectra_jhtd():
    """
    Returns values to plot energy spectrum from JHTD, computed by Takumi in 2019
    Call get_rescaled_energy_spectra_jhtd for scaled energy spectra.

    Parameters
    ----------

    Returns
    -------
    datadict: dict
        data stored is stored in '/reference_data/jhtd_e_specs.h5'
    """
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    datapath = faqm_dir + '/reference_data/jhtd_e_specs.h5'

    datadict = {}
    with h5py.File(datapath, 'r') as data:
        keys = list(data.keys())
        for key in keys:
            if not '_s' in key:
                datadict[key] = data[key][...]

    return datadict


def get_rescaled_energy_spectra_jhtd():
    """
    Returns values to plot rescaled energy spectrum from JHTD, computed by Takumi in 2019
    Call get_energy_spectra_jhtd for dimensionful energy spectra.

    Returns
    -------
    datadict: dict
        data stored in jhtd_e_specs.h5 is stored: Scaled k, Scaled ek
    """
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    datapath = faqm_dir + '/reference_data/jhtd_e_specs.h5'

    datadict = {}
    with h5py.File(datapath, 'r') as data:
        keys = list(data.keys())
        for key in keys:
            if '_s' in key:
                datadict[key] = data[key][...]
    return datadict


def get_rescaled_structure_function_saddoughi(p=2):
    """
    Returns the values of rescaled structure function reported in Saddoughi and Veeravalli 1994 paper: r_scaled, dll
    ... this is a curve about a specific Reynolds number! i.e. there is no universal structure function

    ----------
    p: int
    ... order of the structure function

    Returns
    -------
    r_scaled: nd array, distance with respect to Kolmogorov length scale
    dll: nd array, rescaled structure function
    """
    tflow_dir = os.path.split(os.path.realpath(__file__))[0]
    if p == 2:
        datapath = tflow_dir + '/reference_data/sv_struc_func.h5'
        # datapath = tflow_dir + '/velocity_ref/sv_struc_func.txt'
        # data = np.loadtxt(datapath, skiprows=1, delimiter=',')
        # r_scaled, dll = data[:, 0], data[:, 1]
        with h5py.File(datapath, 'r') as ff:
            r_scaled = np.asarray(ff['r_s'])
            dll = np.asarray(ff['dll'])
        return r_scaled, dll
    else:
        print('... Only the rescaled, second-order structure function is available at the moment!')
        return None, None


######### Treating a large udata #######
# As a design, udata is expected be < 12GB so that you can load an entire data on the RAM.
# It might sound horrible but is advantageous to do temporal averaging quickly.
# If udata is large compared to your RAM, one should do computation every frame.
# i.e.- load a snapshot of a 2D/3D field onto a RAM, compute, move onto the next frame, repeat.
# To do so, use the following helper.

def process_large_udata(udatapath, func=get_spatial_avg_energy, t0=0, t1=None,
                        x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, inc=10, notebook=True,
                        reynolds_decomposition=False,
                        clean=False, cutoff=np.inf, method='nn',
                        median_filter=False, replace_zeros=True, overwrite_udatam=False, **kwargs):
    """
    (Intended for 3D velocity field data)
    Given a path to a hdf5 file which stores udata,
    it returns the result of functions without loading an entire udata onto RAM
    ... example:
        results = process_large_udata(udatapath, func=get_spatial_avg_enstrophy,
                                              inc=inc, dx=dx, dy=dy, dz=dz)
        enst, enst_err = result

    Parameters
    ----------
    udatapath: str, path to a h5 file which stores udata
    func: function
        ... a function to compute a quantity you desire from udata
    t0: int, index of the first frame to be processed
    t1: int, default: None
        ... If t1 were not given, it processes from the beginning to the end.
    inc: int, default: 10
        ... temporal increment at which the function is called
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    inc: int, use every "inc" frames, default: 10
        ... inc = 1 means a velocity field at every frame is used for processing.
    notebook: bool, default: True
        ... if True, it will use tqdm_notebook instead of tqdm to show progress
    clean: bool, default: False
        ... if True, it will clean the v-field data (udata) before processing
        ... Cleaning involves removing spurious vectors by thresholding and median-filtering.
    cutoff: float, default: np.inf
        ... if clean=True, it will remove vectors with magnitude smaller than cutoff
    median_filter: bool, default: False
        ... if True and clean=True, it will apply a median filter to the v-field data.
    replace_zeros: bool, default: True
        ... if True and clean=True, it will replace zeros with the nearest non-zero value.
        ... This is sometimes necessary as zero values are default values instead of nan values for some softwares
            (DaVis- STB).
    kwargs: dict, default: {}, additional keyword arguments to be passed to func()

    Returns
    -------
    datalist: list
        ... outputs of the function
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if t1 is None:
        f = h5py.File(udatapath)
        t1 = f['ux'].shape[-1]
        f.close()
    # number elements saved in the temporal axis
    n = len(range(t0, t1, inc))

    if reynolds_decomposition:
        udata_m = read_data_from_h5(udatapath, ['udata_m'])[0]
        if udata_m is None:
            udata_m = get_mean_flow_field_using_udatapath(udatapath, t0=t0, t1=t1,
                                                          x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, verbose=False,
                                                          clean=clean, cutoff=cutoff, method=method,
                                                          median_filter=median_filter, replace_zeros=replace_zeros)
            add_data2udatapath(udatapath, {'udata_m': udata_m}, overwrite=overwrite_udatam)

    for i, t in enumerate(tqdm(range(t0, t1, inc), desc=func.__name__)):
        # load a dummy data
        try:
            udata, xx, yy, zz = get_udata_from_path(udatapath, return_xy=True, t0=t, t1=t + 1,
                                                    x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, verbose=False)
        except:
            udata, xx, yy = get_udata_from_path(udatapath, return_xy=True, t0=t, t1=t + 1,
                                                x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, verbose=False)
        if clean:
            udata = clean_udata(udata, cutoff=cutoff, verbose=False, median_filter=median_filter,
                                replace_zeros=replace_zeros, showtqdm=False)
        if reynolds_decomposition:
            udata[..., 0] -= udata_m
        if i == 0:
            datalist = []
            result = func(udata, **kwargs)
            if type(result) is tuple:
                n_output = len(result)  # number of objects returned by a function
                for j in range(n_output):
                    shape = list(result[j].shape)
                    if shape[-1] == 1:
                        shape[-1] *= n
                    # initialize a list which will be returned
                    datalist.append(np.empty(shape))
            else:
                n_output = 1
                shape = list(result.shape)
                if shape[-1] == 1:
                    shape[-1] *= n
                # initialize a list which will be returned
                datalist.append(np.empty(shape))
                result = np.asarray([result])
        else:
            result = func(udata, **kwargs)  # compute stuff at t=t
            if n_output == 1:
                result = np.asarray([result])

        # insert the result to the datalist
        for j in range(n_output):
            if datalist[j].shape[-1] == n:
                datalist[j][..., i] = result[j][..., 0]
            else:
                datalist[j] = result[j]
    if notebook:
        from tqdm import tqdm as tqdm

    if len(datalist) == 1: datalist = datalist[0]
    return datalist


def get_time_avg_energy_from_udatapath(udatapath,
                                       x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                       t0=0, t1=None, slicez=None, inc=1,
                                       clean=False, median_filter=False,
                                       thd=np.inf, fill_value=np.nan,
                                       udata_mean=None,
                                       notebook=True, **kwargs):
    """
    Returns time-averaged energy when a path to udata is provided
    ... recommended to use if udata is large (> half of your memory size)
    ... only a snapshot of data exists on memory while time-averaging is performed

    Parameters
    ----------
    udatapath: str
        ... path to udata
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    slicez: int, default: None
        ... Option to return a 2D time-averaged field at z=slicez from 4D udata
        ... This is mostly for the sake of fast turnout of analysis
    inc: int, default: 1
        ... time increment to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    thd: float, default: np.inf
        ... energy > thd will be replaced by fill_value. (Manual screening of data)
    fill_value: float, default: np.nan
        ... value used to fill the data when data value is greater than a threshold
    udata_mean: array, default: None
        ... It subtracts the given array from the udata. This feature can be used to compute the mean fluctuating energy.
    clean: bool, if True, it runs clean_udata() before computing energy
        ... it is often recommmended to pass an additional kwarg 'showtqdm=False' for concise output.
        ... if method='idw', this process could take time so 'showtqdm=True' makes sense to monitor this cleaning process.
    Returns
    -------
    eavg: nd array, time-averaged energy
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    with h5py.File(udatapath, mode='r') as f:
        try:
            height, width, depth, duration = f['ux'].shape
            height, width, depth = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
            dim = 3
        except:
            height, width, duration = f['ux'].shape
            height, width = f['ux'][y0:y1, x0:x1, 0].shape
            dim = 2
        if t1 is None:
            t1 = duration

        if dim == 3:
            if slicez is None:
                eavg = np.zeros((height, width, depth))
                counters = np.zeros((height, width, depth))
                for t in tqdm(range(t0, t1, inc)):
                    udata = np.stack((f['ux'][y0:y1, x0:x1, z0:z1, t],
                                      f['uy'][y0:y1, x0:x1, z0:z1, t],
                                      f['uz'][y0:y1, x0:x1, z0:z1, t]))
                    if udata_mean is not None:
                        udata -= udata_mean
                    if clean:
                        udata = clean_udata(udata, median_filter=median_filter, **kwargs)
                    energy_inst = get_energy(udata)
                    energy_inst[energy_inst > thd] = fill_value
                    eavg = np.nansum(np.stack((eavg, energy_inst)), 0)
                    counters += ~np.isnan(energy_inst)
                eavg /= counters
            else:
                eavg = np.zeros((height, width))
                counters = np.zeros((height, width))
                for t in tqdm(range(t0, t1, inc)):
                    udata = np.stack((f['ux'][y0:y1, x0:x1, slicez - z0, t],
                                      f['uy'][y0:y1, x0:x1, slicez - z0, t],
                                      f['uz'][y0:y1, x0:x1, slicez - z0, t]))
                    if udata_mean is not None:
                        udata -= udata_mean
                    if clean:
                        udata = clean_udata(udata, median_filter=median_filter, **kwargs)
                    energy_inst = get_energy(udata)
                    energy_inst[energy_inst > thd] = fill_value
                    eavg = np.nansum(np.stack((eavg, energy_inst)), 0)
                    counters += ~np.isnan(energy_inst)
                eavg /= counters
        elif dim == 2:
            eavg = np.zeros((height, width))
            counters = np.zeros((height, width))
            for t in tqdm(range(t0, t1, inc)):
                udata = np.stack((f['ux'][y0:y1, x0:x1, t],
                                  f['uy'][y0:y1, x0:x1, t]))
                if udata_mean is not None:
                    udata -= udata_mean
                if clean:
                    udata = clean_udata(udata, median_filter=median_filter, **kwargs)[..., 0]
                energy_inst = get_energy(udata)
                energy_inst[energy_inst > thd] = fill_value
                eavg = np.nansum(np.stack((eavg, energy_inst)), 0)
                counters += ~np.isnan(energy_inst)
            eavg /= counters
    if notebook:
        from tqdm import tqdm
    return eavg


def count_nans_along_axis(udatapath, axis='z',
                          x0=0, x1=None,
                          y0=0, y1=None,
                          z0=0, z1=None,
                          inc=100):
    """
    Count the number of nans in udata along an axis

    Parameters
    ----------
    udatapath: str, path to udata
    axis: str, 'x', 'y', 'z', an axis along which the number of nans gets calculated
    inc: int, time increment to count the number of nans in a udata
        ... if one wants complete statistics, use inc=1 but this is overkill.

    Returns
    -------
    nans: 1d array, (number of nans in udata) / udata.size along the specified axis
    """
    shape = get_udata_dim(udatapath)
    dim = len(shape) - 1
    nnan_list = []

    if x1 is None:
        x1 = shape[1]
    if y1 is None:
        y1 = shape[0]
    if z1 is None:
        z1 = shape[2]
    if x1 <= 0:
        x1 = shape[1] + x1
    if y1 <= 0:
        y1 = shape[0] + y1
    if z1 <= 0:
        z1 = shape[2] + z1

    if dim == 2:
        print('... 2D udata. Returns the ratio of the no. of nans to the column length along %s' % axis)
    else:
        print(
            '... 3D udata. Returns the ratio of the no. of nans to the number of elements on the plane along %s' % axis)

    if axis == 'z':
        for z0_ in range(z0, z1):
            if z0_ == shape[-2]:
                z1_ = None
            else:
                z1_ = z0_ + 1
            udata = get_udata_from_path(udatapath, z0=z0_, z1=z1_, inc=inc, verbose=False,
                                        x0=x0, x1=x1, y0=y0, y1=y1)
            nnan = np.nansum(np.isnan(udata)) / float(udata.size)
            nnan_list.append(nnan)
    elif axis == 'x':
        for x0_ in range(x0, x1):
            if x0_ == shape[1]:
                x1_ = None
            else:
                x1_ = x0_ + 1
            udata = get_udata_from_path(udatapath, x0=x0_, x1=x1_, inc=inc, verbose=False,
                                        y0=y0, y1=y1, z0=z0, z1=z1)
            nnan = np.nansum(np.isnan(udata)) / float(udata.size)
            nnan_list.append(nnan)
    elif axis == 'y':
        for y0_ in range(y0, y1):
            if y0_ == shape[-2]:
                y1_ = None
            else:
                y1_ = y0_ + 1
            udata = get_udata_from_path(udatapath, y0=y0_, y1=y1_, inc=inc, verbose=False,
                                        x0=x0, x1=x1, z0=z0, z1=z1)
            nnan = np.nansum(np.isnan(udata)) / float(udata.size)
            nnan_list.append(nnan)
    return np.asarray(nnan_list)


def get_time_avg_enstrophy_from_udatapath(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                          t0=0, t1=None, slicez=None, inc=1, thd=np.inf, fill_value=np.nan,
                                          udata_mean=None,
                                          notebook=True):
    """
    Returns time-averaged energy from a path to udata
    ... recommended to use if udata is large (> half of your memory size)
    ... only a snapshot of data exists on memory while time-averaging is performed

    Parameters
    ----------
    udatapath: str
        ... path to udata
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    slicez: int, default: None
        ... Option to return a 2D time-averaged field at z=slicez from 4D udata
        ... This is mostly for the sake of fast turnout of analysis
    inc: int, default: 1
        ... time increment to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    thd: float, default: np.inf
        ... energy > thd will be replaced by fill_value. (Manual screening of data)
    fill_value: float, default: np.nan
        ... value used to fill the data when data value is greater than a threshold

    Returns
    -------
    enstavg: ndarray, time-averaged enstrophy
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    with h5py.File(udatapath) as f:
        try:
            duration = f['ux'].shape[-1]
            height, width, depth = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
            dim = 3
        except:
            duration = f['ux'].shape[-1]
            height, width = f['ux'][y0:y1, x0:x1, 0].shape
            dim = 2
        if t1 is None:
            t1 = duration

        if dim == 3:
            if slicez is None:
                enstavg = np.zeros((height, width, depth))
                counters = np.zeros((height, width, depth))
                for t in tqdm(range(t0, t1, inc)):
                    udata, xx, yy, zz = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                                            t0=t, t1=t + 1, return_xy=True, verbose=False)
                    dx, dy, dz = get_grid_spacing(xx, yy, zz)
                    enst_inst = get_enstrophy(udata, dx, dy, dz, xx=xx, yy=yy, zz=zz)[..., 0]
                    enst_inst[enst_inst > thd] = fill_value
                    enstavg = np.nansum(np.stack((enstavg, enst_inst)), 0)
                    counters += ~np.isnan(enst_inst)
                enstavg /= counters
            else:
                enstavg = np.zeros((height, width))
                counters = np.zeros((height, width))
                for t in tqdm(range(t0, t1, inc)):
                    udata, xx, yy, zz = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=slicez - 1,
                                                            z1=slicez + 2,
                                                            t0=t, t1=t + 1, return_xy=True, verbose=False)
                    dx, dy, dz = get_grid_spacing(xx, yy, zz)
                    enst_inst = get_enstrophy(udata, dx, dy, dz, xx=xx, yy=yy, zz=z)[:, :, 1, 0]
                    enst_inst[enst_inst > thd] = fill_value
                    enstavg = np.nansum(np.stack((enstavg, enst_inst)), 0)
                    counters += ~np.isnan(enst_inst)
                enstavg /= counters
        elif dim == 2:
            enstavg = np.zeros((height, width))
            counters = np.zeros((height, width))
            for t in tqdm(range(t0, t1, inc)):
                udata, xx, yy = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, t0=t, t1=t + 1,
                                                    return_xy=True, verbose=False)
                dx, dy = get_grid_spacing(xx, yy)
                enst_inst = get_enstrophy(udata, dx, dy, xx=xx, yy=yy)[:, :, 0]
                enst_inst[enst_inst > thd] = fill_value
                enstavg = np.nansum(np.stack((enstavg, enst_inst)), 0)
                counters += ~np.isnan(enst_inst)
            enstavg /= counters
    if notebook:
        from tqdm import tqdm
    return enstavg


def get_time_avg_field_from_udatapath(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                      t0=0, t1=None, slicez=None, inc=1, thd=np.inf, fill_value=np.nan,
                                      notebook=True):
    """
    Returns time-averaged velocity field when a path to udata is provided
    ... recommended to use if udata is large (> half of your memory size)
    ... only a snapshot of data exists on memory while time-averaging is performed

    Parameters
    ----------
    udatapath: str
        ... path to udata
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    slicez: int, default: None
        ... Option to return a 2D time-averaged field at z=slicez from 4D udata
        ... This is mostly for the sake of fast turnout of analysis
    inc: int, default: 1
        ... time increment to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    thd: float, default: np.inf
        ... energy > thd will be replaced by fill_value. (Manual screening of data)
    fill_value: float, default: np.nan
        ... value used to fill the data when data value is greater than a threshold

    Returns
    -------
    u_m: ndarray, mean velocity field with shape (dim, height, width, (depth))
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    with h5py.File(udatapath) as f:
        try:
            height, width, depth, duration = f['ux'].shape
            height, width, depth = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
            dim = 3
        except:
            height, width, duration = f['ux'].shape
            height, width = f['ux'][y0:y1, x0:x1, 0].shape
            dim = 2
        if t1 is None:
            t1 = duration

        if dim == 3:
            if slicez is None:
                ux_m, uy_m, uz_m = np.zeros((height, width, depth)), np.zeros((height, width, depth)), np.zeros(
                    (height, width, depth))
                counters_ux, counters_uy, counters_uz = np.zeros((height, width, depth)), np.zeros(
                    (height, width, depth)), np.zeros((height, width, depth))
                for t in tqdm(range(t0, t1, inc)):
                    ux_inst, uy_inst, uz_inst = f['ux'][y0:y1, x0:x1, z0:z1, t], f['uy'][y0:y1, x0:x1, z0:z1, t], f['uz'][y0:y1, x0:x1, z0:z1, t]
                    ux_inst[ux_inst > thd] = fill_value
                    uy_inst[uy_inst > thd] = fill_value
                    uz_inst[uz_inst > thd] = fill_value

                    ux_m = np.nansum(np.stack((ux_m, ux_inst)), 0)
                    uy_m = np.nansum(np.stack((uy_m, uy_inst)), 0)
                    uz_m = np.nansum(np.stack((uz_m, uz_inst)), 0)
                    counters_ux += ~np.isnan(ux_inst)
                    counters_uy += ~np.isnan(uy_inst)
                    counters_uz += ~np.isnan(uz_inst)
                ux_m /= counters_ux
                uy_m /= counters_uy
                uz_m /= counters_uz
            else:
                ux_m, uy_m, uz_m = np.zeros((height, width, depth)), np.zeros((height, width, depth)), np.zeros(
                    (height, width, depth))
                counters_ux, counters_uy, counters_uz = np.zeros((height, width, depth)), np.zeros(
                    (height, width, depth)), np.zeros((height, width, depth))
                for t in tqdm(range(t0, t1, inc)):
                    ux_inst, uy_inst, uz_inst = f['ux'][y0:y1, x0:x1, slicez - z0, t], f['uy'][y0:y1, x0:x1,
                                                                                       slicez - z0, t], f['uz'][y0:y1,
                                                                                                        x0:x1,
                                                                                                        slicez - z0, t]
                    ux_inst[ux_inst > thd] = fill_value
                    uy_inst[uy_inst > thd] = fill_value
                    uz_inst[uz_inst > thd] = fill_value
                    ux_m = np.nansum(np.stack((ux_m, ux_inst)), 0)
                    uy_m = np.nansum(np.stack((uy_m, uy_inst)), 0)
                    uz_m = np.nansum(np.stack((uz_m, uz_inst)), 0)
                    counters_ux += ~np.isnan(ux_inst)
                    counters_uy += ~np.isnan(uy_inst)
                    counters_uz += ~np.isnan(uz_inst)
                ux_m /= counters_ux
                uy_m /= counters_uy
                uz_m /= counters_uz
            u_m = np.stack((ux_m, uy_m, uz_m))
        elif dim == 2:
            ux_m, uy_m = np.zeros((height, width)), np.zeros((height, width))
            counters_ux, counters_uy = np.zeros((height, width)), np.zeros((height, width))
            for t in tqdm(range(t0, t1, inc)):
                ux_inst, uy_inst = f['ux'][y0:y1, x0:x1, t], f['uy'][y0:y1, x0:x1, t]
                ux_inst[ux_inst > thd] = fill_value
                uy_inst[uy_inst > thd] = fill_value

                ux_m = np.nansum(np.stack((ux_m, ux_inst)), 0)
                uy_m = np.nansum(np.stack((uy_m, uy_inst)), 0)
                counters_ux += ~np.isnan(ux_inst)
                counters_uy += ~np.isnan(uy_inst)
            ux_m /= counters_ux
            uy_m /= counters_uy
            u_m = np.stack((ux_m, uy_m))
    if notebook:
        from tqdm import tqdm
    return u_m


def export_raw_file_from_dpath(udatapath, func=get_energy, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                               t0=0, t1=None, inc=1, savedir=None, dtype='uint32', thd=None,
                               interpolate=None, notebook=True, running_avg=1, **kwargs):
    """
    Exports a raw file of the output of a given function (default: get_energuy) at each time step.
    (Intended for 4D visualization. e.g. use ORS Dragonfly for visualiztion)

    ... Argument: path to udata
    ... Philosophy: Load 3D data at an instant of time, compute the quantity of interest. Export the 3D array. Repeat.

    Parameters
    ----------
    udatapath: str, a path to the udata file
    func: function, a function that takes in a 3D array and returns a 3D array
       ... default: get_energy()- returns the instantaneous energy field of the flow
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t1: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    inc: int, default: 1
        ... exports the raw data every "inc"-th time step
    savedir: str, default: None
        ... a path of the directory where raw data will be saved
    dtype: str, default: 'uint32'
        ... data type of the raw data
    thd: float, default: None
        ... values above this threshold are set to 0.
    interpolate: str, default None
        ... choose from 'idw' and 'localmean'
        ...... Inverse distance weighting (IDW) - gaussian mean
        ...... localmean -  assigns mean of the surrounding values at the missing points
    notebook: bool, default: True,
        .... if True, use tqdm_notebook instead of tqdm to show a progress bar
    running_avg: int, default: 1
        ... if running_avg>1, it exporsts the running average of the computed quantity
    kwargs: additional arguments for func

    Returns
    -------
    None
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    udatadir, udaname = os.path.split(udatapath)
    pdir = os.path.split(udatadir)[0]
    if savedir is None:
        savedir = os.path.join(pdir, 'raw_files')
        savedir = os.path.join(os.path.join(savedir, udaname[:-3]), func.__name__)

    dummy = get_udata_from_path(udatapath, return_xy=False,
                                x0=x0, x1=x1, y0=y0, y1=y1,
                                z0=z0, z1=z1,
                                t0=0, t1=1, verbose=False)
    shape = dummy.shape[1:]
    duration = get_udata_dim(udatapath)[-1]
    savedir += '_%03dx%03dx%03dx%05dx_inc%d' % (shape[0], shape[1], shape[2], shape[3], inc)
    if interpolate is not None:
        savedir += '_%s' % interpolate

    if t1 is None:
        t1 = duration

    for i, t in enumerate(tqdm(range(t0, t1, inc), desc='saving raw files')):
        udata, xx, yy, zz = get_udata_from_path(udatapath, return_xy=True,
                                                x0=x0, x1=x1, y0=y0, y1=y1,
                                                z0=z0, z1=z1,
                                                t0=t, t1=t + running_avg, verbose=True)
        dx, dy, dz = get_grid_spacing(xx, yy, zz)

        # Save a scalar field derived from
        if running_avg == 1:
            # Perform 3D interpolation
            if interpolate is not None:
                udata = clean_udata(udata, verbose=False, method=interpolate)
            data2save = func(udata, **kwargs)[..., 0]  # save 3d array
        else:
            data2save = func(udata, **kwargs)[...]  # save 3d array
            if interpolate is not None:
                data2save = replace_nans(data2save, method=interpolate, showtqdm=False,
                                         max_iter=10, tol=0.05, kernel_radius=2, kernel_sigma=2)

        if thd is not None:
            keep = data2save < thd
            data2save[~keep] = 0

        data2save = data2save

        savepath = os.path.join(savedir, 't%05d.raw' % t)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        data2save.astype(dtype).tofile(savepath)

    if notebook:
        from tqdm import tqdm


def export_raw_file(data2save, savepath, dtype='uint32', thd=np.inf, interpolate=None, fill_value=np.nan,
                    contrast=True, contrast_value=None, log10=False, notebook=True, **kwargs):
    """
    Exports a raw file from a numpy array with inteprolation features
    ... The intended use is to convert the numpy array to a raw file which can be loaded to Dragonfly
    ... np.nan in the given array could be filled by the following interpolating methods:
        'nn': fills nans with the nearest neighbors (fast)
        'localmean':  fills nans with the local mean (kernel size of 3 by default
            - the kernel size can be changed by passing an extra kwargs to the method called replace_nan_w_nn) (slow)
        'idw':  fills nans with the (inverse distancing weighting) Gaussian kernel
            - the kernel size can be changed by passing an extra kwargs to the method called replace_nan_w_nn) (slow)

    
    Parameters
    ----------
    data2save: nd array, shape (Ny, Nx, Nz, Nt)
    savepath: str, path where the raw file is saved
    dtype: str, default: 'uint32', data type of the raw file
    thd: float default: np.inf
        ... data > thd will be replaced by fill_value
    interpolate: str, default None
        ... choose from 'idw' and 'localmean', 'nn
            ... if specified, it replaces np.nan in array using replace_nan(...)
    fill_value: float, default: np.nan
        ... value that is used to fill in data where data value > thd
    kwargs: passed to replace_nan(...)

    Returns
    -------
    None

    """
    shape = np.asarray(data2save).shape

    if savepath[-4:] == '.raw':
        savepath = savepath[:-4]
    for i in shape:
        savepath += '_%03dx' % i

    data2save[np.isinf(data2save)] = fill_value
    if thd is not None:
        fill = data2save > thd
        data2save[fill] = fill_value  # this fills values. np.nan won't get filled

    if interpolate == 'nn':
        data2save = replace_nan_w_nn(data2save, notebook=notebook)
    elif interpolate in ['localmean', 'idw']:
        data2save = replace_nans(data2save, method=interpolate, notebook=notebook, **kwargs)

    if contrast:
        if contrast_value is None:
            try:
                maxint = 2 ** int(dtype[-2:]) - 1
            except:
                maxint = 2 ** int(dtype[-1:]) - 1
            max_value, min_value = np.nanmax(data2save), np.nanmin(data2save)
            contrast_value = maxint / (max_value - min_value)
            data2save = (data2save - min_value) * contrast_value
            print('intensity was enhanced by %.2f' % contrast_value, maxint, max_value - min_value)
        else:
            data2save = data2save * contrast_value
            print('intensity was multiplied by %.2f' % contrast_value)

    else:
        contrast_value = 1
    if log10:
        data2save = np.log10(data2save)

    savedir = os.path.split(savepath)[0]
    savepath += '%s_thd%06.f_intp%s_ctr%06.f_log%r.raw' % (dtype, thd, interpolate, contrast_value, log10)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    data2save.astype(dtype).tofile(savepath)


######### Helper for volumetric data ########
## Generate a 2D slice from a 3D data
def slicer(xxx, yyy, zzz, n, pt, basis=None, spacing=None, notebook=True):
    """
    Returns coordinates on a slice of a volume at a given point

    ... basisA: Cartesian basis of the volume
    ... basisB: npq basis
    
    Parameters
    ----------
    xxx: nd array, shape (Ny, Nx, Nz), x-coordinates of the volume
    yyy: nd array, shape (Ny, Nx, Nz), y-coordinates of the volume
    zzz: nd array, shape (Ny, Nx, Nz), z-coordinates of the volume
    n: tuple/list/1darray, shape (3,), normal vector of the plane (nx, ny, nz)
    pt: tuple/list/1darray, shape (3,), point on the plane (x, y, z)
    basis: 3x3 array, default None, new basis
        ... the new basis vectors are normal vector of the plane (n) and two orthogonal vectors (p, q) on the plane
        ... the column vectors are the new basis vectors: [n, p, q]
        ... If p, and q are specified, theses vectors are used to span the plane.
    spacing: float, spacing between points spanned on the plane
    notebook: bool, default True
        ... if True, the tqdm_notebook is used instead of tqdm to show progress

    Returns
    -------
    xxp: 3d array, shape (Ny, Nx, Nz), x-coordinates of the slice
    yyp: 3d array, shape (Ny, Nx, Nz), y-coordinates of the slice
    zzp: 3d array, shape (Ny, Nx, Nz), z-coordinates of the slice
    pp: 2d array, p-coordinates of the slice
    qq: 2d array, q-coordinates of the slice
    Mab: 3x3 array, change-of-basis matrix from basisA to basisB
        ... np.matmul(Mab, vector_in_basiaA) = vector_in_basisB
    Mba: 3x3 array, change-of-basis matrix from basisB to basisA
        ... np.matmul(Mba, vector_in_basisB) = vector_in_basisA
    basis_npq: 3x3 array, basisB (npq basis)
        ... the column vectors are the new basis vectors: [n, p, q]
        ... If p and q were not specified, this code randomly generates orthogonal basis to span the plane.
    """

    def mag(x, axis=-1):
        return np.sum(np.asarray(x) ** 2, axis=axis) ** 0.5

    def get_change_of_basis_matrix(basisA, basisB):
        """
        Returns a change-of-basis matrix from basis A to basis B
        ... each basis must consist of linearly independent vectors

        e.g.
            basis_A = np.asarray([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])#standard basis
            basis_B = np.asarray([[1,0, 1],
                                  [1,2,4],
                                  [1,-1,-1]])
        """
        a, b = basisA, basisB
        Mae = a  # change of basis from a to a standard basis
        Mbe = b  # change of basis from b to a standard basis
        if np.linalg.det(Mbe) == 0:
            #         print('... change-of-basis matrix is proven to be ALWAYS invertible but the code says it is.')
            print(
                '... You supplied a inappropriate basis for basis B! Probably it is not linearly indepenent. Please check.')
            sys.exit()

        Mab = np.matmul(np.linalg.inv(Mbe),
                        Mae)  # change of basis from a to b, which is equal to the inverse of Mbe since a is a standard basis
        return Mab

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if spacing is None:
        dx, dy, dz = get_grid_spacing(xxx, yyy, zzz)
        spacing = min([dx, dy, dz])
    n = np.asarray(n) / mag(n)  # normal vector of the plane
    ro = np.asarray(pt)  # a point on the plane
    verts = get_intersecting_vertices(xxx, yyy, zzz, n[0], n[1], n[2], pt[0], pt[1], pt[2])
    roi = verts - np.repeat(ro[np.newaxis, :], verts.shape[0],
                            axis=0)  # relative vectors from the reference to the intersecting vertices
    dists = mag(roi)

    # Basis A (Standard basis)
    basis_std = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])  # udata uses a standard basis (y, x, z)

    # Basis B (New orthonormal basis)
    if basis is None:
        e1 = n  # normal vector
        e2 = verts[np.argmax(dists)] - ro
        e2 /= mag(e2)
        e3 = np.cross(e1, e2)
    else:
        e1, e2, e3 = basis[:, 0], basis[:, 1], basis[:, 2]
        e1 /= mag(e1)
        e2 /= mag(e2)
        e3 /= mag(e3)
    basis_npq = np.stack([e1, e2, e3]).T  # New coordinates: (n, p, q)

    # Get a change-of_basis matrix
    Mab = get_change_of_basis_matrix(basis_std, basis_npq)
    Mba = np.linalg.inv(Mab)

    # Let the coordinates in the e2-e3 basis be p and q
    pmin, pmax = np.min(np.dot(roi, e2)), np.max(np.dot(roi, e2))
    qmin, qmax = np.min(np.dot(roi, e3)), np.max(np.dot(roi, e3))
    # nmin, nmax = np.min(np.dot(roi, e1)), np.max(np.dot(roi, e1)) # nmin and nmax should be always zero!

    # Now make a grid with the new basis
    p = np.linspace(pmin, pmax, int((pmax - pmin) // spacing))
    q = np.linspace(qmin, qmax, int((qmax - qmin) // spacing))
    pp, qq = np.meshgrid(p, q)

    # Coordinates of points on the plane in the standard basis
    xxp = ro[0] + pp * e2[0] + qq * e3[0]
    yyp = ro[1] + pp * e2[1] + qq * e3[1]
    zzp = ro[2] + pp * e2[2] + qq * e3[2]

    if notebook:
        from tqdm import tqdm as tqdm

    return xxp, yyp, zzp, pp, qq, Mab, Mba, basis_npq


def get_intersecting_vertices(xxx, yyy, zzz, a, b, c, x0, y0, z0):
    """
    Returns the vertices of intercetion between a plane and a cuboidal volume
    ... the plane is defined by the equation ax + by + cz = a*x0+ b*y0 + c*z0 = d
    ...... Note that the normal of the plane is parallel to (a, b, c)
    ... the cuboidal volume is defined by (xxx, yyy, zzz)
    Parameters
    ----------
    xxx: 3d array, x coordinates of the cuboidal volume
    yyy: 3d array, y coordinates of the cuboidal volume
    zzz: 3d array, z coordinates of the cuboidal volume
    a: float, x-coefficient of the plane
    b: float, y-coefficient of the plane
    c: float, z-coefficient of the plane
    x0: float, x-coordinate of the point on the plane
    y0: float, y-coordinate of the point on the plane
    z0: float, z-coordinate of the point on the plane

    Returns
    -------
    verts: (n, 3) array, coordinates of the vertices of intersection
    """

    def mag(x, axis=-1):
        return np.sum(np.asarray(x) ** 2, axis=axis) ** 0.5

    def insideBox(x, y, z, xmin, xmax, ymin, ymax, zmin, zmax):
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax and z >= zmin and z <= zmax

    d = a * x0 + b * y0 + c * z0

    xmin, xmax, ymin, ymax, zmin, zmax = xxx.min(), xxx.max(), yyy.min(), yyy.max(), zzz.min(), zzz.max()

    vert01 = [xmax, (d - a * xmax - c * zmin) / b, zmin]
    vert02 = [xmax, (d - a * xmax - c * zmax) / b, zmax]
    vert03 = [xmax, ymin, (d - a * xmax - b * ymin) / c]
    vert04 = [xmax, ymax, (d - a * xmax - b * ymax) / c]

    vert05 = [xmin, (d - a * xmin - c * zmin) / b, zmin]
    vert06 = [xmin, (d - a * xmin - c * zmax) / b, zmax]
    vert07 = [xmin, ymin, (d - a * xmin - b * ymin) / c]
    vert08 = [xmin, ymax, (d - a * xmin - b * ymax) / c]

    vert09 = [(d - b * ymax - c * zmin) / a, ymax, zmin]
    vert10 = [(d - b * ymax - c * zmax) / a, ymax, zmax]
    vert11 = [(d - b * ymin - c * zmin) / a, ymin, zmin]
    vert12 = [(d - b * ymin - c * zmax) / a, ymin, zmax]

    verts = [vert01, vert02, vert03, vert04,
             vert05, vert06, vert07, vert08,
             vert09, vert10, vert11, vert12]
    verts = set([tuple(item) for item in verts])
    verts = [vert for vert in verts if insideBox(vert[0], vert[1], vert[2], xmin, xmax, ymin, ymax, zmin, zmax)]
    inds = [i for i, vert in enumerate(verts) if
            insideBox(vert[0], vert[1], vert[2], xmin, xmax, ymin, ymax, zmin, zmax)]

    if verts == []:
        raise ValueError('Given plane does not intersect with the cuboid')
    else:
        verts = np.stack(verts)
        rg = np.nanmean(verts, axis=0)  # centroid will be used to fiugre out the order of the vertices
        # sort vertices by polar angles on the plane
        n = np.asarray([a, b, c])
        e1 = n = n / mag(n)
        e2 = verts[0] - rg  # consider a relative vector to a vertex from the centroid as one of the basis vectors
        e2 /= mag(e2)
        e3 = np.cross(e1, e2)
        ps = np.dot(e2, (verts - rg).T)
        qs = np.dot(e3, (verts - rg).T)
        angles = np.arctan2(qs, ps)

        angles, verts = sort_n_arrays_using_order_of_first_array([angles, verts])
        verts = np.asarray(verts)
        return verts

def slicer_old(xx, yy, zz, n, pt, basis=None, spacing=None, apply_convention=True, show=False, notebook=True,
               debug=False):
    """
    DEPRICATED (this slicer function uses a brute-force approach. Use slicer() instead.

    Samples points on the cross section of a volume defined by 3D grid (xx, yy, zz) in two different bases and transformation matrices
    ... The area vector (normal to the cross section) and a point on the cross section must be supplied.
    ... It returns xs, ys, zs, ps, qs, ns, Mac, Mca, basisC
    (xyz coordinates, coordinates in a different basis, transformation matrices, the new basis)

    Details:
        Cartesian coordinates use a standard basis (e1, e2, e3).
        This is not a natural basis for an arbitrary cross section of a cuboid since it is a 2d field embedded in 3d!
        Hence, a new basis should be created to extract a 2d slice of a volumetric data.
        We denote the coordinates of the pts in the new basis (n, p, q). The first basis vector is a unit area vector of the cross section.
        This basis is orthonormal by construction, and the transformation between the bases is unitary (i.e. length conserving)
        This way, the final outputs are literally the pts on the cross section of a cuboid.

    How it works:
        1. Create an arbitrary orthonormal basis, one of which is a unit area vector given by you
        2. Find points which are intersections of the plane and the cuboid
        3. Using the intersecting points and the constructed basis (BasisB), it figures out how much the basis vectors need to
        span to cover the entire cross section
        4. It samples points within the bounds set by Step 3. (This could take seconds)
        5. The basis vectors (of basisB) are not usually aligned with the direction along the longest direction of the cross section.
        Fix the basis vector directions based on the sampled points. The new basis is called basisC.
        6. Get a change-of-basis matrix from ilpm.vec or tflow.vec between the standard basis (basisA) and basisC
        7. Apply conventions on the basisC.
            (a) The diretion with a LONGER side of the cross section is always called p.
            (b) The direction of increase in p is the same direction as x. (The same goes for q and y)
            (c) Enforce the basisC to be right-handed. n x p  = q
        7. Transform the xyz coordinates into npq coordinates using the matrix obtained in Step 6
        8. Return xyz coordinates and npq coordinates and tranformation matrices (Mac: change-of-basis matrix FROM basisA to basisC)

    NEW (Dec 2022):
        Efficient algorithm:


    e.g.
    1.
        # xx, yy, zz = get_equally_spaced_grid(udata_stb) # if you have 3D udata
        x, y, z = np.arange(50), np.arange(30), np.arange(40)
        xx, yy, zz = np.meshgrid(y, x, z)

        n = [0.1, -0.1, 0.4] # Area vector (x, y, z)
        pt = [0, 0, 10] # Cartesian coordinates of a point which lives on the plane (x, y, z)
        xs, ys, zs, ps, qs, ns, Mac, Mca, basis = slicer(xx, yy, zz, n, pt, show=False)
        plt.scatter(ps, qs)
    2.
        # Get a different point on the cross section in xyz
        pt_a = np.asarray([0, -10, 20]) # xyz coordinate of a point
        pt_c = np.matmul(Mac, pt_a) # npq coordinate of the same point
        new_pt_c = pt_c + basis[:, 1] * 2.5 + basis[:, 2] * -4 # another point on the cross seciton but in basisC
        new_pt_a = np.matmul(Mca, new_pt_c) # xyz coordinate of the new point

    3.
        # Get different points on the cross section in xyz
        pts_a = np.asarray([0, -10, 20], [-2, -11, 26]).T # xyz coordinate of a point- Shape must be (3, m)
        pts_c = np.matmul(Mac, pts_a) # npq coordinate of the same point
        new_pts_c = pts_c * 0.5 # points on the cross seciton but in basisC
        new_pts_a = np.matmul(Mca, new_pts_c) # xyz coordinate of the new points

    Tips:
        If you have Takumi's plotting module (found at takumi.graph or takumi's github, or ask takumi)
        one can easily visualize the cross section by uncommenting the sections in the function AND setting "show=True".

    Parameters
    ----------
    xx: 3d numpy array, x coordinates of udata is assumed
    yy: 3d numpy array, y coordinates of udata is assumed
    zz: 3d numpy array, z coordinates of udata is assumed
    n: 1d array-like, area vector in the standard basis (no need to be normalized)
    pt: 1d array-like, a Cartesian coordinate of a point which lives on the cross section (x, y, z)
    basis: 1d array-like, use a user-specified basis instead of a randomly created orthonormal basis for basisB
        ... I do not recommend passing a basis unless there is a reason such as repeatability.
        This might give you unexpected errors.
    spacing: float, value must be greater than 0. Spatial resolution of the sampled grid. If None, it uses the spacing of xx
    show: bool, If True, it plots sampled points on the cross section as well as a cuboid (in a wire frame) both in 3D and 2D.
        ... You must have Takumi's latest plotting module to work. This is not absolute dependencies for public version.
        Therefore, ask Takumi to how to activate it.
    debug: bool, for developers
        ... If True, it prints the basisB and basisC at the end of the code.

    Returns
    -------
    xs, ys, zs, npq_coords[1, :], npq_coords[2, :], npq_coords[0, :], Mac, Mca, basisC
    ... Cartesian coordinats of points on the cross section defined by its normal vector and a containing point
    ... New coordinates (n, p, q)- for usefulness, the function outputs in order of (p, q, n). n should be always zero.
    You should print out the values of n whenever you are in doubt.
    ... Mac: change-of-basis matrix from basis A (standard basis) to basisC
    ... Mca: change-of-basis matrix from basis C to basis A (standard basis)- inverse of Mac
    ... basisC: new basis for npq coordinates
    """

    def insideBox(x, y, z, xx, yy, zz):
        xmin, xmax = np.nanmin(xx), np.nanmax(xx)
        ymin, ymax = np.nanmin(yy), np.nanmax(yy)
        zmin, zmax = np.nanmin(zz), np.nanmax(zz)
        #             w, h, d = xmax-xmin, ymax-ymin, zmax-zmin
        return x >= xmin and x <= xmax and y >= ymin and y <= ymax and z >= zmin and z <= zmax

    def get_vertices_of_cross_section_of_a_cuboid(xx, yy, zz, n, pt):
        x0, x1 = np.nanmin(xx), np.nanmax(xx)
        y0, y1 = np.nanmin(yy), np.nanmax(yy)
        z0, z1 = np.nanmin(zz), np.nanmax(zz)

        n[n == 0] = 1e-10
        n = vec.norm(n)
        a, b, c = n[0], n[1], n[2]

        fz = lambda x, y: pt[2] - 1 / c * (a * (x - pt[0]) + b * (y - pt[1]))
        fy = lambda x, z: pt[1] - 1 / b * (a * (x - pt[0]) + c * (z - pt[2]))
        fx = lambda y, z: pt[0] - 1 / a * (b * (y - pt[1]) + c * (z - pt[2]))

        verts = []
        for (x, y) in list(itertools.product([x0, x1], [y0, y1])):
            vert = np.asarray([x, y, fz(x, y)])
            verts.append(vert)
        for (y, z) in list(itertools.product([y0, y1], [z0, z1])):
            vert = np.asarray([fx(y, z), y, z])
            verts.append(vert)
        for (z, x) in list(itertools.product([z0, z1], [x0, x1])):
            vert = np.asarray([x, fy(x, z), z])
            verts.append(vert)
        verts = np.asarray(verts).T
        for i in range(12):
            if any(np.abs(verts[:, i]) > 1e6) or not insideBox(verts[0, i], verts[1, i], verts[2, i], xx, yy, zz):
                verts[:, i] = np.nan
        return verts

    def find_rotation_angle_for_new_basis(us, vs, basis, Mab, show=False):
        from scipy.optimize import curve_fit
        keep = np.diff(vs) < 0
        #         ind = np.argmax(vs[:-1][keep])

        inds = np.argwhere(vs[:-1][keep] == np.amax(vs[:-1][keep]))
        ind1, ind2 = inds[0, 0], inds[-1, 0]

        u1, v1 = us[:-1][keep][:ind1], vs[:-1][keep][:ind1]
        u2, v2 = us[:-1][keep][ind2:], vs[:-1][keep][ind2:]
        try:
            popt1, pcov1 = curve_fit(lambda x, a, b: a * x + b, u1, v1)
            popt2, pcov2 = curve_fit(lambda x, a, b: a * x + b, u2, v2)
            theta1, theta2 = np.arctan(popt1[0]), np.arctan(popt2[0])
            theta = np.pi / 2 - theta1
            #             p = vec.norm(np.asarray([0, 1, popt1[0] ]))#in new basis (basisB)- n, u, v
            #             q = vec.norm(np.asarray([0, 1, popt2[0] ]))
            if ind1 > (len(vs[:-1][keep]) - len(inds)) / 2:
                p = vec.norm(np.asarray([0, 1, popt1[0]]))  # in new basis (basisB)- n, u, v
                q = vec.norm(np.asarray([0, 1, -1 / popt1[0]]))
            else:
                p = vec.norm(np.asarray([0, 1, popt2[0]]))  # in new basis (basisB)- n, u, v
                q = vec.norm(np.asarray([0, 1, -1 / popt2[0]]))

            npq_basis_in_basisB = np.asarray([[1, 0, 0], p, q]).T
            Mba = np.linalg.inv(Mab)
            npq_basis = np.matmul(Mba, npq_basis_in_basisB)  # basisC vectors in basisA
        except:
            print('curve_fit failed. theta was set to be zero.')
            #             print('u1, v1', u1, v1)
            #             print('u2, v2', u2, v2)
            theta = 0
            npq_basis = basis  # in basis A

        if show:
            try:
                fig, ax = graph.scatter(us, vs, s=1, fignum=2)  # in basisB
                fig, ax = graph.scatter(us[:-1][keep], vs[:-1][keep], s=1, fignum=2)
                fig, ax = graph.scatter(us[:-1][keep][ind1:ind2], vs[:-1][keep][ind1:ind2],
                                        s=50, fignum=2, zorder=200, marker='x', color='r')
                fig, ax, _, _ = graph.plot_fit_curve(u1, v1, lambda x, a, b: a * x + b, fignum=2, zorder=100)
                fig, ax, _, _ = graph.plot_fit_curve(u2, v2, lambda x, a, b: a * x + b, fignum=2, zorder=100)
                ax.set_aspect('equal')
            except:
                pass

        return theta, npq_basis

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if spacing is None:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        spacing = min([dx, dy, dz])
    n = vec.norm(np.asarray(n))

    # Basis A (Standard basis)
    basis_std = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])  # udata uses a standard basis (y, x, z)
    if basis is None:
        # Basis B vectors in basis A- the first column is equal to the given normal vector
        basisB = vec.get_an_orthonormal_basis(3, v1=n)  # new basis (w, v, u)
    else:
        basisB = basis  # use a user-specified basis instead of a randomly created orthonormal basis

    # change-of-basis matrix between basis A and basis B
    Mab = vec.get_change_of_basis_matrix(basis_std,
                                         basisB)  # change-of-basis-marix from a standard basis to a new basis
    Mba = np.linalg.inv(Mab)  # change-of-babsis matrix from basis b to the basis a

    # coordinate of the point on the place in terms of basis B
    pt_b = np.matmul(Mab, np.asarray(pt))  # pt in the basis b- (n, u, v)

    # change of coordinates: (x, y, z) to (n, u, v)
    coords_in_vector_a = np.swapaxes(np.asarray(list(itertools.product(xx[0, :, 0], yy[:, 0, 0], zz[0, 0, :]))), 0, 1)
    coords_in_vector_b = np.matmul(Mab, coords_in_vector_a)  # in basis b = {n, u, v}

    # Sample points on the plane in the volume (xx, yy, zz)- ROBUST BUT SLOWER
    #     ## METHOD 1: Brutal search using the basis vectors of basis B
    #     ## This is naturally done in basis B instead of the basis A (std basis)
    #     ### Prepration for sampling on the plane
    #     nmin, nmax = int(np.floor(np.nanmin(coords_in_vector_b[0, :]))), int(np.ceil(np.nanmax(coords_in_vector_b[0, :])))
    #     umin, umax = int(np.floor(np.nanmin(coords_in_vector_b[1, :]))), int(np.ceil(np.nanmax(coords_in_vector_b[1, :])))
    #     vmin, vmax = int(np.floor(np.nanmin(coords_in_vector_b[2, :]))), int(np.ceil(np.nanmax(coords_in_vector_b[2, :])))
    #     ulist = list(range(umin, umax))
    #     vlist = list(range(vmin, vmax))
    #     uvlist = list(itertools.product(ulist, vlist))

    #   METHOD2- Find the
    verts_a = get_vertices_of_cross_section_of_a_cuboid(xx, yy, zz, n, pt)
    verts_b = np.matmul(Mab, verts_a)
    nmin, nmax = int(np.floor(np.nanmin(verts_b[0, :]))), int(np.ceil(np.nanmax(verts_b[0, :])))
    umin, umax = int(np.floor(np.nanmin(verts_b[1, :]))), int(np.ceil(np.nanmax(verts_b[1, :])))
    vmin, vmax = int(np.floor(np.nanmin(verts_b[2, :]))), int(np.ceil(np.nanmax(verts_b[2, :])))

    ulist = list(range(umin, umax))
    vlist = list(range(vmin, vmax))

    ### Initialization for sampling pts on the plane
    xs, ys, zs = [], [], []
    us, vs, ws = [], [], []  # coordinates in the new basis (coefficients)

    # Sample pts on the plane
    for (counter_u, counter_v) in tqdm(list(itertools.product(ulist, vlist))):

        pt_a_tmp = pt + basisB[:, 1] * counter_u * spacing + basisB[:, 2] * counter_v * spacing
        pt_b_tmp = np.matmul(Mab, pt_a_tmp)

        if insideBox(pt_a_tmp[0], pt_a_tmp[1], pt_a_tmp[2], xx, yy, zz):
            #             pt_b_tmp = np.matmul(Mab, pt_a_tmp) # pt in the basis b- (u,v, w)
            us.append(pt_b_tmp[1])
            vs.append(pt_b_tmp[2])
            ws.append(pt_b_tmp[0])

            # in stadard basis
            xs.append(pt_a_tmp[0])
            ys.append(pt_a_tmp[1])
            zs.append(pt_a_tmp[2])

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)
    us = np.asarray(us)
    vs = np.asarray(vs)
    ws = np.asarray(ws)

    Mab = vec.get_change_of_basis_matrix(basis_std, basisB)
    theta, basisC = find_rotation_angle_for_new_basis(us, vs, basisB, Mab, show=show)

    if show:
        # plots in basisB
        fig, ax = graph.arrow(0, 0, np.matmul(Mab, basisB[:, 1])[1] * 20, np.matmul(Mab, basisB[:, 1])[2] * 20,
                              color='c', fignum=2, width=1)  # this should be 1,0 and 0, 1 in basis B
        graph.arrow(0, 0, np.matmul(Mab, basisB[:, 2])[1] * 20, np.matmul(Mab, basisB[:, 2])[2] * 20,
                    color='m', fignum=2, width=1)  # in basisA

        graph.arrow(0, 0, np.matmul(Mab, basisC[:, 1])[1] * 20, np.matmul(Mab, basisC[:, 1])[2] * 20,
                    color='b', fignum=2, width=1)
        graph.arrow(0, 0, np.matmul(Mab, basisC[:, 2])[1] * 20, np.matmul(Mab, basisC[:, 2])[2] * 20,
                    color='g', fignum=2, width=1)
        graph.title(ax, 'find_rotation_angle_for_a_new_basis')
        graph.labelaxes(ax, 'u', 'v')

    Mbc = vec.get_change_of_basis_matrix(basisB, basisC)
    Mac = vec.get_change_of_basis_matrix(basis_std, basisC)  # this will be updated later due to the convention
    Mca = np.linalg.inv(Mac)

    xyz_coords = np.stack((xs, ys, zs))
    wuv_coords = np.stack((ws, us, vs))

    npq_coords = np.matmul(Mac, xyz_coords)  # (3, n)

    # Convention1: Make the longer side of the plane denoted by "p"
    ps, qs = npq_coords[1, :], npq_coords[2, :]
    w = np.nanmax(ps) - np.nanmin(ps)
    h = np.nanmax(qs) - np.nanmin(qs)

    if apply_convention:
        if w < h:
            print('Lengths in the p-q basis (w, h)', w, h)
            basisC[:, [1, 2]] = basisC[:, [2, 1]]
            basisC[:, 2] *= -1

        # Convention2: Let the directionality of p be consistent with x
        if basisC[0, 1] < 0:
            basisC[:, 1] *= -1

        basisC = vec.apply_right_handedness(basisC)

    # Finally, a natural basis on this plane (basisC) is ready
    ## Get a change-of-basis matrix from basis A to basis C
    Mac = vec.get_change_of_basis_matrix(basis_std, basisC)
    Mca = np.linalg.inv(Mac)
    ## Transform coordinates of pts on the plane (expressed in basis A) into coords in basis C
    npq_coords = np.matmul(Mac, xyz_coords)  # (3, n)

    # In case one needs to check bases used in the process
    if debug:
        print('basisB:')
        print(basisB)
        print('basisC: ')
        print(basisC)
        print('Right-handed basisC (this should be the same as above): ')
        print(vec.apply_right_handedness(basisC))

    if show:
        fig1, ax1 = graph.scatter3d(xs, ys, zs, s=1, fignum=1, subplot=121)
        graph.arrow3D(pt[0], pt[1], pt[2], basisC[0, 0] * 50, basisC[1, 0] * 50, basisC[2, 0] * 50,
                      ax=ax1, zorder=1000, mutation_scale=10)
        graph.arrow3D(pt[0], pt[1], pt[2], basisC[0, 1] * 50, basisC[1, 1] * 50, basisC[2, 1] * 50,
                      ax=ax1, zorder=1000, color='b')
        graph.arrow3D(pt[0], pt[1], pt[2], basisC[0, 2] * 50, basisC[1, 2] * 50, basisC[2, 2] * 50,
                      ax=ax1, zorder=1000, color='g')
        graph.draw_cuboid(ax1, xx, yy, zz, color='k', zorder=100)

        # ax1.view_init(90, 0)
        # ax1.view_init(0, 90)

        pt_c = np.matmul(Mac, pt)
        ps, qs, ns = npq_coords[1, :], npq_coords[2, :], npq_coords[0, :]
        fig2, ax2 = graph.scatter(ps, qs, s=1, fignum=1, subplot=122)
        fig2, ax2 = graph.scatter([pt_c[1]], [pt_c[2]], s=200, marker='x', ax=ax2, figsize=(16, 8))
        graph.labelaxes(ax2, 'p', 'q')
        ax2.set_aspect('equal')

    if notebook:
        from tqdm import tqdm as tqdm

    return xs, ys, zs, npq_coords[1, :], npq_coords[2, :], npq_coords[0, :], Mac, Mca, basisC


## Get a slice of a 3D velocity field
def slice_udata_3d(udata, xx, yy, zz, n, pt, spacing=None, show=False,
                   method='nn', max_iter=10, tol=0.05, median_filter=True,
                   basis=None, u_basis='npq', showtqdm=True, verbose=False, notebook=True,
                   **kwargs):
    """
    Returns a spatially 2D udata which is on the cross section of a volumetric data
    ... There are two ways to return the velocity field on the cross section.
        1. In the standard basis (xyz, i.e. ux, uy, uz)
        2. In the NEW basis (which I call npq basis, i.e. un, up, uq)
            the basis vector n is identical to the unit area vector of the cross section
    ... By default, this function returns a 2D velocity field in the NEW basis (npq) as well as its coordinates in the same basis.
        ... This is a natural choice since the cross section is spanned by the new basis vectors p and q.
        ... Any postional vector on the cross section is expresed by its linear combinations.
            r = c1 \hat{p} + c2 \hat{q}
            (c1, c2) are the coordinates in the pq basis.
            (Technically, the basis vectors are n, p, q but the coefficient of the n basis vector is always zero on the cross section.)
            By default, this function returns...
            1. velocity field in the NEW basis (npq) on the cross section
            2. Coordinates in the pq basis (technically npq basis, but the coordinate in the basis vector n is always zero so I don't return it)
    ... You CAN retrieve the velocity field in the standard basis (xyz, i.e. ux, uy, uz).
        Just set "u_basis" equal to "xyz"
        If you want both (in npq and xyz basses), set  "u_basis" as "both"
    ... If you want a change-of-basis matrix from basis A to basis B, use ilpm.vector.get_change_of_basis_matrix(basisA, basisB)
        e.g.
            Mab = vec.get_change_of_basis_matrix(basis_xyz, basis_npq) # transformation matrix from xyz coords to npq coords
            Mba = np.linalg.inv(Mab) # the change-of-basis matrix is ALWAYS unitary.
    ... e.g. You want a v-field on a slice whose normal vector is obtained by rotating the unit z vector rotated by +45 degrees
            theta = np.pi/180.*-45 # Convert degrees to rad
            n = [np.cos(theta), np.sin(theta), 0] # This is the normal vector of the slice
            pt = [0, 0, 0] # A point on the slice
            # Providing the new basis (npq basis) is optional
            # but this helps to speed up the process to find all available points on the slice
            basis = np.asarray([n, [0, 0, 1], [np.sin(theta), -np.cos(theta), 0]]).T # The transpose is necessary
            vdata, pp, qq = vel.slice_udata_3d(udata, xxx, yyy, zzz, n, pt, basis=basis, show=True)
            enst = vel.get_enstrophy(vdata[1:, ...], xx=pp, yy=qq)
            graph.color_plot(pp, qq, enst[..., 0], fignum=1+i)
    ... e.g. get a v-field on the xy plane (z=0)
            n = [0, 0, 1] # area vector // unit z vector
            pt = [0, 0, 0] # the plane contains the origin
            basis = np.asarray([n, [1, 0, 0], [0, 1, 0]]).T
            vdata, pp, qq = vel.slice_udata_3d(udata, xxx, yyy, zzz, n, pt, basis=basis, show=True)
            enst = vel.get_enstrophy(vdata[1:, ...], xx=pp, yy=qq)
            graph.color_plot(pp, qq, enst[..., 0], fignum=1+i)
    ... e.g. Get a slice whose normal vector gets rotated about the y-axis gradually
            t = 0 # slice the udata in the 1st frame
            thetas = np.linspace(0, 2*np.pi, 101)
            for i, theta in enumerate(thetas):
                n = [np.sin(theta), 0, np.cos(theta)]
                vdata, pp, qq = vel.slice_udata_3d(udata[..., t], xxx, yyy, zzz, n, [xc, yc, zc],
                                                   basis = np.asarray([n, [np.cos(theta), 0, -np.sin(theta)], [0, 1, 0]]).T,
                                                   apply_convention=False)
    Example:

    Parameters
    ----------
    udata: spatially 3d udata (4D or 5D array) with shape (dim, height, width, depth, duration) or  (dim, height, width, depth)
    xx: 3d array, grid of x-coordinate
    yy: 3d array, grid of y-coordinate
    zz: 3d array, grid of z-coordinate
    n: 1d array-like, area vector (it does not have to be normalized)
    pt: 1d array-like, xyz coordinates of a point on the plane (Note that n and pt uniquely defines a plane)
    spacing: float, must be greater than 0. Sampling spatial resolution on the plane
    show: bool, tflow.graph or takumi.graph is required. If True, it will automatically plot the sampled points on the
        cross section in a 3D view as well as the new basis vectors
    method: str, interpolation method of udata, options: 'nn', 'localmean', 'idw'
        ... volumetric udata may contain lots of np.nan, and cleaning udata often results better results.
        ... 'nn': nearest neighbor filling
        ... 'localmean': filling using a direct covolution (inpainting with a neighbor averaging kernel)
        ... 'idw': filling using a direct covolution (inpainting with a Gaussian kernel)
    max_iter: int, parameter for replacng nans in udata clean_udata()
    tol: float, parameter for replacng nans in udata clean_udata()
    basis: 3x3 matrix, column vectors are the basis vectors.
        ... If given, these basis vectors are used to span the crosss section.
        ... The basis vectors may be rotated or converted to be right-handed at the end.
        ... See the example above for a sample input
    apply_convention: bool, If True, it enforces the new basis to follow a convention that is
        ... right-handed (n x p // q)
    u_basis: str, default: "npq", options: "npq", "xyz" / "standard" "std", "both"
        ... the basis used to represent a velocity field on the plane
        ... "npq": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (un, up, uq)
            (un, up, uq) = (velocity component parallel to the area vector,
                            vel comp parallel to the second basis vector p,
                            vel comp parallel to the second basis vector q)
        ... "xyz". "standard", or "std": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (ux, uy, uz)

    Returns
    -------
    udata_si_npq_basis: udata in the npq basis on the plane spanned by the basis vectors p and q.
        ... its shape is (dim, height, width, duration) = (3, height, width, duration)
        ... udata_si_npq_basis[0, ...] is velocity u \cdot \hat{n}
        ... udata_si_npq_basis[1, ...] is velocity u \cdot \hat{p}
        ... udata_si_npq_basis[2, ...] is velocity u \cdot \hat{q}
    pp: 2d nd array, p-grid
    qq: 2d nd array, q-grid
        ... plt.quiver(pp, qq, udata_si_npq_basis[1, ..., 0], udata_si_npq_basis[2, ..., 0])
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if spacing is None:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        spacing = min([dx, dy, dz])
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]

    # Clean udata
    udata = clean_udata(udata, method=method, max_iter=max_iter, tol=tol, median_filter=median_filter, showtqdm=verbose)
    # Extract info about the cross section
    xx_plane, yy_plane, zz_plane, pp, qq, Mab, Mba, basis = slicer(xx, yy, zz, n, pt, spacing=spacing, basis=basis, )
    #
    pmin, pmax, qmin, qmax = np.nanmin(pp), np.nanmax(pp), np.nanmin(qq), np.nanmax(qq)
    p = np.linspace(pmin, pmax, int((pmax - pmin) // spacing))
    q = np.linspace(qmin, qmax, int((qmax - qmin) // spacing))
    n = [0]
    pp, qq, nn = np.meshgrid(p, q, n)  # THIS WORKS! DO not touch this
    shape = pp.shape

    for i, t in tqdm(enumerate(range(duration)), disable=not showtqdm):
        # udata_s_tmp = clean_udata(udata[..., t:t + 1])
        fs = interpolate_udata_at_instant_of_time(udata, xx, yy, zz, t=t, bounds_error=False)
        uxi = fs[0]((yy_plane, xx_plane, zz_plane))
        uyi = fs[1]((yy_plane, xx_plane, zz_plane))
        uzi = fs[2]((yy_plane, xx_plane, zz_plane))
        # shape = uxi.shape
        uis_xyz = np.stack((uxi.flatten(), uyi.flatten(), uzi.flatten()))
        uis_npq = np.matmul(Mab, uis_xyz).reshape((3,) + shape)
        uis_xyz = uis_xyz.reshape((3,) + shape)

        # pts_b = np.stack((nn.flatten(), pp.flatten(), qq.flatten()))  # npq
        #
        # pts_a = np.matmul(Mba, pts_b)  # x, y, z
        # pts_a[[0, 1], :] = pts_a[[1, 0], :]  # interpolating function takes (y, x, z) not (x, y, z). Swap axes.
        #
        # ux_si = fs[0](pts_a.T).reshape(shape)
        # uy_si = fs[1](pts_a.T).reshape(shape)
        # uz_si = fs[2](pts_a.T).reshape(shape)
        #
        # uis_si = np.stack((ux_si.flatten(), uy_si.flatten(), uz_si.flatten()))
        # uis_si_npq = np.matmul(Mab, uis_si)
        # un_si, up_si, uq_si = uis_si_npq[0, ...].reshape(shape), uis_si_npq[1, ...].reshape(shape), uis_si_npq[
        #     2, ...].reshape(shape)
        if i == 0:
            master_shape = (3,) + shape + (duration,)
            udata_si_xyz_basis = np.empty(master_shape)
            udata_si_npq_basis = np.empty(master_shape)

        udata_si_xyz_basis[..., t] = uis_xyz
        udata_si_npq_basis[..., t] = uis_npq  # un, up, uq- velocity also obeys the transformation rule as position
    udata_si_xyz_basis = np.squeeze(udata_si_xyz_basis)  # ux, uy, uz
    udata_si_npq_basis = np.squeeze(udata_si_npq_basis)  # un, up, uq

    pp, qq = np.squeeze(pp), np.squeeze(qq)  # Make it into 2D grids

    if notebook:
        from tqdm import tqdm as tqdm

    if u_basis == 'npq':
        return udata_si_npq_basis, pp, qq
    elif u_basis in ['standard', 'std', 'xyz']:
        return udata_si_xyz_basis, pp, qq
    elif u_basis == 'both':
        return udata_si_npq_basis, udata_si_xyz_basis, pp, qq
    else:
        print('... Pass which basis is used to represent the extraceted velocity field on the slice')
        print('... Choices: "npq"- (un, up, uq) velocity parallel to the surface vector, and its transverse directions')
        print('... Choices: "xyz"- (ux, uy, uz) velocity in the standard basis')
        return None


def slice_udata_3d_old(udata, xx, yy, zz, n, pt, spacing=None, show=False,
                       method='nn', max_iter=10, tol=0.05, median_filter=True,
                       basis=None, apply_convention=True,
                       u_basis='npq', notebook=True):
    """
    Returns a spatially 2D udata which is on the cross section of a volumetric data
    ... There are two ways to return the velocity field on the cross section.
        1. In the standard basis (xyz, i.e. ux, uy, uz)
        2. In the NEW basis (which I call npq basis, i.e. un, up, uq)
            the basis vector n is identical to the unit area vector of the cross section
    ... By default, this function returns a 2D velocity field in the NEW basis (npq) as well as its coordinates in the same basis.
        ... This is a natural choice since the cross section is spanned by the new basis vectors p and q.
        ... Any postional vector on the cross section is expresed by its linear combinations.
            r = c1 \hat{p} + c2 \hat{q}
            (c1, c2) are the coordinates in the pq basis.
            (Technically, the basis vectors are n, p, q but the coefficient of the n basis vector is always zero on the cross section.)
            By default, this function returns...
            1. velocity field in the NEW basis (npq) on the cross section
            2. Coordinates in the pq basis (technically npq basis, but the coordinate in the basis vector n is always zero so I don't return it)
    ... You CAN retrieve the velocity field in the standard basis (xyz, i.e. ux, uy, uz).
        Just set "u_basis" equal to "xyz"
        If you want both (in npq and xyz basses), set  "u_basis" as "both"
    ... If you want a change-of-basis matrix from basis A to basis B, use ilpm.vector.get_change_of_basis_matrix(basisA, basisB)
        e.g.
            Mab = vec.get_change_of_basis_matrix(basis_xyz, basis_npq) # transformation matrix from xyz coords to npq coords
            Mba = np.linalg.inv(Mab) # the change-of-basis matrix is ALWAYS unitary.
    ... e.g. You want a v-field on a slice whose normal vector is obtained by rotating the unit z vector rotated by +45 degrees
            theta = np.pi/180.*-45 # Convert degrees to rad
            n = [np.cos(theta), np.sin(theta), 0] # This is the normal vector of the slice
            pt = [0, 0, 0] # A point on the slice
            # Providing the new basis (npq basis) is optional
            # but this helps to speed up the process to find all available points on the slice
            basis = np.asarray([n, [0, 0, 1], [np.sin(theta), -np.cos(theta), 0]]).T # The transpose is necessary
            vdata, pp, qq = vel.slice_udata_3d(udata, xxx, yyy, zzz, n, pt, basis=basis, show=True)
            enst = vel.get_enstrophy(vdata[1:, ...], xx=pp, yy=qq)
            graph.color_plot(pp, qq, enst[..., 0], fignum=1+i)
    ... e.g. get a v-field on the xy plane (z=0)
            n = [0, 0, 1] # area vector // unit z vector
            pt = [0, 0, 0] # the plane contains the origin
            basis = np.asarray([n, [1, 0, 0], [0, 1, 0]]).T
            vdata, pp, qq = vel.slice_udata_3d(udata, xxx, yyy, zzz, n, pt, basis=basis, show=True)
            enst = vel.get_enstrophy(vdata[1:, ...], xx=pp, yy=qq)
            graph.color_plot(pp, qq, enst[..., 0], fignum=1+i)
    ... e.g. Get a slice whose normal vector gets rotated about the y-axis gradually
            t = 0 # slice the udata in the 1st frame
            thetas = np.linspace(0, 2*np.pi, 101)
            for i, theta in enumerate(thetas):
                n = [np.sin(theta), 0, np.cos(theta)]
                vdata, pp, qq = vel.slice_udata_3d(udata[..., t], xxx, yyy, zzz, n, [xc, yc, zc],
                                                   basis = np.asarray([n, [np.cos(theta), 0, -np.sin(theta)], [0, 1, 0]]).T,
                                                   apply_convention=False)
    Example:

    Parameters
    ----------
    udata: spatially 3d udata (4D or 5D array) with shape (dim, height, width, depth, duration) or  (dim, height, width, depth)
    xx: 3d array, grid of x-coordinate
    yy: 3d array, grid of y-coordinate
    zz: 3d array, grid of z-coordinate
    n: 1d array-like, area vector (it does not have to be normalized)
    pt: 1d array-like, xyz coordinates of a point on the plane (Note that n and pt uniquely defines a plane)
    spacing: float, must be greater than 0. Sampling spatial resolution on the plane
    show: bool, tflow.graph or takumi.graph is required. If True, it will automatically plot the sampled points on the
        cross section in a 3D view as well as the new basis vectors
    method: str, interpolation method of udata, options: 'nn', 'localmean', 'idw'
        ... volumetric udata may contain lots of np.nan, and cleaning udata often results better results.
        ... 'nn': nearest neighbor filling
        ... 'localmean': filling using a direct covolution (inpainting with a neighbor averaging kernel)
        ... 'idw': filling using a direct covolution (inpainting with a Gaussian kernel)
    max_iter: int, parameter for replacng nans in udata clean_udata()
    tol: float, parameter for replacng nans in udata clean_udata()
    basis: 3x3 matrix, column vectors are the basis vectors.
        ... If given, these basis vectors are used to span the crosss section.
        ... The basis vectors may be rotated or converted to be right-handed at the end.
        ... See the example above for a sample input
    apply_convention: bool, If True, it enforces the new basis to follow a convention that is
        ... right-handed (n x p // q)
    u_basis: str, default: "npq", options: "npq", "xyz" / "standard" "std", "both"
        ... the basis used to represent a velocity field on the plane
        ... "npq": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (un, up, uq)
            (un, up, uq) = (velocity component parallel to the area vector,
                            vel comp parallel to the second basis vector p,
                            vel comp parallel to the second basis vector q)
        ... "xyz". "standard", or "std": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (ux, uy, uz)



    Returns
    -------
    udata_si_npq_basis: udata in the npq basis on the plane spanned by the basis vectors p and q.
        ... its shape is (dim, height, width, duration) = (3, height, width, duration)
        ... udata_si_npq_basis[0, ...] is velocity u \cdot \hat{n}
        ... udata_si_npq_basis[1, ...] is velocity u \cdot \hat{p}
        ... udata_si_npq_basis[2, ...] is velocity u \cdot \hat{q}
    pp: 2d nd array, p-grid
    qq: 2d nd array, q-grid
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if spacing is None:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        spacing = min([dx, dy, dz])
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]

    # Clean udata
    udata = clean_udata(udata, method=method, max_iter=max_iter, tol=tol, median_filter=median_filter)
    # Extract info about the cross section
    xs, ys, zs, ps, qs, ns, Mab, Mba, basis = slicer_old(xx, yy, zz, n, pt, spacing=spacing,
                                                         basis=basis, apply_convention=apply_convention,
                                                         show=show)
    udata_si_npq_basis = np.empty_like(udata)
    for i, t in tqdm(enumerate(range(duration))):
        # udata_s_tmp = clean_udata(udata[..., t:t + 1])
        fs = interpolate_udata_at_instant_of_time(udata, xx, yy, zz, t=t, bounds_error=False)

        pmin, pmax, qmin, qmax = np.nanmin(ps), np.nanmax(ps), np.nanmin(qs), np.nanmax(qs)
        p = np.arange(pmin, pmax, spacing)
        q = np.arange(qmin, qmax, spacing)
        n = [0]
        pp, qq, nn = np.meshgrid(p, q, n)  # THIS WORKS! DO not touch this
        shape = pp.shape

        pts_b = np.stack((nn.flatten(), pp.flatten(), qq.flatten()))  # npq

        pts_a = np.matmul(Mba, pts_b)  # x, y, z
        pts_a[[0, 1], :] = pts_a[[1, 0], :]  # interpolating function takes (y, x, z) not (x, y, z). Swap axes.

        ux_si = fs[0](pts_a.T).reshape(shape)
        uy_si = fs[1](pts_a.T).reshape(shape)
        uz_si = fs[2](pts_a.T).reshape(shape)

        uis_si = np.stack((ux_si.flatten(), uy_si.flatten(), uz_si.flatten()))
        uis_si_npq = np.matmul(Mab, uis_si)
        un_si, up_si, uq_si = uis_si_npq[0, ...].reshape(shape), uis_si_npq[1, ...].reshape(shape), uis_si_npq[
            2, ...].reshape(shape)
        if i == 0:
            master_shape = (3,) + shape + (duration,)
            udata_si_xyz_basis = np.empty(master_shape)
            udata_si_npq_basis = np.empty(master_shape)

        udata_si_xyz_basis[..., t] = np.stack((ux_si, uy_si, uz_si))
        udata_si_npq_basis[..., t] = np.stack(
            (un_si, up_si, uq_si))  # un, up, uq- velocity also obeys the transformation rule as position
    udata_si_xyz_basis = np.squeeze(udata_si_xyz_basis)  # ux, uy, uz
    udata_si_npq_basis = np.squeeze(udata_si_npq_basis)  # un, up, uq

    pp, qq = np.squeeze(pp), np.squeeze(qq)  # 2D grid

    if notebook:
        from tqdm import tqdm as tqdm

    if u_basis == 'npq':
        return udata_si_npq_basis, pp, qq
    elif u_basis in ['standard', 'std', 'xyz']:
        return udata_si_xyz_basis, pp, qq
    elif u_basis == 'both':
        return udata_si_npq_basis, udata_si_xyz_basis, pp, qq
    else:
        print('... Pass which basis is used to represent the extraceted velocity field on the slice')
        print('... Choices: "npq"- (un, up, uq) velocity parallel to the surface vector, and its transverse directions')
        print('... Choices: "xyz"- (ux, uy, uz) velocity in the standard basis')
        return None


## Get a slice of a 3D scalar field
def slice_3d_scalar_field(field, xx, yy, zz, n, pt, spacing=None,
                          basis=None,
                          showtqdm=True, verbose=False, notebook=True, **kwargs):
    """
    Returns a spatially 2D udata which is on the cross section of a volumetric data
    ... There are two ways to return the velocity field on the cross section.
        1. In the standard basis (xyz, i.e. ux, uy, uz)
        2. In the NEW basis (which I call npq basis, i.e. un, up, uq)
            the basis vector n is identical to the unit area vector of the cross section
    ... By default, this function returns a 2D velocity field in the NEW basis (npq) as well as its coordinates in the same basis.
        ... This is a natural choice since the cross section is spanned by the new basis vectors p and q.
        ... Any postional vector on the cross section is expresed by its linear combinations.
            r = c1 \hat{p} + c2 \hat{q}
            (c1, c2) are the coordinates in the pq basis.
            (Technically, the basis vectors are n, p, q but the coefficient of the n basis vector is always zero on the cross section.)
            By default, this function returns...
            1. velocity field in the NEW basis (npq) on the cross section
            2. Coordinates in the pq basis (technically npq basis, but the coordinate in the basis vector n is always zero so I don't return it)
    ... You CAN retrieve the velocity field in the standard basis (xyz, i.e. ux, uy, uz).
        Just set "u_basis" equal to "xyz"
        If you want both (in npq and xyz basses), set  "u_basis" equal to "both"
    ... If you want a change-of-basis matrix from basis A to basis B, use ilpm.vector.get_change_of_basis_matrix(basisA, basisB)
        e.g.
            Mab = vec.get_change_of_basis_matrix(basis_xyz, basis_npq) # transformation matrix from xyz coords to npq coords
            Mba = np.linalg.inv(Mab) # the change-of-basis matrix is ALWAYS unitary.

    Example:

    Parameters
    ----------
    filed: spatially 3d udata (3D or 4D array) with shape (height, width, depth, duration) or  (dim, height, width, depth)
    xx: 3d array, grid of x-coordinate
    yy: 3d array, grid of y-coordinate
    zz: 3d array, grid of z-coordinate
    n: 1d array-like, area vector (it does not have to be normalized)
    pt: 1d array-like, xyz coordinates of a point on the plane (Note that n and pt uniquely defines a plane)
    spacing: float, must be greater than 0. Sampling spatial resolution on the plane
    show: bool, tflow.graph or takumi.graph is required. If True, it will automatically plot the sampled points on the
        cross section in a 3D view as well as the new basis vectors
    method: str, interpolation method of udata, options: 'nn', 'localmean', 'idw'
        ... volumetric udata may contain lots of np.nan, and cleaning udata often results better results.
        ... 'nn': nearest neighbor filling
        ... 'localmean': filling using a direct covolution (inpainting with a neighbor averaging kernel)
        ... 'idw': filling using a direct covolution (inpainting with a Gaussian kernel)
    max_iter: int, parameter for replacng nans in udata clean_udata()
    tol: float, parameter for replacng nans in udata clean_udata()
    basis: 3x3 matrix, column vectors are the basis vectors.
        ... If given, the basis vectors are used to span the crosss section.
        ... The basis vectors may be rotated or converted to right-handed at the end.
        ...
    apply_convention: bool, If True, it enforces the new basis to follow the convention (right-handed etc.)
    u_basis: str, default: "npq", options: "npq", "xyz" / "standard" "std", "both"
        ... the basis used to represent a velocity field on the plane
        ... "npq": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (un, up, uq)
            (un, up, uq) = (velocity component parallel to the area vector,
                            vel comp parallel to the second basis vector p,
                            vel comp parallel to the second basis vector q)
        ... "xyz". "standard", or "std": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (ux, uy, uz)



    Returns
    -------
    udata_si_npq_basis: udata in the npq basis on the plane spanned by the basis vectors p and q.
        ... its shape is (dim, height, width, duration) = (3, height, width, duration)
        ... udata_si_npq_basis[0, ...] is velocity u \cdot \hat{n}
        ... udata_si_npq_basis[1, ...] is velocity u \cdot \hat{p}
        ... udata_si_npq_basis[2, ...] is velocity u \cdot \hat{q}
    pp: 2d nd array, p-grid
    qq: 2d nd array, q-grid
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if spacing is None:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        spacing = min([dx, dy, dz])
        print('slice_3d_scalar_field:')
        print('... spacing: ', spacing)

    field = np.asarray(field)
    if len(field.shape) == 3:
        field = field.reshape(field.shape + (1,))
    duration = field.shape[-1]

    # Extract info about the cross section
    # Extract info about the cross section
    xx_plane, yy_plane, zz_plane, pp, qq, Mab, Mba, basis = slicer(xx, yy, zz, n, pt, spacing=spacing, basis=basis, )
    #
    pmin, pmax, qmin, qmax = np.nanmin(pp), np.nanmax(pp), np.nanmin(qq), np.nanmax(qq)
    p = np.linspace(pmin, pmax, int((pmax - pmin) // spacing))
    q = np.linspace(qmin, qmax, int((qmax - qmin) // spacing))
    n = [0]
    pp, qq, nn = np.meshgrid(p, q, n)  # THIS WORKS! DO not touch this
    shape = pp.shape

    for i, t in tqdm(enumerate(range(duration)), disable=not showtqdm):
        fi = interpolate_scalar_field_at_instant_of_time(field, xx, yy, zz, t=t,
                                                         bounds_error=False)  # interpolating function
        field_si = fi((yy_plane, xx_plane, zz_plane))

        if i == 0:
            master_shape = pp.shape[:-1] + (duration,)
            field_si_xyz_basis = np.empty(master_shape)

        field_si_xyz_basis[..., t] = field_si

    field_si_xyz_basis = np.squeeze(field_si_xyz_basis)  # ux, uy, uz

    pp, qq = np.squeeze(pp), np.squeeze(qq)  # 2D grid

    if notebook:
        from tqdm import tqdm as tqdm

    return field_si_xyz_basis, pp, qq


def slice_3d_scalar_field_old(field, xx, yy, zz, n, pt, spacing=None, show=False,
                              basis=None, apply_convention=True,
                              notebook=True):
    """
    [DEPRECATED- slice_3d_scalar_field]
    Returns a spatially 2D udata which is on the cross section of a volumetric data
    ... There are two ways to return the velocity field on the cross section.
        1. In the standard basis (xyz, i.e. ux, uy, uz)
        2. In the NEW basis (which I call npq basis, i.e. un, up, uq)
            the basis vector n is identical to the unit area vector of the cross section
    ... By default, this function returns a 2D velocity field in the NEW basis (npq) as well as its coordinates in the same basis.
        ... This is a natural choice since the cross section is spanned by the new basis vectors p and q.
        ... Any postional vector on the cross section is expresed by its linear combinations.
            r = c1 \hat{p} + c2 \hat{q}
            (c1, c2) are the coordinates in the pq basis.
            (Technically, the basis vectors are n, p, q but the coefficient of the n basis vector is always zero on the cross section.)
            By default, this function returns...
            1. velocity field in the NEW basis (npq) on the cross section
            2. Coordinates in the pq basis (technically npq basis, but the coordinate in the basis vector n is always zero so I don't return it)
    ... You CAN retrieve the velocity field in the standard basis (xyz, i.e. ux, uy, uz).
        Just set "u_basis" equal to "xyz"
        If you want both (in npq and xyz basses), set  "u_basis" equal to "both"
    ... If you want a change-of-basis matrix from basis A to basis B, use ilpm.vector.get_change_of_basis_matrix(basisA, basisB)
        e.g.
            Mab = vec.get_change_of_basis_matrix(basis_xyz, basis_npq) # transformation matrix from xyz coords to npq coords
            Mba = np.linalg.inv(Mab) # the change-of-basis matrix is ALWAYS unitary.

    Example:

    Parameters
    ----------
    filed: spatially 3d udata (3D or 4D array) with shape (height, width, depth, duration) or  (dim, height, width, depth)
    xx: 3d array, grid of x-coordinate
    yy: 3d array, grid of y-coordinate
    zz: 3d array, grid of z-coordinate
    n: 1d array-like, area vector (it does not have to be normalized)
    pt: 1d array-like, xyz coordinates of a point on the plane (Note that n and pt uniquely defines a plane)
    spacing: float, must be greater than 0. Sampling spatial resolution on the plane
    show: bool, tflow.graph or takumi.graph is required. If True, it will automatically plot the sampled points on the
        cross section in a 3D view as well as the new basis vectors
    method: str, interpolation method of udata, options: 'nn', 'localmean', 'idw'
        ... volumetric udata may contain lots of np.nan, and cleaning udata often results better results.
        ... 'nn': nearest neighbor filling
        ... 'localmean': filling using a direct covolution (inpainting with a neighbor averaging kernel)
        ... 'idw': filling using a direct covolution (inpainting with a Gaussian kernel)
    max_iter: int, parameter for replacng nans in udata clean_udata()
    tol: float, parameter for replacng nans in udata clean_udata()
    basis: 3x3 matrix, column vectors are the basis vectors.
        ... If given, the basis vectors are used to span the crosss section.
        ... The basis vectors may be rotated or converted to right-handed at the end.
        ...
    apply_convention: bool, If True, it enforces the new basis to follow the convention (right-handed etc.)
    u_basis: str, default: "npq", options: "npq", "xyz" / "standard" "std", "both"
        ... the basis used to represent a velocity field on the plane
        ... "npq": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (un, up, uq)
            (un, up, uq) = (velocity component parallel to the area vector,
                            vel comp parallel to the second basis vector p,
                            vel comp parallel to the second basis vector q)
        ... "xyz". "standard", or "std": It returns velocity vectors in the npq basis. Returning udata[i, ...] = (ux, uy, uz)



    Returns
    -------
    udata_si_npq_basis: udata in the npq basis on the plane spanned by the basis vectors p and q.
        ... its shape is (dim, height, width, duration) = (3, height, width, duration)
        ... udata_si_npq_basis[0, ...] is velocity u \cdot \hat{n}
        ... udata_si_npq_basis[1, ...] is velocity u \cdot \hat{p}
        ... udata_si_npq_basis[2, ...] is velocity u \cdot \hat{q}
    pp: 2d nd array, p-grid
    qq: 2d nd array, q-grid
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if spacing is None:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        spacing = min([dx, dy, dz])
        print('slice_3d_scalar_field:')
        print('... spacing: ', spacing)

    field = np.asarray(field)
    if len(field.shape) == 3:
        field = field.reshape(field.shape + (1,))
    duration = field.shape[-1]

    # Extract info about the cross section
    xs, ys, zs, ps, qs, ns, Mab, Mba, basis = slicer_old(xx, yy, zz, n, pt, spacing=spacing,
                                                         basis=basis, apply_convention=apply_convention,
                                                         show=show)
    for i, t in tqdm(enumerate(range(duration))):
        fi = interpolate_scalar_field_at_instant_of_time(field, xx, yy, zz, t=t,
                                                         bounds_error=False)  # interpolating function

        pmin, pmax, qmin, qmax = np.nanmin(ps), np.nanmax(ps), np.nanmin(qs), np.nanmax(qs)
        p = np.arange(pmin, pmax, spacing)
        q = np.arange(qmin, qmax, spacing)
        n = [0]
        pp, qq, nn = np.meshgrid(p, q, n)  # THIS WORKS! DO not touch this
        shape = pp.shape
        pts_b = np.stack((nn.flatten(), pp.flatten(), qq.flatten()))  # npq

        pts_a = np.matmul(Mba, pts_b)  # x, y, z
        pts_a[[0, 1], :] = pts_a[[1, 0], :]  # interpolating function takes (y, x, z) not (x, y, z). Swap axes.

        field_si = fi(pts_a.T).reshape(shape)
        if i == 0:
            master_shape = shape + (duration,)
            field_si_xyz_basis = np.empty(master_shape)

        field_si_xyz_basis[..., t] = field_si

    field_si_xyz_basis = np.squeeze(field_si_xyz_basis)  # ux, uy, uz

    pp, qq = np.squeeze(pp), np.squeeze(qq)  # 2D grid

    if notebook:
        from tqdm import tqdm as tqdm

    return field_si_xyz_basis, pp, qq


def interpolate_udata_at_instant_of_time(udata, xx, yy, zz=None, t=0, bounds_error=False, **kwargs):
    """
    Returns interpolating functions of each spatial component of a velocity field at instant of time
    using scipy.interpolate.RegularGridInterpolator
    ... Returns a tuple (f_ux, f_uy, f_uz)
    ... f_ux( (y, x, z) ) gives ux at (x, y, z)

    Parameters
    ----------
    udata: ndarray, shape (dim, height, width, (duration))
    xx: 2d/3d array, x-grid
    yy: 2d/3d array, y-grid
    zz: 2d/3d array, z-grid, optional
    t: int, default: 0
        This function generates interpolating functions of udata at time index of
    bounds_error: bool, default: False
        ... If True, when interpolating outside of the input data, raise a ValueError.
    kwargs: kwargs will be passed to scipy.interpolate.RegularGridInterpolator()

    Returns
    -------
    f_ux: interpolating function of ux at time t
    f_uy: interpolating function of uy at time t
    f_uz: interpolating function of uz at time t, optional
    """
    udata = fix_udata_shape(udata)
    try:
        dim, height, width, depth, duration = udata.shape
    except:
        dim, height, width, duration = udata.shape

    if zz is not None:
        y, x, z = yy[:, 0, 0], xx[0, :, 0], zz[0, 0, :]
        if np.all(np.diff(z) < 0):
            z = np.flip(z)
            udata = np.flip(udata, axis=3)
    else:
        y, x = yy[:, 0], xx[0, :]
    if np.all(np.diff(x) < 0):
        x = np.flip(x)
        udata = np.flip(udata, axis=2)
    if np.all(np.diff(y) < 0):
        y = np.flip(y)
        udata = np.flip(udata, axis=1)

    if zz is None:
        f_ux = RegularGridInterpolator((y, x), udata[0, ..., t], bounds_error=bounds_error,
                                       **kwargs)  # ux interpolating function
        f_uy = RegularGridInterpolator((y, x), udata[1, ..., t], bounds_error=bounds_error,
                                       **kwargs)  # uy interpolating function
        if dim == 3:
            f_uz = RegularGridInterpolator((y, x), udata[2, ..., t], bounds_error=bounds_error,
                                           **kwargs)  # uz interpolating function
    else:
        f_ux = RegularGridInterpolator((y, x, z), udata[0, ..., t], bounds_error=bounds_error,
                                       **kwargs)  # ux interpolating function
        f_uy = RegularGridInterpolator((y, x, z), udata[1, ..., t], bounds_error=bounds_error,
                                       **kwargs)  # uy interpolating function
        if dim == 3:
            f_uz = RegularGridInterpolator((y, x, z), udata[2, ..., t], bounds_error=bounds_error,
                                           **kwargs)  # uz interpolating function

    if dim == 3:
        return f_ux, f_uy, f_uz
    else:
        return f_ux, f_uy


def interpolate_scalar_field_at_instant_of_time(field, xx, yy, zz=None, t=0, bounds_error=False, **kwargs):
    """
    Returns an interpolating function a scalar field (energy, enstrophy etc.) at instant of time
    using scipy.interpolate.RegularGridInterpolator
    ... field_i( (y, x, z) ): the field at (x, y, z)

    Parameters
    ----------
    field: ndarray, scalar field, shape (height, width, (depth), duration)
        ... The most right index of "field" is the time index.
    xx: 2d/3d array, x-grid
    yy: 2d/3d array, y-grid
    zz: 2d/3d array, z-grid, optional
    t: int, time index, default: 0
    bounds_error: bool, default: False
        ... If True, when interpolating outside of the input data, raise a ValueError.
    kwargs: kwargs will be passed to scipy.interpolate.RegularGridInterpolator()

    Returns
    -------
    field_i: interpolating function of the scalar field at time t
    """
    dim = len(xx.shape)

    try:
        try:
            height, width, depth, duration = field.shape
        except:
            height, width, duration = field.shape
    except:
        field = field.reshape(field.shape + (1,))
        try:
            height, width, depth, duration = field.shape
        except:
            height, width, duration = field.shape

    if zz is not None:
        y, x, z = yy[:, 0, 0], xx[0, :, 0], zz[0, 0, :]
        if np.all(np.diff(z) < 0):
            z = np.flip(z)
            field = np.flip(field, axis=1)
    else:
        y, x = yy[:, 0], xx[0, :]
    if np.all(np.diff(x) < 0):
        x = np.flip(x)
        field = np.flip(field, axis=1)
    if np.all(np.diff(y) < 0):
        y = np.flip(y)
        field = np.flip(field, axis=0)

    if zz is None:
        field_i = RegularGridInterpolator((y, x), field[..., t], bounds_error=bounds_error,
                                          **kwargs)  # interpolating function
    else:
        field_i = RegularGridInterpolator((y, x, z), field[..., t], bounds_error=bounds_error,
                                          **kwargs)
    return field_i


def interpolate_vector_field_at_instant_of_time(vfield, xx, yy, zz=None, t=0, bounds_error=False, **kwargs):
    """
    Returns interpolating functions of a vector field (e.g. vorticity field: omega)
    using scipy.interpolate.RegularGridInterpolator
    ... e.g.
        Input: vorticity field (omega) with shape (3, height, width, depth, duration)
        Output: a list of interpolating functions for each spatial direction- [omega_x, omega_y, omega_z]

    Parameters
    ----------
    vfield: ndarray, vector field, shape (dim, height, width, (depth), duration)
    xx: 2d/3d array, x-grid
    yy: 2d/3d array, y-grid
    zz: 2d/3d array, z-grid, optional
    t: int, time index, default: 0
    bounds_error: bool, default: False
    ... If True, when interpolating outside of the input data, raise a ValueError.
    kwargs: kwargs will be passed to scipy.interpolate.RegularGridInterpolator()

    Returns
    -------
    funcs: list, interpolating functions
        ... funcs[n] is the interpolating function of the vfield[n, ..., t]
    """
    dim = vfield.shape[0]
    funcs = []
    for d in range(dim):
        func = interpolate_scalar_field_at_instant_of_time(vfield[d, ...], xx, yy, zz=zz, t=t,
                                                           bounds_error=bounds_error, **kwargs)
        funcs.append(func)
    return funcs


def griddata_easy(x, y, data, xi=None, yi=None, dx=None, dy=None, nx=10, ny=10, method='nearest',
                   xmin=None, xmax=None, ymin=None, ymax=None, fill_value=None):
    """
    Conduct 2D interpolation of data from a nonuniformly spaced rectangular grid
    ... A wrapper of scipy.interplate.griddata

    Parameters
    ----------
    xx: nd array-like
        x-coordinate of scattered data
        this will be flattened when passed to griddata
    yy: nd array-like
        y-coordinate of scattered data
        this will be flattened when passed to griddata
    data: nd array-like
        values of scattered data
        this will be flattened when passed to griddata
    xi: 1d array
        x-coordinate of the interpolated grid
        ... The array must be monotonically increasing.
        ... If None, xi = np.arange(xmin, xmax, dx)
    yi: 1d array
        y-coordinate of the interpolated grid
        ... The array must be monotonically increasing.
        ... If None, yi = np.arange(ymin, ymax, dy)
    dx: float
        spacing of 'xi' if 'xi' is not given
    dy: float
        spacing of 'xi' if 'xi' is not given
    nx: int
        if 'dx' were not given, dx is set as (xmax-xmin)/nx
    ny: int
        if 'dy' were not given, dx is set as (ymax-ymin)/ny
    method: method of 2D interpolation
        ... Options: 'nearest', 'linear', 'cubic'
    xmin: float, minimum x value of the interpolated grid
    xmax: float, maximum x value of the interpolated grid
    ymin: float, minimum y value of the interpolated grid
    ymax: float, maximum y value of the interpolated grid
    fill_value: float, value to fill in the interpolated grid, default is None

    Returns
    -------
    xxi: 2d array
        x-coordinate of the grid
    yyi: 2d array
        y-coordinate of the grid
    data_i: 2d array
        values on the grid
    """
    x, y, data = np.asarray(x), np.asarray(y), np.asarray(data)
    if not x.shape == y.shape == data.shape:
        print('x.shape, y.shape, and data.shape must match. ', x.shape, y.shape, data.shape)
        raise ValueError('shapes of x, y, and data do not match.')
    x, y, data1d = x.flatten(), y.flatten(), data.flatten()

    if xmin is None: xmin = np.nanmin(x)
    if xmax is None: xmax = np.nanmax(x)
    if ymin is None: ymin = np.nanmin(y)
    if ymax is None: ymax = np.nanmax(y)

    if xi is None:
        if dx is None:
            dx = (xmax - xmin) / nx
        xi = np.arange(xmin, xmax, dx)
    if yi is None:
        if dy is None:
            dy = (ymax - ymin) / ny
        yi = np.arange(ymin, ymax, dy)
    xxi, yyi = np.meshgrid(xi, yi)

    # interpolate
    data_i = griddata((x, y), data1d, (xxi, yyi), method=method, fill_value=fill_value)
    return xxi, yyi, data_i


def zoom(qty, xxx, yyy, zzz, zf=2, bounds_error=False):
    """
    A quick function to linearly interpolate 3D data by an integer factor
    ... xxx, yyy, zzz = np.meshgrid(x,y,z)
    ... Returns data and positional grids at different resolution

    Parameters
    ----------
    qty: 3d array, a regular grid (data at (x, y, z)
    xxx: 3d array, a regular grid (x component of the position)
    yyy: 3d array, a regular grid (y component of the position)
    zzz: 3d array, a regular grid (z component of the position)
    zf: float, (positive real number)
        ... a zoom factor
    bounds_error

    Returns
    -------
    qty_: 3d array, a regular grid (data at (x, y, z) at a higher resolution)
    xxx_: 3d array, a regular grid (x component of the position at a higher resolution)
    yyy_: 3d array, a regular grid (y component of the position at a higher resolution)
    zzz_: 3d array, a regular grid (z component of the position at a higher resolution)
    """
    x, y, z = xxx[0, :, 0], yyy[:, 0, 0], zzz[0, 0, :]

    if np.all(np.diff(z) < 0):
        z = np.flip(z)
        qty = np.flip(qty, axis=2)
    else:
        y, x = yy[:, 0], xx[0, :]
    if np.all(np.diff(x) < 0):
        x = np.flip(x)
        qty = np.flip(qty, axis=1)
    if np.all(np.diff(y) < 0):
        y = np.flip(y)
        qty = np.flip(qty, axis=0)

    ny, nx, nz = xxx.shape
    dx, dy, dz = get_grid_spacing(xxx, yyy, zzz)

    x_ = np.linspace(np.nanmin(x), np.nanmax(x), int(nx * zf), endpoint=True)
    y_ = np.linspace(np.nanmin(y), np.nanmax(y), int(ny * zf), endpoint=True)
    z_ = np.linspace(np.nanmin(z), np.nanmax(z), int(nz * zf), endpoint=True)

    xxx_, yyy_, zzz_ = np.meshgrid(x_, y_, z_)
    f = RegularGridInterpolator((y, x, z), qty, bounds_error=bounds_error)
    qty_ = f((yyy_, xxx_, zzz_))
    return qty_, xxx_, yyy_, zzz_


def gaussian_blur_scalar_field(field, sigma=7, mode='nearest', **kwargs):
    """
    Gaussian blur a scalar field with shape=(..., duration) such as udata
    using scipy.ndimage.filters.gaussian_filter

    Parameters
    ----------
    field: nd array with shape (dim, ..., duration)
    sigma: int, size of the gaussian kernel

    Returns
    -------
    blurred: gaussian blurred field with the same shape as the field
    """
    print('gaussian_blur_scalar_field():\n'
          '... this assumes the input array has a shape (..., duration)\n'
          '... A Gaussian kernel is applied between axis=0 and axis=-2')
    shape = field.shape
    duration = shape[-1]

    blurred = np.empty_like(field)
    for t in range(duration):
        blurred[..., t] = ndimage.filters.gaussian_filter(field[..., t], sigma=sigma, mode=mode, **kwargs)
    return blurred


def gaussian_blur_vector_field(field, sigma=7, mode='nearest', **kwargs):
    """
    Gaussian blur a scalar/vector field with shape=(dim, ..., duration) such as udata
    using scipy.ndimage.filters.gaussian_filter
    ... e.g. udata
        udata = fix_udata_shape(udata)
        gaussian_blur(udata, sigma=5)

    Parameters
    ----------
    field: nd array with shape (dim, ..., duration)
    sigma: int, size of the gaussian kernel

    Returns
    -------
    blurred: gaussian blurred field with the same shape as the field

    """
    print('gaussian_blur_vector_field():\n'
          '... this assumes the input array has a shape (dim, ..., duration)\n'
          '... A Gaussian kernel is applied to the middle stacks')
    shape = field.shape
    dim, duration = shape[0], shape[-1]

    blurred = np.empty_like(field)
    for t in range(duration):
        for d in range(dim):
            blurred[d, ..., t] = ndimage.filters.gaussian_filter(field[d, ..., t], sigma=sigma, mode=mode, **kwargs)
    return blurred


def median_filter_scaar_field(data, kernel_radius=1):
    """
    Filter input with a median filter using scipy.signal.medfilt()
    ... data must have a shape of (..., duration)
        A median filter will be applied on the stacks (span by axis=0, 1, ..., -2)
        ... The last axis is interpreted as a temporal axis of data

    Parameters
    ----------
    data: nd array
    kernel_radius: int, default: 1
        ... radius of the kernel of the median filter

    Returns
    -------
    filtered: nd array
        ... Filtered data
    """
    duration = data.shape[-1]
    filtered = np.empty_like(data)
    for t in range(duration):
        filtered[..., t] = medfilt(data[..., t], 2 * kernel_radius + 1)
    return filtered


def median_filter_vector_field(vdata, kernel_radius=1):
    """
    Filter a vector field with a median filter using scipy.signal.medfilt()
    ... data must have a shape of (dim, ..., duration)
        A median filter will be applied on the stacks (span by axis= 1, ..., -2)
        ... The last axis is interpreted as a temporal axis of data

    Parameters
    ----------
    data: nd array
    kernel_radius: int, default: 1
        ... radius of the kernel of the median filter

    Returns
    -------
    filtered: nd array
        ... Filtered data
    """
    dim, duration = vdata.shape[0], vdata.shape[-1]
    filtered = np.empty_like(vdata)
    for t in range(duration):
        for d in range(dim):
            filtered[d, ..., t] = medfilt(vdata[d, ..., t], 2 * kernel_radius + 1)
    return filtered


########## FFT tools ########
def get_window_radial(xx, yy, zz=None, wtype='hamming', rmax=None, duration=None,
                      x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                      n=500):
    """
    General method to get a window with shape (xx.shape[:], duration) or (xx.shape[:]) if duration is None
    ... Window types:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs a decay scale),
        tukey (needs taper fraction)

    Parameters
    ----------
    xx: nd array
        x-coordinate of the spatial grid of udata
    yy: nd array
        y-coordinate of the spatial grid of udata
    zz: nd array
        y-coordinate of the spatial grid of udata
    wtype: str
        name of window function such as 'hamming', 'flattop'
    rmax: float
        window function returns zero for r > rmax
    duration: int, default: None
        specifies the temporal dimension of the returning window function.
    x0: int, default: 0
        used to specify the region of the returning window function.
        When coordinates outside the specified region is given to the window function, it returns 0.
    x1: int, default: None
        used to specify the region of the returning window function.
        When coordinates outside the specified region is given to the window function, it returns 0.
    y0: int, default: 0
    y1: int, default: None
        used to specify the region of the returning window function.
        When coordinates outside the specified region is given to the window function, it returns 0.
    z0: int, default: 0
    z1: int, default: None
        used to specify the region of the returning window function.
        When coordinates outside the specified region is given to the window function, it returns 0.
    n: int, default: 500
        number of data points when 1D window function is called for the first time.
        The higher n is, the returning windowing function is more accurate.

    Returns
    -------
    window/windows: nd array
    ... window: hamming window with the shape as xx
    ... window: hamming window with the shape (xx.shape[:], duration)
    """
    # Let the center of the grid be the origin
    if zz is None:
        dim = 2
        if x1 is None:
            x1 = xx.shape[1]
        if y1 is None:
            y1 = xx.shape[0]
        origin = ((xx[0, x1 - 1] + xx[0, x0]) / 2., (yy[y1 - 1, 0] + yy[y0, 0]) / 2.)
        xx, yy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        rr = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2)
    else:
        dim = 3
        if x1 is None:
            x1 = xx.shape[1]
        if y1 is None:
            y1 = xx.shape[0]
        if z1 is None:
            z1 = xx.shape[2]

        origin = (
            (xx[0, x1 - 1] + xx[0, x0]) / 2., (yy[y1 - 1, 0] + yy[y0, 0]) / 2., (zz[0, 0, z1 - 1] + zz[0, 0, z0]) / 2.)
        xx, yy, zz = xx[y0:y1, x0:x1, z0:z1], yy[y0:y1, x0:x1, z0:z1], zz[y0:y1, x0:x1, z0:z1]
        rr = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 + (zz - origin[2]) ** 2)

    if rmax is None:
        xmax, ymax = np.nanmax(xx[0, :]), np.nanmax(yy[:, 0])
        rmax = min(xmax, ymax)
    if wtype is None:
        window = np.ones_like(rr)
    else:
        r = np.linspace(-rmax, rmax, n)
        window_1d = signal.get_window(wtype, n)
        window_func = interpolate.interp1d(r, window_1d, bounds_error=False, fill_value=0)
        window = window_func(rr)
        window[rr > rmax] = 0
    if duration is not None:
        windows = np.repeat(window[..., np.newaxis], duration, axis=dim)
        return windows
    else:
        return window


def compute_signal_loss_due_to_windowing(xx, yy, window, x0=0, x1=None, y0=0, y1=None):
    """
    Returns the inverse of the signal-loss factor by the window
    Signal loss factor: 1 (rectangle... no loss), 0.2-ish (flattop)

    Parameters
    ----------
    xx: 2d/3d numpy array
        x grid
    yy: 2d/3d numpy array
        y grid
    window: str
        name of the window: flattop, hamming, blackman, etc.
    x0: int
        index to specify the region to which the window applies: xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
    x1: int
        index to specify the region to which the window applies: xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
    y0: int
        index to specify the region to which the window applies: xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
    y1: int
        index to specify the region to which the window applies: xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]

    Returns
    -------
    gamma: float
        inverse of the signal-loss factor

    """
    if window == 'rectangle' or window is None:
        signal_intensity_loss = 1.
    else:
        window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
        signal_intensity_loss = np.nanmean(window_arr)
    gamma = 1. / signal_intensity_loss
    return gamma


def get_hamming_window_radial(xx, yy, zz=None, rmax=None, duration=None,
                              x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    Returns the Hamming window as a function of radius
     ... r = 0 corresponds to the center of the window.

    Parameters
    ----------
    r: nd array
    ... radius from the center point
    rmax: float
    ...
    duration: int
    ... if None, this returns the hamming window with shape (height, width),  (height, width, depth)
    ... Otherwise, this returns (height, width, duration),  (height, width, depth, duration)
    ... duration = udata.shape[-1] should work for most of purposes.

    Returns
    -------
    window/windows: nd array
    ... window: hamming window with the shape as xx
    ... window: hamming window with the shape (xx.shape[:], duration)

    """
    # Let the center of the grid be the origin
    if zz is None:
        dim = 2
        if x1 is None:
            x1 = xx.shape[1]
        if y1 is None:
            y1 = xx.shape[0]
        origin = ((xx[0, x1 - 1] - xx[0, x0]) / 2., (yy[y1 - 1, 0] - yy[y0, 0]) / 2.)
        xx, yy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        rr = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2)
    else:
        dim = 3
        if x1 is None:
            x1 = xx.shape[1]
        if y1 is None:
            y1 = xx.shape[0]
        if z1 is None:
            z1 = xx.shape[2]

        origin = ((xx[0, x1 - 1] - xx[0, x0]) / 2., (yy[y1 - 1, 0] - yy[y0, 0]) / 2.)
        xx, yy, zz = xx[y0:y1, x0:x1, z0:z1], yy[y0:y1, x0:x1, z0:z1], zz[y0:y1, x0:x1, z0:z1]
        rr = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 + (zz - origin[2]) ** 2)

    if rmax is None:
        rmax = np.nanmax(rr)

    x = rr + rmax
    window = 0.54 - 0.46 * np.cos(2 * np.pi * (2 * rmax - x) / rmax / 2.)
    window[rr > rmax] = 0
    if duration is not None:
        windows = np.repeat(window[..., np.newaxis], duration, axis=dim)
        return windows
    else:
        return window


# cleaning velocity field data
def clean_udata_cheap(udata, cutoffU=2000, fill_value=np.nan, verbose=True):
    """
    ONLY WORKS FOR THE 2D data
    Conducts a cheap bilinear interpolation for missing data.
    ... literally, computes the average of the values interpolated in the x- and y-directions
    ... griddata performs a better interpolation but this method is much faster but not necessarily accurate.
    ... values near the edges must not be trusted.
    
    Parameters
    ----------
    udata
    cutoffU
    fill_value
    verbose

    Returns
    -------

    """

    def interpolate_using_mask(arr, mask):
        """
        Conduct linear interpolation for data points where their mask values are True

        ... This interpolation is not ideal because this flattens multidimensional array first, and takes a linear interpolation
        for missing values. That is, the interpolated values at the edges of the multidimensional array are nonsense b/c
        actual data does not have a periodic boundary condition.

        Parameters
        ----------
        arr1 : array-like (n x m), float
            array with unphysical values such as nan, inf, and ridiculously large/small values
            assuming arr1 is your raw data
        mask : array-like (n x m), bool

        Returns
        -------
        arr : array-like (n x m), float
            array with unphysical values replaced by appropriate values
        """
        arr1 = copy.deepcopy(arr)
        arr2T = copy.deepcopy(arr).T

        f0 = np.flatnonzero(mask)
        f1 = np.flatnonzero(~mask)

        arr1[mask] = np.interp(f0, f1, arr1[~mask])

        f0 = np.flatnonzero(mask.T)
        f1 = np.flatnonzero(~mask.T)
        arr2T[mask.T] = np.interp(f0, f1, arr1.T[~(mask.T)])
        arr2 = arr2T.T

        arr = (arr1 + arr2) * 0.5
        return arr

    udata_cleaned = np.empty_like(udata)
    print('Cleaning ux...')
    mask = get_mask_for_unphysical(udata[0, ...], cutoffU=cutoffU, fill_value=fill_value, verbose=verbose)
    Ux_filled_with_nans = fill_unphysical_with_sth(udata[0, ...], mask, fill_value=fill_value)
    Ux_interpolated = interpolate_using_mask(Ux_filled_with_nans, mask)
    udata_cleaned[0, ...] = Ux_interpolated[:]
    print('Cleaning uy...')
    mask = get_mask_for_unphysical(udata[1, ...], cutoffU=cutoffU, fill_value=fill_value, verbose=verbose)
    Uy_filled_with_nans = fill_unphysical_with_sth(udata[1, ...], mask, fill_value=fill_value)
    Uy_interpolated = interpolate_using_mask(Uy_filled_with_nans, mask)
    udata_cleaned[1, ...] = Uy_interpolated[:]
    print('...Cleaning Done.')
    return udata_cleaned


def get_mask_for_unphysical(U, cutoffU=2000., fill_value=99999., verbose=True):
    """
    Returns a mask (N-dim boolean array). If elements were below/above a cutoff, np.nan, or np.inf, then they get masked.

    Parameters
    ----------
    U: array-like, a velocity field (udata)
    cutoffU: float
        if |value| > cutoff, this function considers such values unphysical.
    fill_value: float, default=99999.

    Returns
    -------
    mask: nd boolean array

    """
    U = np.asarray(U)
    if verbose:
        print('...Note that nan/inf values in U are replaced by ' + str(fill_value))
        print('...number of invalid values (nan and inf) in the array: ' + str(np.isnan(U).sum() + np.isinf(U).sum()))
        print('...number of nan values in U: ' + str(np.isnan(U).sum()))
        print('...number of inf values in U: ' + str(np.isinf(U).sum()) + '\n')

    # Replace all nan and inf values with fill_value.
    U_fixed = ma.fix_invalid(U, fill_value=fill_value)
    n_invalid = ma.count_masked(U_fixed)
    if verbose:
        print('...number of masked elements by masked_invalid: ' + str(n_invalid))
    # Update the mask to False (no masking)
    U_fixed.mask = False

    # Mask unreasonable values of U_fixed
    a = ma.masked_invalid(U_fixed)
    b = ma.masked_greater(U_fixed, cutoffU)
    c = ma.masked_less(U_fixed, -cutoffU)
    n_greater = ma.count_masked(b)
    n_less = ma.count_masked(c)
    if verbose:
        print('...number of masked elements greater than cutoff: ' + str(n_greater))
        print('...number of masked elements less than -cutoff: ' + str(n_less))

    # Generate a mask for all nonsense values in the array U
    mask = ~(~a.mask * ~b.mask * ~c.mask)

    d = ma.array(U_fixed, mask=mask)
    n_total = ma.count_masked(d)
    # U_filled = ma.filled(d, fill_value)

    # Total number of elements in U
    N = U.size
    if verbose:
        print('...total number of unphysical values: ' + str(ma.count_masked(d)) + '  (' + str(
            (float(n_total) / N * 100)) + '%)\n')
    return mask


def fill_unphysical_with_sth(U, mask, fill_value=np.nan):
    """
    Returns an array whose elements are replaced by fill_value if its mask value is True

    Parameters
    ----------
    U: array-like
    mask: multidimensional boolean array
    fill_value: value that replaces masked values

    Returns
    -------
    U_filled: numpy array, same shape as U, with masked values replaced by fill_value
    """
    U_masked = ma.array(U, mask=mask)
    U_filled = ma.filled(U_masked, fill_value)  # numpy array. This is NOT a masked array.
    return U_filled


def clean_udata(udata,
                mask=None,
                method='nn', max_iter=10, tol=0.05, kernel_radius=2, kernel_sigma=2,
                fill_value=0,  # for method=='fill'
                cutoff=np.inf,
                median_filter=True,
                replace_zeros=True,
                showtqdm=True, verbose=False, notebook=True, make_copy=True):
    """
    Cleans up the velocity field udata.
    ... 1. Thresholding to spot unphysical values
    ... 2. Mask unphysical values, np.nan, and np.inf
    ... 3. ND interpolation over space (but NOT time) using direct convolution (replace_nan(...))
    ...... No interpolation over time is done because udata at two different frames do not have to be related in principle.
    ... 4. Median filter (optional) with kernel size = 2 * kernel_radius + 1

    Parameters
    ----------
    udata: nd array, velocity field data with shape (no of components, y, x, (z), t) or  (no of components, y, x, (z))
    mask: nd array, same shape as udata, default: None
        ... If it were not None, this function interpolates the
    method: str- options are 'fill', 'nn', 'local mean', 'idw'
        ... 'fill' (fill by a constant "fill_value")
        ... 'nn' (nearest neighbor filling),
        ... 'local mean', relevant arg: kernel_radius=2
        ... 'idw' (convolution with a Gaussian kernel, relevant arg: "kernel_sigma")
    max_iter: int, default: 10
        ... Number of iterations for inpainting until tolerance is reached
    tol: float, default: 0.05, tolerance for inpainting
    kernel_radius: int, default: 2, radius of the kernel for inpainting and median filter
    kernel_sigma: float, default: 2, sigma of the Gaussian kernel for inpainting if method == 'idw
    fill_value: float, If method=='fill', the errorneous valeus are replaced by this fill_value.
    cutoff: float, a positive real number- Any velocity vector whose norm is greater than this cutoff is subject to cleaning.
    median_filter: bool, default: True. If True, a median filter is applied to the velocity field after inpainting/nearest-neighbor filling
    replace_zeros: bool default: True
        ... Some softwares assign 0 as a velocity instead of na or np.nan
    showtqdm: bool, default: True
        ... If True, show tqdm for the processes: 1. removing np.nan 2. applying a median filter
    verbose: bool, default: False
        ... If True, it shows a tqdm progress bar for the iterations of the direct convolution method
    notebook: bool, If True, it uses tqdm_notebook instead of tqdm
    make_copy: bool, default: True
        ... If True, it creates a copy of the given udata, then modify the copied array.
            Without copying, it modifies the given array directly. (Why?- it should create a new copy

    Returns
    -------
    udata_i: nd array, same shape as udatawith unphysical values replaced
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if make_copy:
        udata_c = copy.deepcopy(udata)  # make a deepcopy otherwise the passed udata would be modified
    else:
        udata_c = udata
    udata_c = fix_udata_shape(udata_c)
    dim = udata_c.shape[0]

    if dim == 2:
        ncomp, height, width, duration = udata_c.shape
    else:
        ncomp, height, width, depth, duration = udata_c.shape

    if mask is not None:
        udata_c[mask] = np.nan
    # manual cleaning
    udata_c[np.logical_or(np.abs(udata_c) > cutoff, np.isinf(udata_c))] = np.nan
    if replace_zeros:
        udata_c[udata_c == 0] = np.nan

    nnans = np.count_nonzero(np.isnan(udata_c))
    if nnans == 0 and not median_filter:  # udata is already clean (No nan values and no values beyond cutoff)
        return udata_c

    # Method 1
    if method == 'nn':  # nearest neighbor filling
        udata_i = replace_nan_w_nn_udata(udata_c, notebook=notebook, showtqdm=showtqdm)
    elif method == 'fill':  # fill nans by a constant "fill_value"
        udata_i = copy.deepcopy(udata_c)
        udata_i[np.isnan(udata_i)] = fill_value
    else:  # inpainting
        # Initialization of an inpainted field
        udata_i = np.empty_like(udata_c)
        for t in tqdm(range(duration), desc='replacing nans', disable=not showtqdm):
            for i in range(ncomp):
                udata_i[i, ..., t] = replace_nans(udata_c[i, ..., t], max_iter=max_iter, tol=tol,
                                                  kernel_radius=kernel_radius,
                                                  kernel_sigma=kernel_sigma, method=method, showtqdm=verbose)
    if median_filter:
        for t in tqdm(range(duration), desc='median filter', disable=not showtqdm):
            for i in range(ncomp):
                udata_i[i, ..., t] = medfilt(udata_i[i, ..., t], 2 * kernel_radius + 1)

    if notebook:
        from tqdm import tqdm as tqdm
    udata_i = np.squeeze(udata_i) # remove singleton dimensions
    return udata_i


def find_crop_no(udata, tol=0.2):
    """
    Returns a how many pixels should be cropped from the edge if udata contains many nan values near the edge.

    Parameters
    ----------
    udata: nd array, velocity field, shape (ncomp, height, width, (depth),(duration))
    tol: float, default: 0.2
        ... The tolerance for the percentage of nan values near the edge

    Returns
    -------
    i: int, how many pixels should be cropped from the edge

    Examples:
    --------
    udata = fix_udata_shape(udata)
    n = find_crop_no(udata, tol=0.2)
    udata_cropped = udata[:, n:-n, n:-n, ...]
    """

    def count_nans(arr):
        """Returns the number of nans in the given array"""
        nnans = np.count_nonzero(np.isnan(arr))
        return nnans

    udata = fix_udata_shape(udata)
    dim = udata.shape[0]
    print(udata.shape, udata.size)

    if dim == 2:
        ncomp, height, width, duration = udata.shape
    else:
        ncomp, height, width, depth, duration = udata.shape

    # Find the optimal number of pixels that can be cropped off from the data
    n = np.floor(min([height, width, depth]) / 2).astype(int)

    for i in range(1, n):
        if dim == 3:
            nnans = count_nans(udata[:, i:-i, i:-i, i:-i, :])
            ntot = udata[:, i:-i, i:-i, i:-i, :].size
        else:
            nnans = count_nans(udata[:, i:-i, i:-i, :])
            ntot = udata[:, i:-i, i:-i, :].size
        ratio = nnans / ntot
        if ratio < tol:
            break
    print('No of edges to remove in indices: %d' % i, ratio, nnans, ntot)
    return i


# Dealing with missing values in a field
def replace_nan_w_nn(data, notebook=False):
    """
    Nearest neighbor filling for nans in an array
    ... Nearest neighbor filling of ND data
    ... Do not use this function to interpolate udata! Use replace_nan_w_nn_udata(udata) instead.
    ... This is not designed for udata because NN filling for ND data treats all axes equally. Since individual axis of
    udata means (velocity components, height, width, depth, duration), applying this function to udata requires care.
    ... NN filling occurs from the right to left.
        i.e. array[n-3, n-2, n-1, n, n+1] = 100, np.nan, np.nan, np.nan, 200
            -> 100, np.nan, np.nan, 200, 200 # Step1: Fill from the right first
            -> 100, 100, np.nan, 200, 200 # Step2: Then, fill from the left
             -> 100, 100, 200, 200, 200  # Repeat Step1: Fill from the right

    Parameters
    ----------
    data: ND data with missing values (np.nan)
    notebook: bool, default: False
        ... If True, use tqdm_notebook instead of tqdm for a progress bar

    Returns
    -------
    data: ND data, missing values are filled with the nearest neighbor values.
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    isnan = np.isnan(data)

    if not isnan.any():
        # if verbose:
        print('replace_nan_w_nn: no nan in data')
        if notebook:
            from tqdm import tqdm as tqdm
        return data
    else:
        while isnan.any():
            dim = len(data.shape)
            for i in range(dim):
                slices = [slice(None)] * dim  # Initialize
                slices1 = slices[:]
                slices2 = slices[:]
                slices1[i] = slice(0, -1)
                slices2[i] = slice(1, None)

                # replace from the right (from the bottom)
                fill = np.logical_and(isnan[slices1], ~isnan[slices2])
                data[slices1][fill] = data[slices2][fill]
                isnan[slices1][fill] = False

                # replace from the left (from the top)
                fill = np.logical_and(isnan[slices2], ~isnan[slices1])
                data[slices2][fill] = data[slices1][fill]
                isnan[slices2][fill] = False
            isnan = np.isnan(data)

        if notebook:
            from tqdm import tqdm as tqdm
        return data


def replace_nan_w_nn_udata(udata, notebook=True, showtqdm=True):
    """
    Nearest neighbor filling for nans in udata
    The fastest and most robust function to fill the missing values in udata

    Parameters
    ----------
    udata: ndarray, velocity field, shape (dim, height, width, depth, duration)
    notebook: bool, default: False
        ... If True, use tqdm_notebook instead of tqdm for a progress bar
    showtqdm: bool, default: True
        ... If True, show tqdm progress bar.

    Returns
    -------
    udata_filled: ND array with shape (dim, height, width, depth, duration) or  (dim, height, width, duration)
            ... missing values (np.nan) are filled with the nearest neighbors
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]

    udata_filled = np.empty_like(udata)
    for t in tqdm(range(duration), desc='replace_nan_w_nn_udata', disable=not showtqdm):
        for i in range(dim):
            udata_filled[i, ..., t] = replace_nan_w_nn(udata[i, ..., t], notebook=notebook)

    if notebook:
        from tqdm import tqdm as tqdm

    return udata_filled


def replace_nans(array, max_iter=10, tol=0.05, kernel_radius=2, kernel_sigma=2, method='nn',
                 notebook=True, verbose=False, showtqdm=True):
    """
    Replace NaN elements in an array using an iterative inpainting algorithm.
    The algorithm is the following:
    1) For each element in the input array, replace it by a weighted average
    of the neighbouring elements which are not NaN themselves. The weights depends
    of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
    2) Several iterations are needed if there are adjacent NaN elements.
    If this is the case, information is "spread" from the edges of the missing
    regions iteratively, until the variation is below a certain threshold.

    - Generalized to nD array by Takumi Matsuzawa (UChicago) 2020/02/20

    Parameters
    ----------
    array : nd np.ndarray
    an array containing NaN elements that have to be replaced
    max_iter : int, number of iterations
    kernel_radius : int, size of the kernel: 2 * kernel_radius + 1
    kernel_sigma : float, sigma of the gaussian kernel
    method : str
        ... the method used to replace invalid values. Valid options are
            1. nearest neighbor filling (method = "nn")- fast
            2. filling with some constant (method = "fill")- fast
            3. filling with local mean (method = "localmean")- slow
            4. filling with an idw kernel (method = "idw") - slow

    Returns
    -------
    filled : nd np.ndarray, input array with its NaN elements being replaced by some numbers
    """

    def makeGaussianKernel(size, sigma, dim):
        x = np.arange(size)
        xi = tuple([x for i in range(dim)])
        grids = np.meshgrid(*xi)
        pos = np.empty(grids[0].shape + (dim,))
        for i in range(dim):
            pos[..., i] = grids[i]

        # Get Gaussian distribution
        mean = np.ones(dim) * size / 2
        cov = []
        for i in range(dim):
            cov_tmp = []
            for j in range(dim):
                if j == i:
                    cov_tmp.append(sigma)
                else:
                    cov_tmp.append(0)
            cov.append(cov_tmp)
        rv = multivariate_normal(mean=mean, cov=cov)
        gkernel = rv.pdf(pos) / np.sum(rv.pdf(pos))
        return gkernel

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    array = np.asarray(array)
    dim = len(array.shape)

    # fill new array with input elements
    filled = copy.deepcopy(array)

    # make kernel
    kernel_size = kernel_radius * 2 + 1
    kernel_shape = ()
    for i in range(dim):
        kernel_shape += (2 * kernel_size + 1,)
    kernel = np.empty(kernel_shape)

    # indices where array is NaN
    ind_nans = np.asarray(np.nonzero(np.isnan(array)))

    # number of NaN elements
    n_nans = len(ind_nans[0])

    # arrays which contain replaced values to check for convergence
    replaced_new = np.zeros(n_nans)
    replaced_old = np.zeros(n_nans)

    # Make a kernel
    if method == 'localmean':
        #         print('kernel_size', kernel_size)
        kernel = np.ones(kernel_shape)
    elif method == 'idw':  # inverse distance weighting (with gaussian kernel)
        kernel = makeGaussianKernel(kernel_size, kernel_sigma, dim)
    #         print(kernel.shape, 'kernel')
    elif method == 'nn':  # newarest neighbor
        filled = replace_nan_w_nn(array, notebook=notebook)
    else:
        raise ValueError('method is not valid. Choose from idw and localmean')

    # make several passes
    # until we reach convergence
    for it in tqdm(range(max_iter), desc='Loop until it gets below tolerance', disable=not showtqdm):
        # for each NaN element
        for p in tqdm(range(n_nans), desc='looping over missing values', disable=not showtqdm):
            nan_indices = tuple(ind_nans[:, p])

            # initialize nan values to zero
            filled[nan_indices] = 0.0
            n = 0

            # loop over the kernel
            for kindices in itertools.product(range(kernel_size), repeat=dim):
                # Check if a kernel includes region outside the boundaries
                isInsideBoundary = True
                for q in range(dim):
                    index = nan_indices[q] + kindices[q] - kernel_radius
                    isInsideBoundary *= index < array.shape[q] and index >= 0
                # if the element at (indices) were inside the boundary,
                # convolve the original array with the kernel
                if isInsideBoundary:
                    indices = ()
                    for q in range(dim):
                        index = nan_indices[q] + kindices[q] - kernel_radius
                        indices += (index,)
                    # Convolute the array with the kernel if the array element is not nan
                    if not np.isnan(filled[indices]):
                        # Don't sum itself
                        if not all([kindices[q] - kernel_radius == 0 for q in range(dim)]):
                            # convolve kernel with original array
                            filled[nan_indices] = filled[nan_indices] + filled[indices] * kernel[kindices]
                            n = n + 1 * kernel[kindices]

            # divide value by effective number of added elements
            if n != 0:
                filled[nan_indices] = filled[nan_indices] / n
            else:
                filled[nan_indices] = np.nan
            replaced_new[p] = filled[nan_indices]

        # check if mean square difference between values of replaced elements is below a threshold
        if verbose:
            print('convergence check: ', np.nanmean((replaced_new - replaced_old) ** 2))
        variance = np.nanmean((replaced_new - replaced_old) ** 2)
        if variance < tol:
            break
        else:
            if it == max_iter - 1 and not np.isnan(variance):
                print('... replace_nan: did not converge within %d iterations.\n \
                      Refining max_int and tol. is recommended' % it)
                print('variance: ', variance)
            replaced_old = copy.deepcopy(replaced_new)

    if notebook:
        from tqdm import tqdm as tqdm

    return filled


def clean_data(data_org, mask=None,
               method='nn', max_iter=5, tol=0.05, kernel_radius=2, kernel_sigma=2,
               replace_val=None,
               cutoff=np.inf, fill_value=0, verbose=False, notebook=True, makecopy=True,
               median_filter=True, showtqdm=True):
    """
    This fills missing values (np.nan) in an array by one of the listed procedures below.
        The options are
        1. nearest neighbor filling (method = "nn")- fast
        2. filling with some constant (method = "fill")- fast
        3. filling with local mean (method = "localmean")- slow
        4. filling with an idw kernel (method = "idw") - slow

    Parameters
    ----------
    data_org: nd array, array to be cleaned
    mask: nd array
    method: str, choose from 'fill', 'nn', 'localmean', 'idw'
        ... computation is more expensive as it gets more right.
        ... fill: it filles
    max_iter: int, number of iterations conducted for the direct convolution methods
        ... When the method uses direct convolution, it repeats the operation until certain smoothness is achieved.
        ... relevant parameter for method==localmean or idw'
    tol: relevant parameter for method==localmean or idw'
    kernel_radius: float, size of the kernel is 2 * kernel_radius + 1, relevant parameter for method==localmean'
    kernel_sigma: float, std of a gaussian kernel, relevant parameter for method==idw'
    replace_val: float, default: None. If not None, it replaces the values of the array with np.nan if value=replac_val.
        ... This feature could be useful if the scalar data assigns a specific value instead of np.nans.
        ... This is certainly the case whenever DaVis outputs pressure. It outputs 0 instead of nan. This causes a problem as
        0 could be physically meaningful while DaVis assigned that value solely because STB did not have sufficient tracks to
         infer the actual (intrinsic) value. In this case, simply replace all 0s with np.nan, then replace nans with nn, localmean, etc.
    cutoff: float, default: np.inf. If not np.inf, it will invalidate values above cutoff.
    fill_value: float, default: 0. If not 0, it will replace values above cutoff with fill_value
        ... relevant parameter for method=='fill'
    verbose: bool, default: False. If True, it will print out the mean square difference between iterations
    notebook: bool, default: True. If True, it will use tqdm_notebook instead of tqdm to show progress bar

    Returns
    -------
    data: nd array with the same shape as data_org
    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if makecopy:
        data = copy.deepcopy(data_org)  # without copying, the original data also gets modified

    if mask is not None:
        data[mask] = np.nan
    # manual cleaning
    data[np.logical_or(np.abs(data) > cutoff, np.isinf(data))] = np.nan
    if replace_val is not None:
        data[data == replace_val] = np.nan

    nnans = np.count_nonzero(np.isnan(data))
    if nnans == 0:  # udata is already clean (No nan values and no values beyond cutoff)
        return data

    if method == 'nn':  # nearest neighbor filling
        data = replace_nan_w_nn(data, notebook=notebook)
    elif method == 'fill':  # fill nans by a constant "fill_value"
        data[np.isnan(data)] = fill_value
    else:
        # Initialization of an inpainted field
        data = replace_nans(data, max_iter=max_iter, tol=tol, kernel_radius=kernel_radius,
                            kernel_sigma=kernel_sigma, method=method, showtqdm=verbose)

    if median_filter:
        for t in tqdm(range(data.shape[-1]), desc='median filter', disable=not showtqdm):
            data[..., t] = medfilt(data[..., t], 2 * kernel_radius + 1)

    if notebook:
        from tqdm import tqdm as tqdm

    return data


def clean_data_interp1d(y, x=None, thd=None, p=0.98):
    """
    This function interpolates the missing and erroneous data using 1D interpolation.
    ... errorneous data points are identified by the rate of change of the data.
    
    Parameters
    ----------
    y: 1/2d array with shape (n,) or (n, duration)
    x: 1/2d array with shape (n,) or (n, duration) (optional)
        ... MUST have the same shape as y
    thd: float
        ... threshold on the slope (dy/dx) used to spot the spurious values
        ... If np.abs( dy/dx )  / y) < thd, the points in y will be removed, and will be interpolated as well as np.nan in y.
    p: float, 0 < p < 1
        ... if thd is None, it will remove (1-p) of the data points in y, and replaces with the interpolated values

    Returns:
    ----------
    x_output: nd array with shape (n, 1) or (n, duration)
        ... new x for new y
    y_output: nd array with shape (n, 1) or (n, duration)
        ... interpolated y
    """
    y = np.asarray(y)
    if len(y.shape) == 1:
        shape = y.shape
        y = y.reshape(shape + (1,))
    duration = y.shape[1]

    if x is None:
        x = np.arange(len(y))
        # x = x.reshape(x.shape + (1,))
        x = np.tile(x[:, np.newaxis], duration)
    else:
        x = np.asarray(x)
        if len(x.shape) == 1:
            # shape = x.shape
            # x = x.reshape(shape + (1,))
            x = np.tile(x[:, np.newaxis], duration)

    x_output, y_output = np.empty_like(x), np.empty_like(y)
    for t in range(duration):
        y_tmp = y[:, t]
        x_tmp = x[:, t]

        keep_x_0, keep_y_0 = ~np.isnan(x_tmp), ~np.isnan(y_tmp)
        keep_x_1, keep_y_1 = ~np.isinf(x_tmp), ~np.isinf(y_tmp)
        keep = keep_x_0 * keep_y_0 * keep_x_1 * keep_y_1
        x_keep, y_keep = x_tmp[keep], y_tmp[keep]

        fractional_dydx = np.gradient(y_keep, x_keep) / y_keep
        print(fractional_dydx)
        if thd is None:
            zeta = sorted(np.abs(fractional_dydx))
            n_zeta = len(zeta)
            try:
                thd = zeta[int(n_zeta * p)]
            except:
                thd = zeta[-1]
        mask = np.abs(fractional_dydx) < thd
        mask = np.roll(mask, 1)  # shift the resulting array (due to the convention of np.gradient)

        x_clean, y_clean = x_keep[mask], y_keep[mask]
        f = interpolate.interp1d(x_clean, y_clean)

        xmin, xmax = np.nanmin(x_clean), np.nanmax(x_clean)
        new_x = np.linspace(xmin, xmax, len(x_tmp))
        y_interpolated = f(new_x)
        x_output[:, t] = new_x
        y_output[:, t] = y_interpolated
    return x_output, y_output


def get_instantaneous_center_of_energy(udata, xx, yy, zz=None,
                                       x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    This function calculates the center of energy from the instantaneous energy field.
    
    Parameters
    ----------
    udata: nd array with shape (dim, width, height, (depth), (duration))
    xx: nd array with shape (width, height, (depth))
    yy: nd array with shape (width, height, (depth))
    zz: nd array with shape (width, height, (depth)) (optional)
    x0: int, index used to define the x-range of the v-field: udata[:, y0:y1, x0:x1, z0:z1, ...]
    x1: int, index used to define the x-range of the v-field: udata[:, y0:y1, x0:x1, z0:z1, ...]
    y0: int, index used to define the x-range of the v-field: udata[:, y0:y1, x0:x1, z0:z1, ...]
    y1: int, index used to define the x-range of the v-field: udata[:, y0:y1, x0:x1, z0:z1, ...]
    z0: int, index used to define the x-range of the v-field: udata[:, y0:y1, x0:x1, z0:z1, ...]
    z1: int, index used to define the x-range of the v-field: udata[:, y0:y1, x0:x1, z0:z1, ...]

    Returns
    -------
    centers: nd array with shape (dim, duration), center of energy for each instantenous energy field
        ... centers[0, t] = x-coordinate of the center of energy
        ... centers[1, t] = y-coordinate of the center of energy
    """
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]
    energy = get_energy(udata)

    centers = np.empty((dim, duration))
    coords = [xx, yy, zz]
    for t in range(duration):
        for d in range(dim):
            centers[d, t] = np.nansum(coords[d] * energy[..., t]) / np.nansum(energy[..., t])
    return centers


# FUNCTIONS FOR STB DATA ANALYSIS
def get_center_of_energy(dpath, inc=10, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    Returns the center of energy (essentially the center of the blob)- (xc, yc, zc)
    using the TIME-AVERAGED ENERGY FIELD.
    - It computes the time-averaged energy from udatapath
    - Then, it computes the center of energy like the center of mass.
    (i.e. the energy density serves as a weight function)

    Parameters
    ----------
    dpath: str, path where velocity field data is stored
    inc: int, it uses every inc-th frame of the data to compute the center of energy
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])

    Returns
    -------
    center_of_energy: numpy array, center of energy for the time-avg-energy field
    """
    # dummy
    udata = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=0, t1=1,
                                return_xy=False, verbose=False)
    dim = udata.shape[0]
    if dim == 3:
        # Load dummy udata
        udata, xx, yy, zz = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=0, t1=1,
                                                return_xy=True, verbose=False)
        etavg = get_time_avg_energy_from_udatapath(dpath, inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        xc = np.nansum(etavg * xx) / np.nansum(etavg)
        yc = np.nansum(etavg * yy) / np.nansum(etavg)
        zc = np.nansum(etavg * zz) / np.nansum(etavg)
        center_of_energy = np.asarray([xc, yc, zc])
    else:
        # Load dummy udata
        udata, xx, yy = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, t0=0, t1=1,
                                            return_xy=True, verbose=False)
        etavg = get_time_avg_energy_from_udatapath(dpath, inc=inc, x0=x0, x1=x1, y0=y0, y1=y1)
        xc = np.nansum(etavg * xx) / np.nansum(etavg)
        yc = np.nansum(etavg * yy) / np.nansum(etavg)
        center_of_energy = np.asarray([xc, yc])
    return center_of_energy


def get_center_of_vorticity(udata, xx, yy, sigma=5, sign='auto',
                            x0=0, x1=None,
                            y0=0, y1=None,
                            thd=0
                            ):
    """
    Get the center of vorticity field
    ... This computes the normalized, first moment of the vorticity field.
    ... The center of field is not well-defined if values can be positive and negative
    ...... To avoid this, the sign of the vorticity field must be specified.
    ...... If sign='auto', the sign is automatically determined based on the average vorticity in the field.
    ...... Otherwise, the sign must be specified as 'positive' or 'negative'
    ...... The opposite vorticity will be considered to be zero.

    Parameters
    ----------
    udata: nd array, velocity field
    xx: nd array, x-coordinates
    yy: nd array, y-coordinates
    sigma: int, default: 5, std of the gaussian kernel that is passed to ndimage.gaussian_filter()
    sign: str, default: 'auto', 'positive' or 'negative'
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    thd: float, default: 0
        ... |vorticity| < thd will be considered to be zero

    Returns
    -------
    centers: nd array with shape (dim, duration), center of energy for each instantenous vorticity field
        ... centers[0, t] = x-coordinate of the center of energy
        ... centers[1, t] = y-coordinate of the center of energy
    """
    dim = 2
    dx, dy = get_grid_spacing(xx, yy)
    omega = curl(udata, dx, dy, xx=xx, yy=yy)
    duration = udata.shape[-1]
    centers = np.empty((dim, duration))

    if sign == 'auto':
        if np.nanmean(omega) < 0:
            sign = 'negative'
        else:
            sign = 'positive'

    for t in tqdm(range(duration)):
        omega_blurred = ndimage.gaussian_filter(omega[..., t], sigma=sigma)
        if sign == 'positive':
            omega_blurred[omega_blurred < 0] = 0
        else:
            omega_blurred[omega_blurred >= 0] = 0

        # vorticity field contains huge noise due to differentiation
        # ... Set field values to be zero below a noise threshold
        omega_blurred[np.abs(omega_blurred) < thd] = 0
        centers[0, t] = np.nansum(xx[y0:y1, x0:x1] * omega_blurred[y0:y1, x0:x1]) / np.nansum(
            omega_blurred[y0:y1, x0:x1])
        centers[1, t] = np.nansum(yy[y0:y1, x0:x1] * omega_blurred[y0:y1, x0:x1]) / np.nansum(
            omega_blurred[y0:y1, x0:x1])

    return centers


# plotting usual stuff # DEPENDENCY: graph module.
def plot_energy_spectra(udata, dx, dy, dz=None, x0=0, x1=None, y0=0, y1=None, window='flattop', epsilon_guess=10 ** 5,
                        nu=1.004, label='',
                        plot_e22=False, plot_ek=False, fignum=1, t0=0, legend=True, loc=3):
    """
    A method to quickly plot the 1D energy spectra

    Parameters
    ----------
    udata: nd array, velocity field,
    dx: float, grid spacing in x-direction
    dy: float, grid spacing in y-direction
    dz: float, grid spacing in z-direction
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    window: str, default: 'flattop', window function to be used in the FFT, see scipy.signal.get_window() for more options
    epsilon_guess: float, default: 10 ** 5, estimate of the dissipation rate of the flow
    nu: float, default: 1.004, kinematic viscosity of the fluid
    label: str, default: '', label of the plot (e.g. 'Data on MM/DD/YY')
    plot_e22: bool, default: False, if True, plot the E22 energy spectrum
    plot_ek: bool, default: False, if True, plot the 3D energy spectrum
    fignum: int, default: 1, figure number
    t0: int, default: 0, index of the time to specify the instantaneous v-field
    legend: bool, default: True, if True, plot legend
    loc: int, default: 3, location of the legend

    Returns
    -------
    fig: matplotlib.figure.Figure, figure object
    axes: tuple of two matplotlib.axes._subplots.AxesSubplot objects
    """
    __fontsize__ = 25
    __figsize__ = (16, 8)
    # See all available arguments in matplotlibrc
    params = {'figure.figsize': __figsize__,
              'font.size': __fontsize__,  # text
              'legend.fontsize': 18,  # legend
              'axes.labelsize': __fontsize__,  # axes
              'axes.titlesize': __fontsize__,
              'xtick.labelsize': __fontsize__,  # tick
              'ytick.labelsize': __fontsize__,
              'lines.linewidth': 8,
              'axes.titlepad': 10}
    graph.update_figure_params(params)

    ax1_ylabel = '$E_{11}$ ($mm^3/s^2$)'
    ax2_ylabel = '$E_{11} / (\epsilon\\nu^5)^{1/4}$'

    eiis, err, k11 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    if epsilon_guess is not None:
        e11_s, k11_s = scale_energy_spectrum(eiis[0, ...], k11, epsilon=epsilon_guess, nu=nu)
        e22_s, k22_s = scale_energy_spectrum(eiis[1, ...], k11, epsilon=epsilon_guess, nu=nu)
        eiis_s = np.stack((e11_s, e22_s))
        epsilon = epsilon_guess
    else:
        eiis_s, _, k11_s = get_1d_rescaled_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nu=nu)
        epsilon = get_epsilon_using_sij(udata, dx, dy, dz, nu=nu)[t0]
    fig1, ax1 = graph.plot(k11[1:], kolmogorov_53_uni(k11[1:], epsilon, c=0.5),
                           label='$C_1\epsilon^{2/3}\kappa^{-5/3}$', fignum=1, subplot=121, color='k')
    fig1, ax2 = graph.plot_saddoughi(fignum=fignum, subplot=122, color='k', label='Scaled $E_{11}$ (SV 1994)')

    fig1, ax1 = graph.plot(k11[1:], eiis[0, 1:, t0], label='$E_{11}$' + label, fignum=fignum, subplot=121)
    fig1, ax2 = graph.plot(k11_s[1:], eiis_s[0, 1:, t0], label='Scaled $E_{11}$' + label, fignum=fignum, subplot=122)

    if plot_e22:
        fig1, ax1 = graph.plot(k11[1:], eiis[1, 1:, t0], label='$E_{22}$' + label, fignum=fignum, subplot=121)
        fig1, ax2 = graph.plot(k11_s[1:], eiis_s[1, 1:, t0], label='Scaled $E_{22}$' + label, fignum=fignum,
                               subplot=122)
        ax1_ylabel = ax1_ylabel[:-13] + ', $E_{22}$ ($mm^3/s^2$)'
        ax2_ylabel = ax2_ylabel + ', $E_{22} / (\epsilon\\nu^5)^{1/4}$'

    if plot_ek:
        ek, _, kk = get_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
        ek_s, kk_s = scale_energy_spectrum(ek, kk, epsilon=epsilon, nu=nu)
        fig1, ax1 = graph.plot(kk[:, t0], ek[:, t0], label='$E$' + label, fignum=fignum, subplot=121)
        fig1, ax2 = graph.plot(kk_s[:, t0], ek_s[:, t0], label='Scaled $E$' + label, fignum=fignum, subplot=122)

        ax1_ylabel = ax1_ylabel[:-13] + ', $E$ ($mm^3/s^2$)'
        ax2_ylabel = ax2_ylabel + ', $E / (\epsilon\\nu^5)^{1/4}$'

    graph.tologlog(ax1)
    graph.tologlog(ax2)
    if legend:
        ax1.legend(loc=loc)
        ax2.legend(loc=loc)

    graph.labelaxes(ax1, '$\kappa$ ($mm^{-1}$)', ax1_ylabel)
    graph.labelaxes(ax2, '$\kappa \eta $ ', ax2_ylabel)

    # graph.setaxes(ax1, 10 ** -1.5, 10 ** 0.8, 10 ** 0.3, 10 ** 5.3)
    graph.setaxes(ax2, 10 ** -3.8, 2, 10 ** -3.5, 10 ** 6.5)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig1, (ax1, ax2)


def plot_energy_spectra_w_energy_heatmap(udata, dx, dy, dz=None, x0=0, x1=None, y0=0, y1=None, window='flattop',
                                         epsilon_guess=10 ** 5, nu=1.004, label='',
                                         plot_e22=False, plot_ek=False, fignum=1, t0=0, legend=True, loc=3,
                                         crop_edges=5, yoffset_box=20, sb_txtloc=(-0.1, 0.4),
                                         vmax=10 ** 4.8):
    """
    A method to quickly plot the energy spectra (snapshot) and time-averaged energy

    Parameters
    ----------
    udata: nd array, velocity field,
    dx: float, grid spacing in x-direction
    dy: float, grid spacing in y-direction
    dz: float, grid spacing in z-direction
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    window: str, default: 'flattop', window function to be used in the FFT, see scipy.signal.get_window() for more options
    epsilon_guess: float, default: 10 ** 5, estimate of the dissipation rate of the flow
    nu: float, default: 1.004, kinematic viscosity of the fluid
    label: str, default: '', label of the plot (e.g. 'Data on MM/DD/YY')
    plot_e22: bool, default: False, if True, plot the E22 energy spectrum
    plot_ek: bool, default: False, if True, plot the 3D energy spectrum
    fignum: int, default: 1, figure number
    t0: int, default: 0, index of the time to specify the instantaneous v-field
    legend: bool, default: True, if True, plot legend
    loc: int, default: 3, location of the legend
    crop_edges: int, default: 5, crop the edges of the energy spectra to remove the high frequency noise
    yoffset_box: int, default: 20, y-offset of the energy heatmap with respect to the center of the box
    sb_txtloc: tuple, default: (-0.1, 0.4), location of the scale bar text

    Returns
    -------
    fig: matplotlib.figure.Figure, figure object
    axes: tuple of three matplotlib.axes._subplots.AxesSubplot objects
    """
    __fontsize__ = 20
    __figsize__ = (24, 8)
    # See all available arguments in matplotlibrc
    params = {'figure.figsize': __figsize__,
              'font.size': __fontsize__,  # text
              'legend.fontsize': 18,  # legend
              'axes.labelsize': __fontsize__,  # axes
              'axes.titlesize': __fontsize__,
              'xtick.labelsize': __fontsize__,  # tick
              'ytick.labelsize': __fontsize__,
              'lines.linewidth': 5,
              'axes.titlepad': 10}
    graph.update_figure_params(params)

    dim = udata.shape[0]
    n = crop_edges
    if x1 is None:
        x1 = udata.shape[2] - 1
    if y1 is None:
        y1 = udata.shape[1] - 1

    ax2_ylabel = '$E_{11}$ ($mm^3/s^2$)'
    ax3_ylabel = '$E_{11} / (\epsilon\\nu^5)^{1/4}$'

    # Compute energy heatmap and draw rectangle
    energy_avg = np.nanmean(get_energy(udata), axis=dim)
    xx, yy = get_equally_spaced_grid(udata, spacing=dx)
    # Time-averaged Energy
    fig1, ax1, cc1 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], energy_avg[n:-n, n:-n],
                                      label='$\\frac{1}{2} \langle U_i   U_i \\rangle$ ($mm^2/s^2$)',
                                      vmin=0, vmax=vmax, fignum=fignum, subplot=131)
    graph.draw_box(ax1, xx, yy, yoffset=yoffset_box, sb_txtloc=sb_txtloc)
    graph.draw_rectangle(ax1, xx[y0, x0], yy[y0, x0], np.abs(xx[y0, x1] - xx[y0, x0]), np.abs(yy[y1, x0] - yy[y0, x0]),
                         edgecolor='C0', linewidth=5)

    # Coompute energy spectra
    eiis, err, k11 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)

    if epsilon_guess is not None:
        e11_s, k11_s = scale_energy_spectrum(eiis[0, ...], k11, epsilon=epsilon_guess, nu=nu)
        e22_s, k22_s = scale_energy_spectrum(eiis[1, ...], k11, epsilon=epsilon_guess, nu=nu)
        eiis_s = np.stack((e11_s, e22_s))
        epsilon = epsilon_guess
    else:
        eiis_s, _, k11_s = get_1d_rescaled_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nu=nu)
        epsilon = get_epsilon_using_sij(udata, dx, dy, dz, nu=nu)[t0]
    fig1, ax2 = graph.plot(k11[1:], kolmogorov_53_uni(k11[1:], epsilon, c=0.5),
                           label='$C_1\epsilon^{2/3}\kappa^{-5/3}$', fignum=fignum, subplot=132, color='k')
    fig1, ax3 = graph.plot_saddoughi(fignum=fignum, subplot=133, color='k', label='Scaled $E_{11}$ (SV 1994)')
    fig1, ax2 = graph.plot(k11[1:], eiis[0, 1:, t0], label='$E_{11}$' + label, fignum=fignum, subplot=132)
    fig1, ax3 = graph.plot(k11_s[1:], eiis_s[0, 1:, t0], label='Scaled $E_{11}$' + label, fignum=fignum, subplot=133)

    if plot_e22:
        fig1, ax2 = graph.plot(k11[1:], eiis[1, 1:, t0], label='$E_{22}$' + label, fignum=fignum, subplot=132)
        fig1, ax3 = graph.plot(k11_s[1:], eiis_s[1, 1:, t0], label='Scaled $E_{22}$' + label, fignum=fignum,
                               subplot=133)
        ax2_ylabel = ax2_ylabel[:-13] + ', $E_{22}$ ($mm^3/s^2$)'
        ax3_ylabel = ax3_ylabel + ', $E_{22} / (\epsilon\\nu^5)^{1/4}$'

    if plot_ek:
        ek, _, kk = get_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
        ek_s, kk_s = scale_energy_spectrum(ek, kk, epsilon=epsilon, nu=nu)
        fig1, ax2 = graph.plot(kk, ek[:, t0], label='$E$' + label, fignum=fignum, subplot=132)
        fig1, ax3 = graph.plot(kk_s, ek_s[:, t0], label='Scaled $E$' + label, fignum=fignum, subplot=133)

        ax2_ylabel = ax2_ylabel[:-13] + ', $E$ ($mm^3/s^2$)'
        ax3_ylabel = ax3_ylabel + ', $E / (\epsilon\\nu^5)^{1/4}$'

    graph.tologlog(ax2)
    graph.tologlog(ax3)
    if legend:
        ax2.legend(loc=loc)
        ax3.legend(loc=loc)

    graph.labelaxes(ax2, '$\kappa$ ($mm^{-1}$)', ax2_ylabel)
    graph.labelaxes(ax3, '$\kappa \eta $ ', ax3_ylabel)

    # graph.setaxes(ax1, 10 ** -1.5, 10 ** 0.8, 10 ** 0.3, 10 ** 5.3)
    graph.setaxes(ax3, 10 ** -3.8, 2, 10 ** -3.5, 10 ** 6.5)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig1, (ax1, ax2, ax3)


def plot_energy_spectra_avg_w_energy_heatmap(udata, dx, dy, dz=None, x0=0, x1=None, y0=0, y1=None, window='flattop',
                                             epsilon_guess=10 ** 5, nu=1.004, label='',
                                             plot_e11=True, plot_e22=False, plot_ek=False, plot_kol=False, plot_sv=True,
                                             color_ref='k', alpha_ref=0.6,
                                             fignum=1, legend=True, loc=3,
                                             crop_edges=5, yoffset_box=20, sb_txtloc=(-0.1, 0.4), errorfill=True,
                                             vmin=0, vmax=10 ** 5,
                                             return_spectra=False,
                                             figparams=None):
    """
    A method to quickly plot the energy spectra (Time-averaged) and time-averaged energy

    Parameters
    ----------
    udata: nd array, velocity field,
    dx: float, grid spacing in x-direction
    dy: float, grid spacing in y-direction
    dz: float, grid spacing in z-direction
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    window: str, default: 'flattop', window function to be used in the FFT, see scipy.signal.get_window() for more options
    epsilon_guess: float, default: 10 ** 5, estimate of the dissipation rate of the flow
    nu: float, default: 1.004, kinematic viscosity of the fluid
    label: str, default: '', label of the plot (e.g. 'Data on MM/DD/YY')
    plot_e22: bool, default: False, if True, plot the E22 energy spectrum
    plot_ek: bool, default: False, if True, plot the 3D energy spectrum
    fignum: int, default: 1, figure number
    t0: int, default: 0, index of the time to specify the instantaneous v-field
    legend: bool, default: True, if True, plot legend
    loc: int, default: 3, location of the legend
    crop_edges: int, default: 5, crop the edges of the energy spectra to remove the high frequency noise
    yoffset_box: int, default: 20, y-offset of the energy heatmap with respect to the center of the box
    sb_txtloc: tuple, default: (-0.1, 0.4), location of the scale bar text
    return_spectra: bool, default: False, if True, return the values of energy spectra

    Returns
    -------
    fig: matplotlib.figure.Figure, figure object
    axes: tuple of three matplotlib.axes._subplots.AxesSubplot objects
    spectra_dict: dict, optional. Returned if return_spectra is True
    """
    if figparams is None:
        __fontsize__ = 20
        __figsize__ = (24, 8)
        # See all available arguments in matplotlibrc
        params = {'figure.figsize': __figsize__,
                  'font.size': __fontsize__,  # text
                  'legend.fontsize': 18,  # legend
                  'axes.labelsize': __fontsize__,  # axes
                  'axes.titlesize': __fontsize__,
                  'xtick.labelsize': __fontsize__,  # tick
                  'ytick.labelsize': __fontsize__,
                  'lines.linewidth': 5,
                  'axes.titlepad': 10
                  }
    else:
        params = figparams
    graph.update_figure_params(params)

    dim = udata.shape[0]
    n = crop_edges
    if x1 is None: x1 = udata.shape[2] - 1
    if y1 is None: y1 = udata.shape[1] - 1

    ax2_ylabel = '$E_{11}$ ($mm^3/s^2$)'
    ax3_ylabel = '$E_{11} / (\epsilon\\nu^5)^{1/4}$'

    # Compute energy heatmap and draw rectangle
    energy_avg = np.nanmean(get_energy(udata), axis=dim)
    xx, yy = get_equally_spaced_grid(udata, spacing=dx)
    # Time-averaged Energy
    fig1, ax1, cc1 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], energy_avg[n:-n, n:-n],
                                      label='$\\frac{1}{2} \langle U_i   U_i \\rangle$ ($mm^2/s^2$)',
                                      vmin=vmin, vmax=vmax, fignum=fignum, subplot=131)
    graph.draw_box(ax1, xx, yy, yoffset=yoffset_box, sb_txtloc=sb_txtloc)
    graph.draw_rectangle(ax1, xx[y0, x0], yy[y0, x0], np.abs(xx[y0, x1] - xx[y0, x0]), np.abs(yy[y1, x0] - yy[y0, x0]),
                         edgecolor='C0', linewidth=5)

    # Compute energy spectra
    eiis_raw, err, k11 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    eiis = np.nanmean(eiis_raw, axis=2)
    eii_errs = np.nanstd(eiis_raw, axis=2) / 2.
    if epsilon_guess is not None:
        e11_s, k11_s = scale_energy_spectrum(eiis_raw[0, ...], k11, epsilon=epsilon_guess, nu=nu)
        e22_s, k22_s = scale_energy_spectrum(eiis_raw[1, ...], k11, epsilon=epsilon_guess, nu=nu)
        eiis_s = np.stack((np.nanmean(e11_s, axis=1), np.nanmean(e22_s, axis=1)))
        eii_errs_s = np.stack((np.nanstd(e11_s, axis=1), np.nanstd(e22_s, axis=1))) / 2.
        epsilon = epsilon_guess
    else:
        eiis_s_raw, _, k11_s = get_1d_rescaled_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nu=nu)
        eiis_s = np.nanmean(eiis_s_raw, axis=2)
        eii_errs_s = np.nanstd(eiis_s_raw, axis=2)
        k11_s = np.nanmean(k11_s, axis=1)
        epsilon = np.nanmean(get_epsilon_using_sij(udata, dx, dy, dz, nu=nu))
    if plot_kol:
        fig1, ax2 = graph.plot(k11[1:], kolmogorov_53_uni(k11[1:], epsilon, c=0.5),
                               label='$C_1\epsilon^{2/3}\kappa^{-5/3}$',
                               fignum=fignum, subplot=132,
                               color=color_ref, alpha=alpha_ref, lw=params['lines.linewidth'] * 0.6)
    if plot_sv:
        fig1, ax3 = graph.plot_saddoughi(fignum=fignum, subplot=133,
                                         color=color_ref, alpha=alpha_ref, lw=params['lines.linewidth'] * 0.6,
                                         label='Scaled $E_{11}$ (SV 1994)')
    if plot_e11:
        if not errorfill:
            fig1, ax2 = graph.plot(k11[1:], eiis[0, 1:], label='$E_{11}$' + label, fignum=fignum, subplot=132,
                                   linewidth=10)
            fig1, ax3 = graph.plot(k11_s[1:], eiis_s[0, 1:], label='Scaled $E_{11}$' + label, fignum=fignum,
                                   subplot=133, linewidth=10)
        else:
            fig1, ax2, _ = graph.errorfill(k11[1:], eiis[0, 1:], eii_errs[0, 1:], label='$E_{11}$' + label,
                                           fignum=fignum, subplot=132)
            fig1, ax3, _ = graph.errorfill(k11_s[1:], eiis_s[0, 1:], eii_errs_s[0, 1:], label='Scaled $E_{11}$' + label,
                                           fignum=fignum, subplot=133)

    if plot_e22:
        if not errorfill:
            fig1, ax2 = graph.plot(k11[1:], eiis[1, 1:], label='$E_{22}$' + label, fignum=fignum, subplot=132)
            fig1, ax3 = graph.plot(k11_s[1:], eiis_s[1, 1:], label='Scaled $E_{22}$' + label, fignum=fignum,
                                   subplot=133)
        else:
            fig1, ax2, _ = graph.errorfill(k11[1:], eiis[1, 1:], eii_errs[1, 1:], label='$E_{22}$' + label,
                                           fignum=fignum, subplot=132)
            fig1, ax3, _ = graph.errorfill(k11_s[1:], eiis_s[1, 1:], eii_errs_s[1, 1:], label='Scaled $E_{22}$' + label,
                                           fignum=fignum, subplot=133)
        ax2_ylabel = ax2_ylabel[:-13] + ', $E_{22}$ ($mm^3/s^2$)'
        ax3_ylabel = ax3_ylabel + ', $E_{22} / (\epsilon\\nu^5)^{1/4}$'

    if plot_ek:
        ek_raw, _, kk = get_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
        ek_s_raw, kk_s = scale_energy_spectrum(ek_raw, kk, epsilon=epsilon, nu=nu)

        ek, ek_s = np.nanmean(ek_raw, axis=1), np.nanmean(ek_s_raw, axis=1)
        ek_err, ek_s_err = np.nanstd(ek_raw, axis=1), np.nanstd(ek_s_raw, axis=1)
        if not errorfill:
            fig1, ax2 = graph.plot(kk, ek, label='$E$' + label, fignum=fignum, subplot=132, color='b', linestyle='--',
                                   linewidth=10)
            fig1, ax3 = graph.plot(kk_s, ek_s, label='Scaled $E$' + label, fignum=fignum, subplot=133, color='b',
                                   linestyle='--', linewidth=10)
        else:
            fig1, ax2, _ = graph.errorfill(kk, ek, ek_err, label='$E$' + label, fignum=fignum, subplot=132)
            fig1, ax3, _ = graph.errorfill(kk_s, ek_s, ek_s_err, label='Scaled $E$' + label, fignum=fignum, subplot=133)

        ax2_ylabel = ax2_ylabel[:-13] + ', $E$ ($mm^3/s^2$)'
        ax3_ylabel = ax3_ylabel + ', $E / (\epsilon\\nu^5)^{1/4}$'

    graph.tologlog(ax2)
    graph.tologlog(ax3)
    if legend:
        ax2.legend(loc=loc)
        ax3.legend(loc=loc)

    graph.labelaxes(ax2, '$\kappa$ ($mm^{-1}$)', ax2_ylabel)
    graph.labelaxes(ax3, '$\kappa \eta $ ', ax3_ylabel)

    # graph.setaxes(ax1, 10 ** -1.5, 10 ** 0.8, 10 ** 0.3, 10 ** 5.3)
    graph.setaxes(ax3, 10 ** -3.8, 2, 10 ** -3.5, 10 ** 6.5)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    if return_spectra:
        spectra_dict = {'dx': dx, 'dy': dy,
                        'epsilon': epsilon, 'nu': nu,
                        'k11': k11[1:], 'e11': eiis[1, 1:], 'e11_err': eii_errs[1, 1:],
                        'k22': k11[1:], 'e22': eiis[0, 1:], 'e11_err': eii_errs[0, 1:],
                        'k11_s': k11_s[1:], 'e11_s': eiis_s[1, 1:], 'e11_errs_s': eii_errs_s[1, 1:],
                        'k22_s': k11_s[1:], 'e22_s': eiis_s[0, 1:], 'e11_errs_s': eii_errs_s[0, 1:],
                        }
        if plot_ek:
            spectra_dict.update({'kr': kk, 'ek': ek, 'ek_err': ek_err,
                                 'kr_s': kk_s, 'ek_s': ek_s, 'ek_s_err': ek_s_err, })
        return fig1, (ax1, ax2, ax3), spectra_dict
    else:
        return fig1, (ax1, ax2, ax3)


def plot_mean_flow(udata, xx, yy, f_p=5., crop_edges=4, fps=1000., data_spacing=1, umin=-200, umax=200, tau0=0, tau1=10,
                   yoffset_box=20.):
    """
    A method to quickly plot the mean flow (Only applicable to 2D v-field), time-averaged energy and time-averaged  enstrophy
    ... dependencies: tflow.graph

    Parameters
    ----------
    udata: nd array, velcity field
    xx: 2d array, x-coordinates
    yy: 2d array, y-coordinates
    f_p: float, forcing frequency
    crop_edges: int, default: 5, crop the edges of the energy spectra to remove the high frequency noise
    fps: int, frame rate of the udata
    data_spacing: int, default=1,  Forcing period = tau_p = int(1. / f_p * fps / data_spacing)
    umin: float, ignore velocity values below this value
    umax: float, ignore velocity values above this value
    tau0: float, used to plot vertical bands
    tau1: float, used to plot vertical bands
    yoffset_box: int, default: 20, y-offset of the energy heatmap with respect to the center of the box

    Returns
    -------
    fig: matplotlib.figure.Figure, figure object
    axes: tuple of three matplotlib.axes._subplots.AxesSubplot objects
    """
    __figsize__, __fontsize__ = (24, 20), 16
    params = {'figure.figsize': __figsize__,
              'font.size': __fontsize__,  # text
              'legend.fontsize': 16,  # legend
              'axes.labelsize': __fontsize__,  # axes
              'axes.titlesize': __fontsize__,
              'xtick.labelsize': __fontsize__,  # tick
              'ytick.labelsize': __fontsize__,
              'axes.edgecolor': 'black',
              'axes.linewidth': 0.8,
              'lines.linewidth': 5.}
    graph.update_figure_params(params)
    # Forcing Period
    tau_p = int(1. / f_p * fps / data_spacing)  # 1/f *(fps/data_spacing) in frames
    # no. of pixels to ignore at the edge
    n = crop_edges

    # PLOTTING
    gridshape = (7, 6)
    ax_evst = plt.subplot2grid(gridshape, (0, 0), colspan=6)
    ax_enstvst = ax_evst.twinx()

    ax_eavg = plt.subplot2grid(gridshape, (1, 1), colspan=2, rowspan=2)
    ax_e_mf = plt.subplot2grid(gridshape, (1, 3), colspan=2, rowspan=2)

    ax_enstavg = plt.subplot2grid(gridshape, (3, 1), colspan=2, rowspan=2)
    ax_enst_mf = plt.subplot2grid(gridshape, (3, 3), colspan=2, rowspan=2)

    ax_ux_mf = plt.subplot2grid(gridshape, (5, 0), colspan=2, rowspan=2)
    ax_uy_mf = plt.subplot2grid(gridshape, (5, 2), colspan=2, rowspan=2)
    ax_omega_mf = plt.subplot2grid(gridshape, (5, 4), colspan=2, rowspan=2)

    mask = np.empty(udata.shape[-1])
    for i in range(len(mask)):
        if (i - tau0) % tau_p < tau1:
            mask[i] = True
        else:
            mask[i] = False
    mask = mask.astype('bool')

    # Compute quantities
    ## spacing
    dx = np.abs(xx[0, 1] - xx[0, 0])
    dy = np.abs(yy[1, 0] - yy[0, 0])
    ## time average using all data
    udata_m_all, udata_t_all = reynolds_decomposition(udata)
    e_avg_all, _ = get_spatial_avg_energy(udata, x0=n, x1=-n, y0=n, y1=-n)
    e_t_avg_all, _ = get_spatial_avg_energy(udata_t_all, x0=n, x1=-n, y0=n, y1=-n)
    enst_avg_all, _ = get_spatial_avg_enstrophy(udata, x0=n, x1=-n, y0=n, y1=-n, dx=dx, dy=dy)

    ## Reynolds decomposition (For a specified region)
    udata_m, udata_t = reynolds_decomposition(udata[..., mask])
    ## energy / enstrophy
    time = np.arange(udata.shape[-1]) / (fps / data_spacing)
    e_t_avg, _ = get_spatial_avg_energy(udata_t, x0=n, x1=-n, y0=n, y1=-n)

    energy = get_energy(udata[..., mask])
    enstrophy = get_enstrophy(udata[..., mask], dx=dx, dy=dy, xx=None, yy=None)

    e = np.nanmean(energy, axis=2)
    enst = np.nanmean(enstrophy, axis=2)
    energy_m = get_energy(udata_m)
    omega_m = curl(udata_m, dx=dx, dy=dy)
    enst_m = omega_m ** 2

    # PLOT
    # E vs t
    l_e = ax_evst.plot(time, e_avg_all, label='$\langle E \\rangle_{space}$', alpha=0.8)
    l_k = ax_evst.plot(time, e_t_avg_all, label='$\langle k \\rangle_{space}$', alpha=0.8)
    l_enst = ax_enstvst.plot(time, enst_avg_all, label='$\langle \omega_z^2 \\rangle_{space}$', color='C2',
                             alpha=0.8)
    graph.tosemilogy(ax_evst)
    graph.tosemilogy(ax_enstvst)
    ax_evst.set_ylim(10 ** 3.5, 10 ** 5)
    ax_enstvst.set_ylim(10 ** 2.5, 10 ** 4)
    ax_enstvst.set_xlim(time[0], time[0] + (time[-1] - time[0]) * 1.01)
    graph.labelaxes(ax_evst, '$t$ ($s$)', '$E, k$ ($mm^2/s^2$)')
    graph.labelaxes(ax_enstvst, '$t$ ($s$)', '$\langle \omega_z^2 \\rangle$ ($1/s^2$)')
    ## Vertical Bands
    t0 = tau0 * data_spacing / fps
    t1 = t0 + tau1 * data_spacing / fps
    while t1 < time[-1]:
        graph.axvband(ax_enstvst, t0, t1, zorder=0)
        t0 += 1. / f_p
        t1 += 1. / f_p
    ## Label
    lns = l_e + l_k + l_enst
    labs = [l.get_label() for l in lns]
    ax_evst.legend(lns, labs, loc=1, ncol=3, facecolor='white', framealpha=1.0, frameon=False)

    # Time-averaged Energy
    fig, ax_eavg, cc1 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], e[n:-n, n:-n], ax=ax_eavg,
                                         label='$\\frac{1}{2} \langle U_i   U_i \\rangle$ ($mm^2/s^2$)',
                                         vmin=0, vmax=10 ** 4.8)
    # Mean Flow Energy
    fig, ax_e_mf, cc2 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], energy_m[n:-n, n:-n], ax=ax_e_mf,
                                         label='$\\frac{1}{2} \langle U_i  \\rangle  \langle U_i  \\rangle$ ($mm^2/s^2$)',
                                         vmin=0, vmax=10 ** 4.2)

    # Time-averaged Enstrophy
    fig, ax_enstavg, cc3 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], enst[n:-n, n:-n], ax=ax_enstavg,
                                            label='$\langle \omega_z^2 \\rangle$ ($1/s^2$)', vmin=0,
                                            vmax=10 ** 3.7)

    # Mean Flow Enstrophy
    fig, ax_enst_mf, cc4 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], enst_m[n:-n, n:-n, 0],
                                            ax=ax_enst_mf, vmin=0, vmax=250,
                                            label='$\langle \Omega_z  \\rangle  ^2$ ($1/s^2$)')

    # Mean Flow Ux
    fig, ax_ux_mf, cc5 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], udata_m[0, n:-n, n:-n],
                                          ax=ax_ux_mf, cmap='bwr',
                                          label='$\langle U_x \\rangle$ ($mm/s$)', vmin=umin, vmax=umax)

    # Mean Flow Uy
    fig5, ax_uy_mf, cc6 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], udata_m[1, n:-n, n:-n],
                                           ax=ax_uy_mf, cmap='bwr',
                                           label='$\langle U_y \\rangle$ ($mm/s$)', vmin=umin, vmax=umax)

    # Mean Flow Vorticity
    fig, ax_omega_mf, cc = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], omega_m[n:-n, n:-n, 0],
                                            ax=ax_omega_mf, cmap='bwr',
                                            label='$\langle \Omega_z \\rangle$ ($1/s$)', vmin=-20, vmax=20)

    axes_to_add_box = [ax_eavg, ax_e_mf, ax_enstavg, ax_enst_mf, ax_ux_mf, ax_uy_mf, ax_omega_mf]
    for ax in axes_to_add_box:
        if ax in [ax_ux_mf, ax_uy_mf, ax_omega_mf]:
            graph.draw_box(ax, xx, yy, yoffset=yoffset_box, facecolor='white', sb_txtcolor='k', sb_txtloc=(-0.1, 0.4))
        else:
            graph.draw_box(ax, xx, yy, yoffset=yoffset_box, sb_txtloc=(-0.1, 0.4))
    fig.tight_layout()

    axes = [ax_evst, ax_enstvst] + axes_to_add_box
    return fig, axes


def plot_time_avg_energy(udata, xx, yy, x0=0, x1=None, y0=0, y1=None, t0=0, t1=None,
                         label='$\\frac{1}{2} \langle U_i U_i\\rangle~(mm^2/s^2)$',
                         xlabel='$x~(mm)$', ylabel='$y~(mm)$', vmin=None, vmax=None, **kwargs):
    """

    Parameters
    ----------
    udata
    xx
    yy
    x0
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])


    Returns
    -------

    """
    energy = get_energy(udata[:, y0:y1, x0:x1, t0:t1])
    e = np.nanmean(energy, axis=-1)
    fig, ax, cc = graph.color_plot(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], e, label=label, vmin=vmin, vmax=vmax, **kwargs)
    graph.labelaxes(ax, xlabel, ylabel)

    return fig, ax, cc


def plot_time_avg_enstrophy(udata, xx, yy, x0=0, x1=None, y0=0, y1=None, t0=0, t1=None,
                            label='$\\frac{1}{2} \langle \omega_z ^2\\rangle~(1/s^2)$',
                            xlabel='$x~(mm)$', ylabel='$y~(mm)$', vmin=None, vmax=None, **kwargs):
    """
    Plots time-averaged enstrophy

    Parameters
    ----------
    udata: nd array, velocity field,
    dx: float, grid spacing in x-direction
    dy: float, grid spacing in y-direction
    dz: float, grid spacing in z-direction
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1, t0:t1])
    x1: int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1, t0:t1])
    y0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1, t0:t1])
    y1: int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1, t0:t1])
    t0: int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1, t0:t1])
    t1: int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1, t0:t1])
    label: str, default: '$\\frac{1}{2} \langle \omega_z ^2\\rangle~(1/s^2)$'
    xlabel: str, xlabel
    ylabel: str, ylabel
    vmin: float, default: None, color range
    vmax: float, default: None, color range
    kwargs: passed to pcolormesh()

    Returns
    -------
    fig: figure
    ax: axes
    cc: mappable
    """
    enstrophy = get_enstrophy(udata[:, y0:y1, x0:x1, t0:t1], xx=xx, yy=yy)
    en = np.nanmean(enstrophy, axis=-1)
    fig, ax, cc = graph.color_plot(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], en, label=label, vmin=vmin, vmax=vmax, **kwargs)
    graph.labelaxes(ax, xlabel, ylabel)

    return fig, ax, cc


def plot_spatial_avg_energy(udata, time, x0=0, x1=None, y0=0, y1=None, t0=0, t1=None,
                            ylabel='$\\frac{1}{2} \langle U_i U_i\\rangle~(mm^2/s^2)$',
                            xlabel='$t~(s)$', xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
    """
    Plots spatially averaged energy vs time
    ... <0.5 ux^2 + uy^2 + uz^2>_space

    Parameters
    ----------
    udata: nd array- velocity field data
    xx: 2d array- positional grids (x-coordinate)
    yy: 2d array- positional grids (y-coordinate)
    x0: int, xx[y0:y1, x0:x1] will be plotted
    x1: int, xx[y0:y1, x0:x1] will be plotted
    y0: int, xx[y0:y1, x0:x1] will be plotted
    y1: int, xx[y0:y1, x0:x1] will be plotted
    t0: int, This function considers udata[..., t0:t1]
    t1: int, This function considers udata[..., t0:t1]
    xlabel: str, xlabel
    ylabel: str, ylabel
    vmin: float, default: None, color range
    vmax: float, default: None, color range
    kwargs: passed to pcolormesh()

    Returns
    -------
    fig: figure
    ax: axes
    """
    energy = get_energy(udata[:, y0:y1, x0:x1, t0:t1])
    e = np.nanmean(energy, axis=(0, 1))
    fig, ax = graph.plot(time, e)
    graph.labelaxes(ax, xlabel, ylabel)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    return fig, ax


# movie
def make_movie(imgname=None, imgdir=None, movname=None, indexsz='05', framerate=10, crf=12,
               bkgColor=None, bkgWidth=800, bkgHeight=800, rm_images=False,
               save_into_subdir=False, start_number=0, framestep=1, ext='png', option='normal', overwrite=False,
               invert=False, add_commands=[], ffmpeg_path=os.path.join(moddirpath, 'ffmpeg')):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    ... ffmpeg is not smart enough to recognize a pattern like 0, 50, 100, 150... etc.
        It tries up to an interval of 4. So 0, 3, 6, 9 would work, but this hinders practicality.
        Use the glob feature in that case. i.e. option='glob'
    ... As for images with transparent background, ffmpeg outputs a bad quality movie.
        In that case, provide [bkgColor='white', bkgWidth=800, bkgHeight=800]
        ... bkgWidth and bkgHeight must have the same aspect ratio as the images
        ... this uses an option -filter_complex which is not compatible with the other filtering option -vf.
        ... If one would like to use -vf for inverting images for example, one must code it differently

    Parameters
    ----------
    imgname : str
        ... path and filename for the images to turn into a movie
        ... could be a name of directory where images are stored if option is 'glob'
    movname : str
        path and filename for output movie (movie name)
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    crf: int, 0-51
        ... constant rate factor- the lower the value is, the better the quality becomes
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None
    option: str
        If "glob", it globs all images with the extention in the directory.
        Therefore, the images does not have to be numbered.
    add_commands: list
        A list to add extra commands for ffmpeg. The list will be added before output name
        i.e. ffmpeg -i images command add_commands movie_name
        exmaple: add_commands=['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']
    """
    # if movie name is not given, name it as same as the name of the img directory
    if movname is None:
        if os.path.isdir(imgname):
            if imgname[-1] == '/':
                movname = imgname[:-1]
            else:
                movname = imgname
        else:
            pdir, filename = os.path.split(imgname)
            movname = pdir

    if not option == 'glob':
        if bkgColor is None:
            command = [ffmpeg_path,
                       '-framerate', str(int(framerate)),
                       '-start_number', str(start_number),
                       '-i', imgname + '%' + indexsz + 'd.' + ext,
                       '-pix_fmt', 'yuv420p',
                       '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '%d' % crf, '-threads', '0', '-r', '100']
        else:
            command = [ffmpeg_path,
                       '-f', 'lavfi',
                       '-i', 'color=c=%s:s=%dx%d' % (bkgColor, bkgWidth, bkgHeight),  # input0: monotoneous bkg
                       '-framerate', str(int(framerate)),
                       '-start_number', str(start_number),
                       '-i', imgname + '%' + indexsz + 'd.' + ext,
                       '-shortest', '-filter_complex',  #
                       '[0][1]scale2ref[2][imgref];[2][imgref]overlay=shortest=1',
                       # overlay the img[0] on the white bkg[1]
                       '-pix_fmt', 'yuv420p',
                       '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '%d' % crf, '-threads', '0', '-r', '100']
    else:
        # If images are not numbered or not labeled in a sequence, you can use the glob feature.
        # On command line,
        # ffmpeg -r 1
        # -pattern_type glob
        # -i '/Users/stephane/Documents/git/takumi/library/image_processing/images2/*.png'  ## It is CRITICAL to include '' on the command line!!!!!
        # -vcodec libx264 -crf 25  -pix_fmt yuv420p /Users/stephane/Documents/git/takumi/library/image_processing/images2/sample.mp4
        if bkgColor is None:
            command = [ffmpeg_path,
                       '-pattern_type', 'glob',  # Use glob feature
                       '-framerate', str(int(framerate)),  # framerate
                       '-i', imgname + '/*.' + ext,  # images
                       '-vcodec', 'libx264',  # codec
                       '-crf', '%d' % crf,  # quality
                       '-pix_fmt', 'yuv420p']
        else:
            command = [ffmpeg_path,
                       '-f', 'lavfi',
                       '-i', 'color=c=%s:s=%dx%d' % (bkgColor, bkgWidth, bkgHeight),  # input0: monotoneous bkg
                       # '-i', 'color=c=%s:s=800x400' % (bkgColor),  # input0: monotoneous bkg
                       '-pattern_type', 'glob',  # Use glob feature
                       '-framerate', str(int(framerate)),  # framerate
                       '-i', imgname + '/*.' + ext,  # images
                       '-shortest', '-filter_complex',  #
                       '[0][1]scale2ref[2][imgref];[2][imgref]overlay=shortest=1',
                       # overlay the img[0] on the white bkg[1]
                       '-vcodec', 'libx264',  # codec
                       '-crf', '%d' % crf,  # quality
                       '-pix_fmt', 'yuv420p']
    if overwrite:
        command.append('-y')
    if invert:
        if bkgColor is None:
            command.append('-vf')
            command.append('negate')
        else:
            print('WARNING: inversion cannot be completed as -vf is not compatible with -filter_complex.')

    if bkgColor is None:
        command += ['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']

    print(command)
    command += add_commands

    command.append(movname + '.mp4')
    subprocess.call(command)

    # Delete the original images
    if rm_images:
        print('Deleting the original images...')
        if not save_into_subdir and imgdir is None:
            imdir = os.path.split(imgname)
        print('Deleting folder ' + imgdir)
        subprocess.call(['rm', '-r', imgdir])


def make_time_evo_movie_from_udata(qty, xx, yy, time, t=1, inc=100, label='$\\frac{1}{2} U_i U_i$ ($mm^2/s^2$)',
                                   x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                   vmin=0, vmax=None, cmap='magma', option='scientific',
                                   draw_box=True, xlabel='$x$ ($mm$)', ylabel='$y$ ($mm$)',
                                   invert_y=False,
                                   savedir='./', qtyname='qty', framerate=10,
                                   ffmpeg_path=os.path.join(moddirpath, 'ffmpeg'), overwrite=True, redo=False,
                                   notebook=True, verbose=False, box_kwargs={}):
    """
    Make a movie about the running average (number of frames to average is specified by "t")

    Parameters
    ----------
    qty: 3D array (height, width, duration)
        ... quantity to show as a movie (energy, enstrophy, vorticity, etc)
    xx: 2D array, x-coordinates (height, width)
    yy: 2D array, y-coordinates (height, width)
    time: 1D array, time
    t: int, number of frames to average
    inc: int, number of frames to skip between two frames
    label: str, label of the quantity, default is '$\\frac{1}{2} U_i U_i$ ($mm^2/s^2$)'
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    vmin: float, default: 0, color range minimum
    vmax: float, default: None, color range minimum
    draw_box: bool, default: True, draw a box around the heatmap
    xlabel: str, default: '$x$ ($mm$)'
    ylabel: str, default: '$y$ ($mm$)'
    savedir: str, default: './', directory where the movie is saved
    qtyname: str, default: 'qty', name of the quantity
    framerate: int, default: 10, framerate of the movie
    ffmpeg_path: str, path to ffmpeg
    overwrite: bool, default: True, if True, it overwrites the existing movie
    redo: bool, default: False, if True, it generates the images again even if the movie exists
    notebook: bool, default: True, if True, use the notebook to make the movie

    Returns
    -------

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm
    movname = savedir + '/' + qtyname
    if not os.path.exists(movname + '.mp4') or redo:
        if t != 1:
            qty_ravg = get_running_avg_nd(qty, t)
        else:
            qty_ravg = qty
        for t_ind, t in enumerate(tqdm(time[:-t][::inc])):
            fig1, ax1, cc1 = graph.color_plot(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], qty_ravg[y0:y1, x0:x1, t_ind * inc],
                                              fignum=1, vmin=vmin, vmax=vmax, label=label, option=option,
                                              cmap=cmap)
            if invert_y:
                ax1.invert_yaxis()
            if draw_box:
                graph.draw_box(ax1, xx, yy, **box_kwargs)
            graph.labelaxes(ax1, xlabel, ylabel)
            graph.title(ax1, '$t=%02.3f$ s' % t)
            fig1.tight_layout()
            graph.save(savedir + '/' + qtyname + '/img%07d' % t_ind, ext='png', transparent=False, close=True,
                       savedata=False, verbose=verbose)

    if ffmpeg_path is None:
        print('... a path to ffmpeg is missing! Cannot make a movie')
    else:
        movname = savedir + '/' + qtyname
        if not overwrite:
            counter = 0
            while os.path.exists(movname + '.mp4'):
                movname = movname + '%03d' % counter
                counter += 1
        make_movie(imgname=savedir + '/' + qtyname + '/img', movname=movname, indexsz='07', framerate=framerate,
                   ffmpeg_path=ffmpeg_path, overwrite=True)

    if notebook:
        from tqdm import tqdm as tqdm


#
# def make_time_evo_movie_from_udata_1d(qty, x, time, t=1, inc=100, label='$\\frac{1}{2} U_i U_i$ ($mm^2/s^2$)',
#                                    option='loglog', xlim = [None, None], ylim=[None, None],
#                                    xlabel='$x$ ($mm$)', ylabel='$y$ ($mm$)',
#                                    savedir='./', qtyname='qty', framerate=10,
#                                    ffmpeg_path='ffmpeg', overwrite=True, only_movie=False,
#                                    notebook=True, verbose=False):
#     ## Plotting helpers
#     def tosemilogx(ax=None, **kwargs):
#         if ax == None:
#             ax = plt.gca()
#         ax.set_xscale("log", **kwargs)
#
#     def tosemilogy(ax=None, **kwargs):
#         if ax == None:
#             ax = plt.gca()
#         ax.set_yscale("log", **kwargs)
#
#     def tologlog(ax=None, **kwargs):
#         if ax == None:
#             ax = plt.gca()
#         ax.set_xscale("log", **kwargs)
#         ax.set_yscale("log", **kwargs)
#
#
#     if notebook:
#         from tqdm import tqdm_notebook as tqdm
#         print('Using tqdm_notebook. If this is a mistake, set notebook=False')
#     else:
#         from tqdm import tqdm
#
#     if not only_movie:
#         if t != 1:
#             qty_ravg = get_running_avg_nd(qty, t)
#         else:
#             qty_ravg = qty
#         for t_ind, t in enumerate(tqdm(time[:-t][::inc])):
#             fig1, ax1 = graph.plot(x[:, t_ind*inc], qty[:, t_ind*inc], label=label)
#             graph.labelaxes(ax1, xlabel, ylabel)
#             graph.title(ax1, '$t=%02.3f$ s' % t)
#             if option == 'loglog':
#                 tologlog(ax1)
#             elif option == 'semilogx':
#                 tosemilogx(ax1)
#             elif option == 'semilogy':
#                 tosemilogy(ax1)
#             ax1.set_xlim(xlim)
#             ax1.set_ylim(ylim)
#             fig1.tight_layout()
#             graph.save(savedir + '/' +  qtyname + '/img%07d' % t_ind, ext='png',
#                        transparent=False, close=True,
#                        savedata=False, verbose=verbose)
#
#
#     if ffmpeg_path is None:
#         print('... a path to ffmpeg is not given! cannot make a movie')
#     else:
#         movname = savedir + '/' + qtyname
#         if not overwrite:
#             counter = 0
#             while os.path.exists(movname + '.mov'):
#                 movname = movname[:-4] + '%03d' % counter
#                 counter += 1
#         make_movie(imgname=savedir + '/' +  qtyname + '/img', movname=movname, indexsz='07', framerate=framerate,
#                    ffmpeg_path=ffmpeg_path, overwrite=True)
#
#     if notebook:
#         from tqdm import tqdm as tqdm


# convenient tools
def get_binned_stats(arg, var, n_bins=100, mode='linear',
                     statistic='mean',
                     bin_center=True, return_std=False):
    """
    Make a histogram out of a pair of 1d arrays.
    ... Returns arg_bins, var_mean, var_err
    ... The given arrays could contain nans and infs. They will be ignored.

    Parameters
    ----------
    arg: 1d array, controlling variable
    var: 1d array, data array to be binned
    n_bins: int, default: 100
    mode: str, deafult: 'linear'
        If 'linear', var will be sorted to equally spaced bins. i.e. bin centers increase linearly.
        If 'log', the bins will be not equally spaced. Instead, they will be equally spaced in log.
        ... bin centers will be like... 10**0, 10**0.5, 10**1.0, 10**1.5, ..., 10**9
    return_std: bool
        If True, it returns the STD of the statistics instead of the error = STD / np.sqrt(N)
    Returns
    -------
    arg_bins: 1d array, bin centers
    var_mean: 1d array, mean values of data in each bin
    var_err: 1d array, error of data in each bin
        ... If return_std=True, it returns the STD of the statistics instead of the error = STD / np.sqrt(N)

    """

    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return np.asarray(arr1), np.asarray(arr2)

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    arg, var = np.asarray(arg), np.asarray(var)

    # make sure rr and corr do not contain nans
    mask1 = get_mask_for_nan_and_inf(arg)
    mask1 = ~mask1
    mask2 = get_mask_for_nan_and_inf(var)
    mask2 = ~mask2
    mask = mask1 * mask2

    if mode == 'log':
        argmin, argmax = np.nanmin(arg), np.nanmax(arg)
        mask_for_log10arg = get_mask_for_nan_and_inf(np.log10(arg))
        exp_min, exp_max = np.nanmin(np.log10(arg)[~mask_for_log10arg]), np.nanmax(np.log10(arg)[~mask_for_log10arg])
        exp_interval = (exp_max - exp_min) / n_bins
        exp_bin_centers = np.linspace(exp_min, exp_max, n_bins)
        exp_bin_edges = np.append(exp_bin_centers, exp_max + exp_interval) - exp_interval / 2.
        bin_edges = 10 ** (exp_bin_edges)
        bins = bin_edges
        mask_for_arg = get_mask_for_nan_and_inf(bins)
        bins = bins[~mask_for_arg]
    else:
        bins = n_bins

    # get a histogram
    if not bin_center:
        arg_means, arg_edges, binnumber = binned_statistic(arg[mask], arg[mask], statistic=statistic, bins=bins)
    var_mean, bin_edges, binnumber = binned_statistic(arg[mask], var[mask], statistic=statistic, bins=bins)
    var_std, _, _ = binned_statistic(arg[mask], var[mask], statistic='std', bins=bins)
    counts, _, _ = binned_statistic(arg[mask], var[mask], statistic='count', bins=bins)

    # bin centers
    if mode == 'log':
        bin_centers = 10 ** ((exp_bin_edges[:-1] + exp_bin_edges[1:]) / 2.)
    else:
        binwidth = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - binwidth / 2

    # Sort arrays
    if bin_center:
        arg_bins, var_mean = sort2arr(bin_centers, var_mean)
        arg_bins, var_std = sort2arr(bin_centers, var_std)
    else:
        arg_bins, var_mean = sort2arr(arg_means, var_mean)
        arg_bins, var_std = sort2arr(arg_means, var_std)
    if return_std:
        return arg_bins, var_mean, var_std
    else:
        return arg_bins, var_mean, var_std / np.sqrt(counts)


def get_binned_stats2d(x, y, var, n_bins=100, nx_bins=None, ny_bins=None, bin_center=True,
                       xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Make a histogram out of a pair of 1d arrays.
    ... Returns arg_bins, var_mean, var_err
    ... The given arrays could contain nans and infs. They will be ignored.

    Parameters
    ----------
    x: 2d array, control variable
    y: 2d array, control variable
    var: 2d array, data array to be binned
    n_bins: int, default: 100
    mode: str, deafult: 'linear'
        If 'linear', var will be sorted to equally spaced bins. i.e. bin centers increase linearly.
        If 'log', the bins will be not equally spaced. Instead, they will be equally spaced in log.
        ... bin centers will be like... 10**0, 10**0.5, 10**1.0, 10**1.5, ..., 10**9

    Returns
    -------
    xx_binned: 2d array, bin centers about x
    yy_binned: 2d array, bin centers about y
    var_mean: 2d array,  mean values of data in each bin
    var_err: 2d array, standard error of data in each bin

    """

    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return np.asarray(arr1), np.asarray(arr2)

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    x, y, var = np.asarray(x), np.asarray(y), np.asarray(var)

    # make sure rr and corr do not contain nans
    mask_x = get_mask_for_nan_and_inf(x)
    mask_x = ~mask_x
    mask_y = get_mask_for_nan_and_inf(y)
    mask_y = ~mask_y
    mask_var = get_mask_for_nan_and_inf(var)
    mask_var = ~mask_var
    mask = mask_x * mask_y * mask_var

    if xmin is not None:
        mask_x_less = x > xmin
        mask *= mask_x_less
    if xmax is not None:
        mask_x_greater = x < xmax
        mask *= mask_x_greater
    if ymin is not None:
        mask_y_less = y > ymin
        mask *= mask_y_less
    if ymax is not None:
        mask_y_greater = y < ymax
        mask *= mask_y_greater
    # if mode == 'log':
    #     argmin, argmax = np.nanmin(arg), np.nanmax(arg)
    #     argmin, argmax = np.nanmin(arg), np.nanmax(arg)
    #     mask_for_log10arg = get_mask_for_nan_and_inf(np.log10(arg))
    #     exp_min, exp_max = np.nanmin(np.log10(arg)[~mask_for_log10arg]), np.nanmax(np.log10(arg)[~mask_for_log10arg])
    #     exp_interval = (exp_max - exp_min) / n_bins
    #     exp_bin_centers = np.linspace(exp_min, exp_max, n_bins)
    #     exp_bin_edges = np.append(exp_bin_centers, exp_max + exp_interval) - exp_interval / 2.
    #     bin_edges = 10 ** (exp_bin_edges)
    #     bins = bin_edges
    #     mask_for_arg = get_mask_for_nan_and_inf(bins)
    #     bins = bins[~mask_for_arg]
    # else:
    #     bins = n_bins
    if nx_bins is None and ny_bins is None:
        bins = [n_bins, n_bins]
    else:
        bins = [ny_bins, nx_bins]

    # get a histogram
    var_mean, y_edge, x_edge, binnumber = binned_statistic_2d(y[mask], x[mask], var[mask], statistic='mean', bins=bins)
    var_std, _, _, _ = binned_statistic_2d(y[mask], x[mask], var[mask], statistic='std', bins=bins)
    counts, _, _, _ = binned_statistic_2d(y[mask], x[mask], var[mask], statistic='count', bins=bins)
    var_err = var_std / np.sqrt(counts)

    # bin centers
    # if mode == 'log':
    #     bin_centers = 10 ** ((exp_bin_edges[:-1] + exp_bin_edges[1:]) / 2.)
    # else:
    #     binwidth = (bin_edges[1] - bin_edges[0])
    #     bin_centers = bin_edges[1:] - binwidth / 2

    if bin_center:
        binwidth_x = (x_edge[1] - x_edge[0])
        binwidth_y = (y_edge[1] - y_edge[0])
        bin_centers_x = x_edge[1:] - binwidth_x / 2
        bin_centers_y = y_edge[1:] - binwidth_y / 2
        arg_bins_x = bin_centers_x
        arg_bins_y = bin_centers_y
    else:
        arg_bins_x = x_edge[:-1]
        arg_bins_y = y_edge[:-1]
    #
    # # Sort arrays
    # if bin_center:
    #     arg_bins, var_mean = sort2arr(bin_centers, var_mean)
    #     arg_bins, var_err = sort2arr(bin_centers, var_err)
    # else:
    #     arg_bins, var_mean = sort2arr(arg_means, var_mean)
    #     arg_bins, var_err = sort2arr(arg_means, var_err)

    yy_binned, xx_binned = np.meshgrid(arg_bins_y, arg_bins_x)
    xx_binned, yy_binned = np.meshgrid(arg_bins_x, arg_bins_y)

    return xx_binned, yy_binned, var_mean, var_err


def get_binned_stats3d(x, y, z, var, n_bins=100, nx_bins=None, ny_bins=None, nz_bins=None, bin_center=True,
                       xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None, return_count=False):
    """
    Make a histogram out of a trio of 1d arrays.
    ... Returns arg_bins, var_mean, var_err
    ... The given arrays could contain nans and infs. They will be ignored.

    Parameters
    ----------
    arg: 1d array, controlling variable
    var: 1d array, data array to be binned
    n_bins: int, default: 100
    mode: str, deafult: 'linear'
        If 'linear', var will be sorted to equally spaced bins. i.e. bin centers increase linearly.
        If 'log', the bins will be not equally spaced. Instead, they will be equally spaced in log.
        ... bin centers will be like... 10**0, 10**0.5, 10**1.0, 10**1.5, ..., 10**9

    Returns
    -------
    xx_binned: 3d array, bin centers about x
    yy_binned: 3d array, bin centers about y
    var_mean: 3d array,  mean values of data in each bin
    var_err: 3d array, standard error of data in each bin

    """

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    x, y, z, var = np.asarray(x), np.asarray(y), np.asarray(z), np.asarray(var)

    # make sure rr and corr do not contain nans
    mask_x = get_mask_for_nan_and_inf(x)
    mask_x = ~mask_x
    mask_y = get_mask_for_nan_and_inf(y)
    mask_y = ~mask_y
    mask_z = get_mask_for_nan_and_inf(z)
    mask_z = ~mask_z
    mask_var = get_mask_for_nan_and_inf(var)
    mask_var = ~mask_var
    mask = mask_x * mask_y * mask_z * mask_var

    if xmin is not None:
        mask_x_less = x > xmin
        mask *= mask_x_less
    if xmax is not None:
        mask_x_greater = x < xmax
        mask *= mask_x_greater
    if ymin is not None:
        mask_y_less = y > ymin
        mask *= mask_y_less
    if ymax is not None:
        mask_y_greater = y < ymax
        mask *= mask_y_greater
    if zmin is not None:
        mask_z_less = z > zmin
        mask *= mask_z_less
    if zmax is not None:
        mask_z_greater = z < ymax
        mask *= mask_z_greater
    # if mode == 'log':
    #     argmin, argmax = np.nanmin(arg), np.nanmax(arg)
    #     argmin, argmax = np.nanmin(arg), np.nanmax(arg)
    #     mask_for_log10arg = get_mask_for_nan_and_inf(np.log10(arg))
    #     exp_min, exp_max = np.nanmin(np.log10(arg)[~mask_for_log10arg]), np.nanmax(np.log10(arg)[~mask_for_log10arg])
    #     exp_interval = (exp_max - exp_min) / n_bins
    #     exp_bin_centers = np.linspace(exp_min, exp_max, n_bins)
    #     exp_bin_edges = np.append(exp_bin_centers, exp_max + exp_interval) - exp_interval / 2.
    #     bin_edges = 10 ** (exp_bin_edges)
    #     bins = bin_edges
    #     mask_for_arg = get_mask_for_nan_and_inf(bins)
    #     bins = bins[~mask_for_arg]
    # else:
    #     bins = n_bins
    if nx_bins is None and ny_bins is None:
        bins = [n_bins, n_bins, n_bins]
    else:
        bins = [ny_bins, nx_bins, nz_bins]

    # get a histogram
    var_mean, (y_edge, x_edge, z_edge), binnumber = binned_statistic_dd([y[mask], x[mask], z[mask]], var[mask],
                                                                        statistic='mean', bins=bins)
    var_std, _, _ = binned_statistic_dd([y[mask], x[mask], z[mask]], var[mask], statistic='std', bins=bins)
    counts, _, _ = binned_statistic_dd([y[mask], x[mask], z[mask]], var[mask], statistic='count', bins=bins)
    var_err = var_std / np.sqrt(counts)

    # bin centers
    # if mode == 'log':
    #     bin_centers = 10 ** ((exp_bin_edges[:-1] + exp_bin_edges[1:]) / 2.)
    # else:
    #     binwidth = (bin_edges[1] - bin_edges[0])
    #     bin_centers = bin_edges[1:] - binwidth / 2

    if bin_center:
        binwidth_x = (x_edge[1] - x_edge[0])
        binwidth_y = (y_edge[1] - y_edge[0])
        binwidth_z = (z_edge[1] - z_edge[0])
        bin_centers_x = x_edge[1:] - binwidth_x / 2
        bin_centers_y = y_edge[1:] - binwidth_y / 2
        bin_centers_z = z_edge[1:] - binwidth_z / 2
        arg_bins_x = bin_centers_x
        arg_bins_y = bin_centers_y
        arg_bins_z = bin_centers_z
    else:
        arg_bins_x = x_edge[:-1]
        arg_bins_y = y_edge[:-1]
        arg_bins_z = z_edge[:-1]
    #
    # # Sort arrays
    # if bin_center:
    #     arg_bins, var_mean = sort2arr(bin_centers, var_mean)
    #     arg_bins, var_err = sort2arr(bin_centers, var_err)
    # else:
    #     arg_bins, var_mean = sort2arr(arg_means, var_mean)
    #     arg_bins, var_err = sort2arr(arg_means, var_err)

    xx_binned, yy_binned, zz_binned = np.meshgrid(arg_bins_x, arg_bins_y, arg_bins_z)
    if return_count:
        return xx_binned, yy_binned, zz_binned, var_mean, var_err, counts
    else:
        return xx_binned, yy_binned, zz_binned, var_mean, var_err

# LOADING UDATA
def get_udata(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                        t0=0, t1=None, inc=1, frame=None, return_xy=False, verbose=True,
                        slicez=None, crop=None, mode='r',
                        reverse_x=False, reverse_y=False, reverse_z=False, ind=0):
    """
    Returns udata from a path to udata
    If return_xy is True, it returns udata, xx(2d grid), yy(2d grid)

    Parameters
    ----------
    udatapath
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])

    inc: int
        time increment of data to load from udatapath, default: 1
    frame: array-like or int, default: None
        If an integer is given, it returns a velocity field at that instant of time
        If an array or a list is given, it returns a velocity field at the given time specified by the array/list.

        By default, it loads data by a specified increment "inc".
        If "frame" is given, it is prioritized over the incremental loading.
    return_xy: bool, defualt: False
    verbose: bool
        If True, return the time it took to load udata to memory
    ind: int, id for a file with multiple piv data (udata)
        ... A file may include ux, uy, x, y under /piv/piv000/, /piv/piv001, /piv/piv001, ... when piv is conducted on the same footage.

    Returns
    -------
    udata: nd array with shape (dim, height, width, (depth), duration)
        ... udata[0, ...]: ux, udata[1, ...]: uy, udata[2, ...]: uz
        ... udata[0, ..., t] is a 2D/3D array which stores the x-component of the velocity field.
        ... Intuitively speaking, udata is organized like udata[dim, y, x, (z), time]

    (optional)
    xx: 2d/3d array of a positional grid (x) stored in the given file
    yy: 2d/3d array of a positional grid (y) stored in the given file
    zz: 2d/3d array of a positional grid (z) stored in the given file
    """
    return get_udata_from_path(*args, **kwargs)


def get_udata_from_path(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                        t0=0, t1=None, inc=1, frame=None, return_xy=False, verbose=True,
                        slicez=None, crop=None, mode='r',
                        reverse_x=False, reverse_y=False, reverse_z=False, ind=0):
    """
    Returns udata from a path to udata
    If return_xy is True, it returns udata, xx(2d grid), yy(2d grid)

    Parameters
    ----------
    udatapath
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])

    inc: int
        time increment of data to load from udatapath, default: 1
    frame: array-like or int, default: None
        If an integer is given, it returns a velocity field at that instant of time
        If an array or a list is given, it returns a velocity field at the given time specified by the array/list.

        By default, it loads data by a specified increment "inc".
        If "frame" is given, it is prioritized over the incremental loading.
    return_xy: bool, defualt: False
    verbose: bool
        If True, return the time it took to load udata to memory
    ind: int, id for a file with multiple piv data (udata)
        ... A file may include ux, uy, x, y under /piv/piv000/, /piv/piv001, /piv/piv001, ... when piv is conducted on the same footage.

    Returns
    -------
    udata: nd array with shape (dim, height, width, (depth), duration)
        ... udata[0, ...]: ux, udata[1, ...]: uy, udata[2, ...]: uz
        ... udata[0, ..., t] is a 2D/3D array which stores the x-component of the velocity field.
        ... Intuitively speaking, udata is organized like udata[dim, y, x, (z), time]

    (optional)
    xx: 2d/3d array of a positional grid (x) stored in the given file
    yy: 2d/3d array of a positional grid (y) stored in the given file
    zz: 2d/3d array of a positional grid (z) stored in the given file
    """
    ### Determine wheteher this file is a nested udata
    f = h5py.File(udatapath, 'r')
    keys = list(f.keys())
    f.close()
    ###
    if not 'ux' in keys:
        print(
            '... get_udata_from_path() is not compatible with a data structure in this h5.  \n Try get_udata_from_path_nested() instead.')
        results = get_udata_from_path_nested(udatapath, ind=ind,
                                             x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                             t0=t0, t1=t1, inc=inc, frame=frame, return_xy=return_xy, verbose=verbose,
                                             slicez=slicez, crop=crop, mode=mode,
                                             reverse_x=reverse_x, reverse_y=reverse_y, reverse_z=reverse_z)
        return results
    else:
        if verbose:
            tau0 = time_mod.time()
            print('... reading udata from a given path')
        if crop is not None and [x0, x1, y0, y1, z0, z1] == [0, None, 0, None, 0, None]:
            x0, x1, y0, y1, z0, z1 = crop, -crop, crop, -crop, crop, -crop

        if mode == 'w' or mode == 'wb':
            raise ValueError('... w was passed to h5Py.File(...) which would truncate the file if it exists. \n'
                             'Probably, this is not what you want. Pass r for read-only')

        with h5py.File(udatapath, 'r') as f:
            if 'z' in f.keys():
                dim = 3
            else:
                dim = 2

            if dim == 2:
                if frame is None:
                    ux = f['ux'][y0:y1, x0:x1, t0:t1:inc]
                    uy = f['uy'][y0:y1, x0:x1, t0:t1:inc]
                else:
                    frame = np.asarray(frame)
                    ux = f['ux'][y0:y1, x0:x1, frame]
                    uy = f['uy'][y0:y1, x0:x1, frame]

                udata = np.stack((ux, uy))

                if return_xy:
                    xx, yy = f['x'][y0:y1, x0:x1], f['y'][y0:y1, x0:x1]
            elif dim == 3:
                if frame is None and slicez is None:
                    ux = f['ux'][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                    uy = f['uy'][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                    uz = f['uz'][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                elif frame is None and slicez is not None:
                    ux = f['ux'][y0:y1, x0:x1, slicez, t0:t1:inc]
                    uy = f['uy'][y0:y1, x0:x1, slicez, t0:t1:inc]
                    uz = f['uz'][y0:y1, x0:x1, slicez, t0:t1:inc]
                elif frame is not None and slicez is not None:
                    frame = np.asarray(frame)
                    ux = f['ux'][y0:y1, x0:x1, slicez, frame]
                    uy = f['uy'][y0:y1, x0:x1, slicez, frame]
                    uz = f['uz'][y0:y1, x0:x1, slicez, frame]
                else:
                    frame = np.asarray(frame)
                    ux = f['ux'][y0:y1, x0:x1, z0:z1, frame]
                    uy = f['uy'][y0:y1, x0:x1, z0:z1, frame]
                    uz = f['uz'][y0:y1, x0:x1, z0:z1, frame]
                udata = np.stack((ux, uy, uz))
                if return_xy:
                    if slicez is None:
                        xx, yy, zz = f['x'][y0:y1, x0:x1, z0:z1], f['y'][y0:y1, x0:x1, z0:z1], f['z'][y0:y1, x0:x1,
                                                                                               z0:z1]
                    else:
                        xx, yy, zz = f['x'][y0:y1, x0:x1, slicez], f['y'][y0:y1, x0:x1, slicez], f['z'][0, 0, slicez]
        tau1 = time_mod.time()
        if verbose:
            print(f'... time took to load udata: {tau1 - tau0:.2f} s')

        if return_xy:
            if dim == 2:
                if reverse_x:
                    udata[0, ...] = -udata[0, ...]
                    xx[...] = -xx[:, :]

                if reverse_y:
                    udata[1, ...] = -udata[1, ...]
                    yy[...] = -yy[:, :]
                return udata, xx, yy
            elif dim == 3:
                if reverse_x:
                    udata[...] = -udata[:, :, ::-1, :, :]
                    xx[...] = -xx[:, :, :]

                if reverse_y:
                    udata[1, ...] = -udata[1, ...]
                    yy[...] = -yy[:, :, :]

                if reverse_z:
                    udata[2, ...] = -udata[2, :, :, :, :]
                    zz[...] = -zz[:, :, :]

                return udata, xx, yy, zz
        else:
            return udata


def get_scalar_data_from_path(udatapath, name='pressure', x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                              t0=0, t1=None, inc=1, frame=None, return_xy=False, verbose=True,
                              slicez=None, crop=None, mode='r',
                              reverse_x=False, reverse_y=False, reverse_z=False):
    """
    Returns a scalar data from a path of udata
    ... There could be a case that a scalar data such as temperature and pressure is also stored in udata.h5
    ... This function serves as a reader of such a quantity
    If return_xy is True, it returns udata, xx(2d grid), yy(2d grid)

    Parameters
    ----------
    udatapath: str, a path to udata
    name: str, name of the dataset in the udata h5
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])

    inc: int
        time increment of data to load from udatapath, default: 1
    frame: array-like or int, default: None
        If an integer is given, it returns a velocity field at that instant of time
        If an array or a list is given, it returns a velocity field at the given time specified by the array/list.

        By default, it loads data by a specified increment "inc".
        If "frame" is given, it is prioritized over the incremental loading.
    return_xy: bool, defualt: False
    verbose: bool
        If True, return the time it took to load udata to memory

    Returns
    -------
    pdata: ndarray, scalar data with shape (y, x, (z), t)
    xx, yy, zz(if 3D): ndarray, 2/3d positional grid with shape (y, x, (z))

    """
    f = h5py.File(udatapath, 'r')
    keys = list(f.keys())
    f.close()
    ###
    if not name in keys:
        raise ValueError('%s does not exist in the given path' % name)
    else:
        if verbose:
            tau0 = time_mod.time()
            print('... reading %s from the path' % name)
        if crop is not None and [x0, x1, y0, y1, z0, z1] == [0, None, 0, None, 0, None]:
            x0, x1, y0, y1, z0, z1 = crop, -crop, crop, -crop, crop, -crop

        if mode == 'w' or mode == 'wb':
            raise ValueError('... w was passed to h5Py.File(...) which would delete the file if it exists. \n'
                             'Probably, this is not what you want. Pass r for read-only')

        with h5py.File(udatapath, 'r') as f:
            if 'z' in f.keys():
                dim = 3
            else:
                dim = 2

            if dim == 2:
                if frame is None:
                    pdata = f[name][y0:y1, x0:x1, t0:t1:inc]
                else:
                    frame = np.asarray(frame)
                    pdata = f[name][y0:y1, x0:x1, frame]

                if return_xy:
                    xx, yy = f['x'][y0:y1, x0:x1], f['y'][y0:y1, x0:x1]
            elif dim == 3:
                if frame is None and slicez is None:
                    pdata = f[name][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                elif frame is None and slicez is not None:
                    pdata = f[name][y0:y1, x0:x1, slicez, t0:t1:inc]
                elif frame is not None and slicez is not None:
                    frame = np.asarray(frame)
                    pdata = f[name][y0:y1, x0:x1, slicez, frame]
                else:
                    frame = np.asarray(frame)
                    pdata = f[name][y0:y1, x0:x1, z0:z1, frame]
                if return_xy:
                    if slicez is None:
                        xx, yy, zz = f['x'][y0:y1, x0:x1, z0:z1], f['y'][y0:y1, x0:x1, z0:z1], f['z'][y0:y1, x0:x1,
                                                                                               z0:z1]
                    else:
                        xx, yy, zz = f['x'][y0:y1, x0:x1, slicez], f['y'][y0:y1, x0:x1, slicez], f['z'][0, 0, slicez]
        tau1 = time_mod.time()
        if verbose:
            print(f'... time took to load udata: {tau1 - tau0:.2f} s')

        if return_xy:
            if dim == 2:
                if reverse_x:
                    pdata[...] = pdata[:, ::-1, :]
                    xx[...] = xx[:, ::-1]
                    yy[...] = yy[:, ::-1]

                if reverse_y:
                    pdata[...] = pdata[:, ::-1, :, :]
                    xx[...] = xx[::-1, :]
                    yy[...] = yy[::-1, :]
                return pdata, xx, yy
            elif dim == 3:
                if reverse_x:
                    pdata[...] = pdata[:, ::-1, :, :]
                    xx[...] = xx[:, ::-1, :]
                    yy[...] = yy[:, ::-1, :]
                    zz[...] = zz[:, ::-1, :]

                if reverse_y:
                    pdata[...] = pdata[::-1, :, :, :]
                    xx[...] = xx[::-1, :, :]
                    yy[...] = yy[::-1, :, :]
                    zz[...] = zz[::-1, :, :]

                if reverse_z:
                    pdata[...] = pdata[:, :, ::-1, :]
                    xx[...] = xx[:, :, ::-1]
                    yy[...] = yy[:, :, ::-1]
                    zz[...] = zz[:, :, ::-1]

                return pdata, xx, yy, zz
        else:
            return pdata


def get_udata_from_path_nested(udatapath, ind=0,
                               x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                               t0=0, t1=None, inc=1, frame=None, return_xy=False, verbose=True,
                               slicez=None, crop=None, mode='r',
                               reverse_x=False, reverse_y=False, reverse_z=False):
    """
    A function to read udata from a hdf5 file
    ... This function is suited for a nested structure Takumi used temporarily  between 2017-18.
        This format stores udata generated by multiple PIV settings
        (different Dt, interrogation window size, etc) on the same movie

    ... The hdf5 should have a following structure
        /piv/exp/__quantity_like_piston_velocity_piston_position_etc
        /piv/piv000/__quantity_like_ux_uy_x_y_lambda_etc
        /piv/piv001/__quantity_like_ux_uy_x_y_lambda_etc
        ...
        /piv/piv010/__quantity_like_ux_uy_x_y_lambda_etc

    ... Specify which udata to load via "ind"


    Parameters
    ----------
    udatapath, str
    ind: int, index to specify which piv data to load in a nested h5
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    inc: int
    frame: int
    return_xy: bool, If True, it retursn a 2D/3D grid which can be used for pcolormesh etc.
    verbose: bool, If True, it reports progress
    slicez: int, default: None
        ... Feature for loading a planar udata from a volumetric udata
            If given, it returns a (ux, uy, uz) on a plane z=slizez (in index space)
    crop: int, default: None
        If given, it will make x0=y0=z0=crop, x1=y1=z1=-crop
        ... This literally "crops" a velocity field in a square/cubic mannar
    reverse_x: bool, If True, it reverses the order of x
        ... This is sometimes useful if external function requires a monotonically increasing array.
            e.g. interpolation in scipy/numpy
    reverse_y: bool, If True, it reverses the order of y
    reverse_z: bool, If True, it reverses the order of z


    Returns
    -------
    udata: nd array, velocity field
    xx, yy, zz: 2D/3D array, position grid, (if return_xy is True)

    """

    if verbose:
        tau0 = time_mod.time()
        print('... reading udata from path')
    if crop is not None and [x0, x1, y0, y1, z0, z1] == [0, None, 0, None, 0, None]:
        x0, x1, y0, y1, z0, z1 = crop, -crop, crop, -crop, crop, -crop

    if mode == 'w' or mode == 'wb':
        print('... w was passed to h5Py.File(...) which would truncate the file if it exists. \n'
              'Probably, this is not what you want. Pass r for read-only')
        raise ValueError

    with h5py.File(udatapath, mode='r') as fyle:
        if verbose:
            print('... Top keys are...', fyle.keys())
        if 'piv' in fyle.keys():
            if verbose:
                print('... This must be a set of udata which might be accompanied with its derivatives')
            pivnames = sorted(fyle['piv'].keys())
            n_pivdata = len(pivnames)

            if verbose:
                print('... %d udata are found.' % n_pivdata)
                print('... Loading the %d-th udata (key: %s)' % (ind, pivnames[ind]))
                print('... the loading data has these %d derivatives (make sure (ux, uy, x, y) exist)'
                      % len(fyle['piv'][pivnames[ind]].keys()),
                      fyle['piv'][pivnames[ind]].keys())

            f = fyle['piv'][pivnames[ind]]

            if 'z' in f.keys():
                dim = 3
            else:
                dim = 2

            if dim == 2:
                if frame is None:
                    ux = f['ux'][y0:y1, x0:x1, t0:t1:inc]
                    uy = f['uy'][y0:y1, x0:x1, t0:t1:inc]
                else:
                    frame = np.asarray(frame)
                    ux = f['ux'][y0:y1, x0:x1, frame]
                    uy = f['uy'][y0:y1, x0:x1, frame]

                udata = np.stack((ux, uy))

                if return_xy:
                    xx, yy = f['x'][y0:y1, x0:x1], f['y'][y0:y1, x0:x1]
            elif dim == 3:
                if frame is None and slicez is None:
                    ux = f['ux'][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                    uy = f['uy'][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                    uz = f['uz'][y0:y1, x0:x1, z0:z1, t0:t1:inc]
                elif frame is None and slicez is not None:
                    ux = f['ux'][y0:y1, x0:x1, slicez, t0:t1:inc]
                    uy = f['uy'][y0:y1, x0:x1, slicez, t0:t1:inc]
                    uz = f['uz'][y0:y1, x0:x1, slicez, t0:t1:inc]
                elif frame is not None and slicez is not None:
                    frame = np.asarray(frame)
                    ux = f['ux'][y0:y1, x0:x1, slicez, frame]
                    uy = f['uy'][y0:y1, x0:x1, slicez, frame]
                    uz = f['uz'][y0:y1, x0:x1, slicez, frame]
                else:
                    frame = np.asarray(frame)
                    ux = f['ux'][y0:y1, x0:x1, z0:z1, frame]
                    uy = f['uy'][y0:y1, x0:x1, z0:z1, frame]
                    uz = f['uz'][y0:y1, x0:x1, z0:z1, frame]
                udata = np.stack((ux, uy, uz))
                if return_xy:
                    if slicez is None:
                        xx, yy, zz = f['x'][y0:y1, x0:x1, z0:z1], f['y'][y0:y1, x0:x1, z0:z1], f['z'][y0:y1, x0:x1,
                                                                                               z0:z1]
                    else:
                        xx, yy, zz = f['x'][y0:y1, x0:x1, slicez], f['y'][y0:y1, x0:x1, slicez], f['z'][0, 0, slicez]

    tau1 = time_mod.time()
    if verbose:
        print('... time took to load udata in sec: ', tau1 - tau0)

    if return_xy:
        if dim == 2:
            if reverse_x:
                udata[...] = udata[:, :, ::-1, :]
                xx[...] = xx[:, ::-1]
                yy[...] = yy[:, ::-1]

            if reverse_y:
                udata[...] = udata[:, ::-1, :, :]
                xx[...] = xx[::-1, :]
                yy[...] = yy[::-1, :]
            return udata, xx, yy
        elif dim == 3:
            if reverse_x:
                udata[...] = udata[:, :, ::-1, :, :]
                xx[...] = xx[:, ::-1, :]
                yy[...] = yy[:, ::-1, :]
                zz[...] = zz[:, ::-1, :]

            if reverse_y:
                udata[...] = udata[:, ::-1, :, :, :]
                xx[...] = xx[::-1, :, :]
                yy[...] = yy[::-1, :, :]
                zz[...] = zz[::-1, :, :]

            if reverse_z:
                udata[...] = udata[:, :, :, ::-1, :]
                xx[...] = xx[:, :, ::-1]
                yy[...] = yy[:, :, ::-1]
                zz[...] = zz[:, :, ::-1]

            return udata, xx, yy, zz
    else:
        return udata


# Spatial Pofile / radial profile
def get_spatial_profile(xx, yy, qty, xc=None, yc=None, x0=0, x1=None, y0=0, y1=None, n=50,
                        return_center=False,
                        zz=None, zc=None, z0=0, z1=None,
                        method=None,  # if qty contains nan, choose how to clean the array
                        cutoff=np.inf, notebook=True, showtqdm=True
                        ):
    """
    Returns a spatial profile (radial distribution) of 3D object with shape (height, width, duration)
    ... Computes a histogram of a given quantity as a function of distance from the center (xc, yc)
    ... If (xc, yc) are not given, it uses the center of the mass of the quantity at the first frame.

    Parameters
    ----------
    xx: 2/3d array, x-coordinates of the qty, shape (height, width, (depth))
    yy: 2/3d array, y-coordinates of the qty shape (height, width, (depth))
    qty: 3/4D numpy array
    ... energy, enstrophy, etc. with shape (height, width, (depth), duration)
    xc: float
    ... x-coordinate of the origin of the polar coordinate
    yc: float
    ... y-coordinate of the origin of the polar coordinate
    x0: int, default: 0
    ... used to specify a portion of data for computing the histogram. xx[y0:y1, x0:x1]
    x1: int, default: None
    y0: int, default: 0
    y1: int, default: None
    n: int
    ... number of bins for the computed histograms
    return_center: bool, default: False
    zz: 2/3d array, z-coordinates of the qty, shape (height, width, (depth))
    zc: float
    ... y-coordinate of the origin of the polar coordinate
    z0: int, default: 0
    ... used to specify a portion of data for computing the histogram xx[y0:y1, x0:x1, z0:z1]
    z1: int, default: 0
    ... used to specify a portion of data for computing the histogram xx[y0:y1, x0:x1, z0:z1]
    method: str, default: None, method to clean the "qty" array if it contains nan values
        ... 'nn' - nearest neighbor interpolation (recommended)
        ... 'linear' - linear interpolation
        ... 'cubic' - cubic interpolation
    cutoff: float, default: np.inf
        ... |qty| < cutoff will be ignored from the statistics
    notebook: bool, default: True, if True, it will use tqdm_notebook instead of tqdm to display a progress bar
    showtqdm: bool, default: True, if True, it will display a progress bar

    Returns
    -------
    rs: 2d numpy array
    ... radial distance with shape (n, duration)
    qty_ms: 2d numpy array
    ... mean values of the quantity between r and r+dr with shape (n, duration)
    qty_errs: 2d numpy array
    ... std of the quantity between r and r+dr with shape (n, duration)- not standard error
    """
    if notebook: from tqdm import tqdm_notebook as tqdm

    if not np.isinf(cutoff):
        qty[qty > cutoff] = np.nan

    if method is not None:
        print('get_spatial_profile:')
        print('... clean the input quantity (replace np.nans)')
        print('... Cleaning method: ', method)
        qty = clean_data(qty, method=method, fill_value=0)

    if zz is None:
        x, y = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
        if len(qty.shape) == 2:
            qty = qty.reshape(qty.shape + (1,))
        qty_local = qty[y0:y1, x0:x1, ...]

        duration = qty.shape[-1]

        if xc is None or yc is None:
            # find a center of the mass from the initial image
            yc_i, xc_i = ndimage.measurements.center_of_mass(qty_local[..., 0])
            yc, xc = y[int(np.round(yc_i)), int(np.round(xc_i))], x[int(np.round(yc_i)), int(np.round(xc_i))]
            print('spatial_profile: (xc, yc)=(%.2f, %.2f)' % (xc, yc))


        elif xc is None and yc is None:
            xc, yc = 0, 0
        r, theta = cart2pol(x - xc, y - yc)

        shape = (n, duration)
        rs = np.empty(shape)
        qty_ms = np.empty(shape)
        qty_errs = np.empty(shape)
        for t in tqdm(list(range(duration))):
            rs[:, t], qty_ms[:, t], qty_errs[:, t] = get_binned_stats(r.flatten(),
                                                                      qty_local[..., t].flatten(),
                                                                      n_bins=n,
                                                                      return_std=True)
    else:
        x, y, z = xx[y0:y1, x0:x1, z0:z1], yy[y0:y1, x0:x1, z0:z1], zz[y0:y1, x0:x1, z0:z1],
        if len(qty.shape) == 3:
            qty = qty.reshape(qty.shape + (1,))
        qty_local = qty[y0:y1, x0:x1, z0:z1, ...]

        duration = qty.shape[-1]

        if xc is None or yc is None:
            # find a center of the mass from the initial image
            yc_i, xc_i, zc_i = ndimage.measurements.center_of_mass(qty_local[..., 0])

            # xc = np.nanmean(qty_local[..., 0] * x) / np.nanmean(qty_local[..., 0])
            # yc = np.nanmean(qty_local[..., 0] * y) / np.nanmean(qty_local[..., 0])
            # zc = np.nanmean(qty_local[..., 0] * z) / np.nanmean(qty_local[..., 0])

            xc = x[int(np.round(yc_i)), int(np.round(xc_i)), int(np.round(zc_i))]
            yc = y[int(np.round(yc_i)), int(np.round(xc_i)), int(np.round(zc_i))]
            zc = z[int(np.round(yc_i)), int(np.round(xc_i)), int(np.round(zc_i))]
            print('spatial_profile: (xc, yc, zc)=(%.2f, %.2f, %.2f)' % (xc, yc, zc))
        elif xc is None and yc is None and zc is None:
            xc, yc, zc = 0, 0, 0
        r, theta, phi = cart2sph(x - xc, y - yc, z - zc)

        shape = (n, duration)
        rs = np.empty(shape)
        qty_ms = np.empty(shape)
        qty_errs = np.empty(shape)
        for t in tqdm(list(range(duration)), disable=~showtqdm):
            rs[:, t], qty_ms[:, t], qty_errs[:, t] = get_binned_stats(r.flatten(),
                                                                      qty_local[..., t].flatten(),
                                                                      n_bins=n,
                                                                      return_std=True)

    if notebook: from tqdm import tqdm as tqdm

    if not return_center:
        return rs, qty_ms, qty_errs
    else:
        if zz is None:
            return rs, qty_ms, qty_errs, np.asarray([xc, yc])
        else:
            return rs, qty_ms, qty_errs, np.asarray([xc, yc, zc])


def get_radial_profile(*args, **kwargs):
    """See get_spatial_profile for documentation"""
    return get_spatial_profile(*args, **kwargs)


########## Helpers ###########

def fix_udata_shape(udata):
    """
    Adds a singleton dimension to the shape of udata if it does not have one.
    It is better to always have udata with shape (3, height, width, depth, duration) (3D) or  (2, height, width, duration) (2D)
    This method fixes the shape of udata whose shape is (3, height, width, depth) or (2, height, width)

    Parameters
    ----------
    udata: nd array,
          ... with shape (height, width, depth) (3D) or  (height, width, duration) (2D)
          ... OR shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)

    Returns
    -------
    udata: nd array, with shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)

    """
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if shape[0] == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
            return udata
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))
            return np.stack((ux, uy))

    elif shape[0] == 3:
        dim = 3
        ux, uy, uz = udata[0, ...], udata[1, ...], udata[2, ...]
        try:
            nrows, ncols, nstacks, duration = ux.shape
            return udata
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
            return np.stack((ux, uy, uz))

def get_time_from_path(dpath, fps, inc=1, t0=0, t1=None, save=False, overwrite=False):
    """
    Returns real time that corresponds to udata
    ... time = np.arange(t0, t1, inc) / fps

    Parameters
    ----------
    dpath: str, a path where a velocity field is stored
    fps: float, frame rate in FPS
    inc: int, increment
    t0: int, starting frame number
    t1: int, ending frame number
    save: bool, if True, it adds the created 1d array at /t
    overwrite: bool, if True AND save is True, it saves the created 1d array at /t even if the data already exists there.

    Returns
    -------
    realtimes: 1d array, time (in sec)
    """
    duration = get_udata_dim(dpath)[-1]
    if t1 is None: t1 = duration
    timesteps = np.arange(t0, t1, inc)
    realtimes = timesteps / fps
    if save:
        add_data2udatapath(dpath, {'t': realtimes}, overwrite=overwrite)
    return realtimes

def truncateXY(xx, yy, x0=0, x1=None, y0=0, y1=None):
    """
    Returns a truncated grid

    Parameters
    ----------
    xxx: 2d array, x-coordinate
    yyy: 2d array, y-coordinate
    x0: int, index used to truncate a 2d array like xx[y0:y1, x0:x1]
    x1: int, index used to truncate a 2d array like xx[y0:y1, x0:x1]
    y0: int, index used to truncate a 2d array like xx[y0:y1, x0:x1]
    y1: int, index used to truncate a 2d array like xx[y0:y1, x0:x1]

    Returns
    -------
    xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
    """
    return xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]

def truncateXYZ(xxx, yyy, zzz, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    Returns a truncated grid

    Parameters
    ----------
    xxx: 3d array, x-coordinate
    yyy: 3d array, y-coordinate
    zzz: 3d array, z-coordinate
    x0: int, index used to truncate a 3d array like xxx[y0:y1, x0:x1, z0:z1]
    x1: int, index used to truncate a 3d array like xxx[y0:y1, x0:x1, z0:z1]
    y0: int, index used to truncate a 3d array like xxx[y0:y1, x0:x1, z0:z1]
    y1: int, index used to truncate a 3d array like xxx[y0:y1, x0:x1, z0:z1]
    z0: int, index used to truncate a 3d array like xxx[y0:y1, x0:x1, z0:z1]
    z1: int, index used to truncate a 3d array like xxx[y0:y1, x0:x1, z0:z1]

    Returns
    -------
    xxx[y0:y1, x0:x1, z0:z1], yyy[y0:y1, x0:x1, z0:z1], zzz[y0:y1, x0:x1, z0:z1]
    """
    return xxx[y0:y1, x0:x1, z0:z1], yyy[y0:y1, x0:x1, z0:z1], zzz[y0:y1, x0:x1, z0:z1]


def fill_udata(udata, keep, duplicate=True, fill_value=np.nan):
    """
    Replaces values in ~keep with 'filled_value'
    ... For actual masking which does not involve replacing values of the array, use a numpy masked array.

    Parameters
    ----------
    udata: nd array of a v-field
    keep: 2/3d boolean array, where True indicates values to keep
    copy: bool, default: True
    ... If True, the given udata remains untouched.
    Returns
    -------
    vdata: nd array- a masked array
    """
    if duplicate:
        vdata = copy.deepcopy(udata)
    else:
        vdata = udata
    vdata = fix_udata_shape(vdata)
    dim = vdata.shape[0]
    for t in range(vdata.shape[-1]):
        for d in range(dim):
            vdata[d, ..., t][~keep] = fill_value
    return vdata


def mask_udata(udata, mask):
    """
    Returns a numpy.masked array of udata
    ... The shape of the mask must be one of ushape, ushape[1:], ushape[1:-1] where ushape=udata.shape

    Parameters
    ----------
    udata: nd array, v-field data
    mask: md boolean array
    ... The shape of the mask must be one of ushape=udata.shape, ushape[1:], ushape[1:-1] where ushape=udata.shape
    ... The mask must be a boolean array, where True indicates values to hide

    Returns
    -------
    udata: numpy masked nd array

    """
    ushape = udata.shape
    if ushape == mask.shape:
        return np.ma.masked_array(udata, mask)
    else:
        udata = fix_udata_shape(udata)
        d = ushape[0]
        if ushape[1:] == mask.shape:
            rep = [1] * len(ushape)
            rep[0] = d
            mask_ = np.tile(mask[np.newaxis, ...], rep)
            udata = np.ma.masked_array(udata, mask=mask_)
        elif ushape[1:-1] == mask.shape:
            rep = [1] * len(ushape)
            rep[0] = d
            rep[-1] = ushape[-1]
            mask_ = np.tile(mask[np.newaxis, ..., np.newaxis], rep)
            udata = np.ma.masked_array(udata, mask=mask_)
        else:
            raise ValueError('mask shape must be one of these:', ushape, ushape[1:], ushape[1:-1])
    return udata


def get_speed(udata):
    """
    Returns the speed of the v-field

    Parameters
    ----------
    udata: nd array, v-field data

    Returns
    -------
    speed: (n-1)d array, speed of the v-field
    """
    speed = np.zeros_like(udata[0, ...])
    dim = udata.shape[0]
    for d in range(dim):
        speed += udata[d, ...] ** 2
    speed = np.sqrt(speed)
    return speed


def normalize_udata(udata):
    """
    Returns the normalized v-field

    Parameters
    ----------
    udata: nd array, v-field data

    Returns
    -------
    norm_udata: nd array, normalized v-field
    """
    norm_udata = np.zeros_like(udata)
    dim = udata.shape[0]
    speed = get_speed(udata)
    for d in range(dim):
        norm_udata[d, ...] = udata[d, ...] / speed
        norm_udata[d, speed == 0] = 0
    norm_udata[np.isnan(udata)] = 0
    return norm_udata


def expand_dim(arr):
    """
    Returns arr[..., np.newaxis]
    ... shape of arr changes from arr.shape to (arr.shape, 1)

    Parameters
    ----------
    arr: nd array

    Returns
    -------
    arr[..., np.newaxis]
    """

    return arr[..., np.newaxis]


def get_jacobian_xyz_ijk(xx, yy, zz=None):
    """
    Returns diagonal elements of Jacobian between index space basis and physical space
    ... This returns xyz_orientations for get_duidxj_tensor().
    ... Further details can be found in the docstring of get_duidxj_tensor()

    Parameters
    ----------
    xx: 2d/3d array, a grid of x-coordinates
    yy: 2d/3d array, a grid of y-coordinates
    zz: 2d/3d array, a grid of z-coordinates

    Returns
    -------
    jacobian: 1d array
        ... expected input of xyz_orientations for get_duidxj_tensor
    """

    dim = len(xx.shape)
    if dim == 2:
        x = xx[0, :]
        y = yy[:, 0]
        increments = [np.nanmean(np.diff(x)), np.nanmean(np.diff(y))]
    elif dim == 3:
        x = xx[0, :, 0]
        y = yy[:, 0, 0]
        z = zz[0, 0, :]
        increments = [np.nanmean(np.diff(x)), np.nanmean(np.diff(y)), np.nanmean(np.diff(z))]
    else:
        raise ValueError('... xx, yy, zz must have dimensions of 2 or 3. ')

    mapping = {True: 1, False: -1}
    jacobian = np.asarray([mapping[increment > 0] for increment in increments])  # Only diagonal elements
    return jacobian


def compute_rms(qty, axes=None):
    """
    Returns the root mean square of qty

    Parameters
    ----------
    qty: nd array, quantity
    axes: list of ints, axes to compute rms over

    Returns
    -------
    rms: float, root mean square of qty
    """
    rms = np.sqrt(np.nanmean(qty ** 2, axis=axes))
    return rms


def count_nans(arr):
    """Returns the number of nans in the given array"""
    nnans = np.count_nonzero(np.isnan(arr))
    return nnans


def get_equally_spaced_grid(udata, d=1):
    """
    Returns an evenly spaced grid for the given udata

    Parameters
    ----------
    udata: nd array, v-field data
    d: spacing of the grid

    Returns
    -------
    xx, yy, (zz): 2D or 3D numpy arrays
    """
    dim = len(udata)
    if dim == 2:
        height, width, duration = udata[0].shape
        x, y = list(range(width)), list(range(height))
        xx, yy = np.meshgrid(x, y)
        return xx * spacing, yy * d
    elif dim == 3:
        height, width, depth, duration = udata[0].shape
        x, y, z = list(range(width)), list(range(height)), list(range(depth))
        xx, yy, zz = np.meshgrid(x, y, z)
        return xx * d, yy * d, zz * d


def get_equally_spaced_kgrid(udata, d=1):
    """
    Returns a equally spaced grid to plot FFT of udata

    Parameters
    ----------
    udata
    dx: spacing of the grid in the real space

    Returns
    -------
    kxx, kyy, (kzz): 2D or 3D numpy arrays
    """
    dim = len(udata)
    if dim == 2:
        ncomp, height, width, duration = udata.shape
        # k space grid
        kx = np.fft.fftfreq(width, d=d)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=d)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi  # Convert inverse length into wavenumber
        return kxx, kyy
    elif dim == 3:
        ncomp, height, width, depth, duration = udata.shape
        # k space grid
        kx = np.fft.fftfreq(width, d=d)
        ky = np.fft.fftfreq(height, d=d)
        kz = np.fft.fftfreq(depth, d=d)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
        return kxx, kyy, kzz


def get_grid_spacing(xx, yy, zz=None):
    """
    Returns the spacing of the grid

    Index i, j, k corresponds to physical space y, x, z
    ... x = xx[0, :, (0, optional)]
    ... y = yy[:, 0, (0, optional)]
    ... z = zz[0, 0, :]

    Parameters
    ----------
    xx: 2d/3d array, a grid of x-coordinates
    yy: 2d/3d array, a grid of y-coordinates
    zz: 2d/3d array, a grid of z-coordinates

    Returns
    -------
    dx, dy, dz: floats, spacing of the grid along each axis
    """
    dim = len(xx.shape)
    if dim == 2:
        dx = np.abs(xx[0, 1] - xx[0, 0])
        dy = np.abs(yy[1, 0] - yy[0, 0])
        return dx, dy
    elif dim == 3:
        dx = np.abs(xx[0, 1, 0] - xx[0, 0, 0])
        dy = np.abs(yy[1, 0, 0] - yy[0, 0, 0])
        dz = np.abs(zz[0, 0, 1] - zz[0, 0, 0])
        return dx, dy, dz


def get_data_size_in_GB(data):
    "Returns the data size in GB"
    dsize_in_GB = sys.getsizeof(data) / (2 ** 30)  # byte -> GB
    print('%f GB' % dsize_in_GB)
    return dsize_in_GB


def kolmogorov_53(k, k0=50):
    """
    Customizable Kolmogorov Energy spectrum
    Returns the value(s) of k0 * k^{-5/3}

    Parameters
    ----------
    k: array-like, wavenumber: convention is k= 1/L NOT 2pi/L
    k0: float, coefficient

    Returns
    -------
    e_k: power-law spectrum with exponent -5/3 for a given k and k0
    """
    e_k = k0 * k ** (-5. / 3)
    return e_k


def kolmogorov_53_uni(k, epsilon, c=1.6):
    """
    Universal Kolmogorov Energy spectrum
    Returns the value(s) of C \epsilon^{2/3} k^{-5/3}

    Parameters
    ----------
    k: array-like, wavenumber
    epsilon: float, dissipation rate
    c: float, Kolmogorov constant c=1.6 (default)
    ... E(k) = c epsilon^(2/3) k^(-5/3)
    ... E11(k) = c1 epsilon^(2/3) k^(-5/3)
    ... E22(k) = c2 epsilon^(2/3) k^(-5/3)
    ... c1:c2:c = 1: 4/3: 55/18
    ... If c = 1.6, c1 = 0.491, c2 = 1.125
    ... Exp. values: c = 1.5, c1 = 0.5, c2 = ?

    Returns
    -------
    e_k: array-like, Kolmogorov energy spectrum for a given range of k
    """

    e_k = c * epsilon ** (2. / 3) * k ** (-5. / 3)
    return e_k


def model_energy_spectrum(k, epsilon, nu, L, c=1.6, p0=2, cL=6.78, ceta=0.4, beta=5.2):
    """
    Returns a model energy spectrum E(k)=c epsilon^(2/3) * k**(-5/3) f(kL) g(kEta)

    Parameters
    ----------
    k: 1d array, wavenumbers
    epsilon: float/1d array, dissipation rate
    nu: float, viscosity
    L: float, integral length scale
    c: float, kolmogorov constant (default: 1.6)
    p0: float, power in the energy containing regime (default: 2, p0=4: Karman spectrum)
    cL: float, a parameter related to the onset of the inertial range (cL=6.78, Pope p233)
    ceta: float, a parameter related to the onset of the inertial range (ceta=0.4, Pope p233)
    beta: float: strength of the exponential in the dissipation range

    Returns
    -------
    ek: 1d array, energy spectrum
    """

    if cL < 0: raise ValueError('... cL must be a postive constant')
    eta = (nu ** 3 / epsilon) ** 0.25
    fL = lambda k, L, p0: ((k * L) / ((k * L) ** 2 + cL) ** 0.5) ** (5 / 3. + p0)
    feta = lambda k, eta, neta, ceta: np.exp(-beta * ((k * eta) ** 4 + ceta ** 4) ** 0.25 - ceta)
    ek = c * epsilon ** (2 / 3.) * k ** (-5 / 3.) * fL(k, L, p0) * feta(k, eta, beta, ceta)
    return ek


def scaled_model_energy_spectrum(keta, epsilon, nu, L, c=1.6, p0=2, cL=0.1, ceta=0, beta=5.2):
    """
    Returns a model energy spectrum rescaled by eta and epsilon
    ... model spectrum: E(k)=c epsilon^(2/3) * k**(-5/3) f(kL) g(kEta)

    e.g.
        keta = np.logspace(-3, 0)
        plt.plot(keta, vel.scaled_model_energy_spectrum(keta, 1e5, 1.004, 10))

    Parameters
    ----------
    keta: 1d array, wavenumbers * kolmogorov length
    epsilon: float/1d array, dissipation rate
    nu: float, viscosity
    L: float, integral length scale
    c: float, kolmogorov constant (default: 1.6)
    p0: float, power in the energy containing regime (default: 2, p0=4: Karman spectrum)
    cL: float, a parameter related to the onset of the inertial range (cL=6.78, Pope p233)
    ceta: float, a parameter related to the onset of the inertial range (ceta=0.4, Pope p233)
    beta: float: strength of the exponential in the dissipation range

    Returns
    -------
    eks: 1d array, rescaled energy spectrum
    """
    if cL < 0: raise ValueError('... cL must be a postive constant')
    eta = (nu ** 3 / epsilon) ** 0.25
    k = keta / eta
    fL = lambda k, L, p0: ((k * L) / ((k * L) ** 2 + cL) ** 0.5) ** (5 / 3. + p0)
    feta = lambda k, eta, neta, ceta: np.exp(-beta * ((k * eta) ** 4 + ceta ** 4) ** 0.25 - ceta)
    eks = c * keta ** (-5 / 3) * fL(k, L, p0) * feta(k, eta, beta, ceta)
    return eks


def compute_kolmogorov_lengthscale_simple(epsilon, nu):
    """
    Return Kolmogorov length scale for a given set of dissipation rate and viscosity
    Parameters
    ----------
    epsilon: float, dissipation rate
    nu: float, viscosity

    Returns
    -------
    eta, float, Kolmogorov length scale
    """
    eta = (nu ** 3 / epsilon) ** 0.25
    return eta


def get_characteristic_velocity(udata):
    """
    Return 1-component RMS velocity, u'
    ... energy = dim / 2 *  u'^2

    Parameters
    ----------
    udata: nd array, velocity field

    Returns
    -------
    u_irms: 1d array, 1-component RMS velocity
    """
    dim = len(udata)
    u_irms = np.sqrt(2. / dim * get_spatial_avg_energy(udata)[0])
    return u_irms


def kronecker_delta(i, j):
    """
    kronecker_delta Delta function: \delta_{i, j}

    Parameters
    ----------
    i: tensor index
    j: tensor index

    Returns
    -------
    1 if i==j, 0 otherwise

    """
    if i == j:
        return 1
    else:
        return 0


def cart2pol(x, y):
    """
    Transformation: Cartesian coord to polar coord

    Parameters
    ----------
    x: numpy array, x-coord (Cartesian)
    y: numpy array, y-coord (Cartesian)

    Returns
    -------
    r: numpy array, radius (polar)
    phi: numpy array, angle (polar)
    """
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return r, phi


def cart2pol_udata(x, y, udata):
    """
    Transformation: Cartesian coord to polar coord (x, y, (ux, uy)) -> (r, phi, (ur, uphi))
    ... Returns polar coordiantes AND velocity field in the polar basis

    Parameters
    ----------
    x: numpy array, x-coord (Cartesian)
    y: numpy array, y-coord (Cartesian)
    udata, nd array, velocity field

    Returns
    -------
    r: numpy array, radius (polar)
    phi: numpy array, angle (polar)
    udata_pol, nd array, velocity field in the polar basis (ur, uphi)
    """
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    if udata.shape[1:] == x.shape:
        ur = udata[0, ...] * np.cos(phi) + udata[1, ...] * np.sin(phi)
        utheta = -udata[0, ...] * np.sin(phi) + udata[1, ...] * np.cos(phi)
        udata_pol = np.stack((ur, utheta))
    else:
        udata = fix_udata_shape(udata)
        udata_pol = np.empty_like(udata)
        for t in range(udata.shape[-1]):
            udata_pol[0, ..., t] = udata[0, ..., t] * np.cos(phi) + udata[1, ..., t] * np.sin(phi)
            udata_pol[1, ..., t] = -udata[0, ..., t] * np.sin(phi) + udata[1, ..., t] * np.cos(phi)
    return r, phi, udata_pol


def cart2sph(x, y, z):
    """
    Transformation: cartesian to spherical
    z = r cos theta
    y = r sin theta sin phi
    x = r sin theta cos phi

    Parameters
    ----------
    x: numpy array, x-coord (Cartesian)
    y: numpy array, y-coord (Cartesian)
    z: numpy array, z-coord (Cartesian)

    Returns
    -------
    r: radial distance
    theta: polar angle [-pi/2, pi/2] (angle from the z-axis)
    phi: azimuthal angle [-pi, pi] (angle on the x-y plane)
    """
    # hxy = np.hypot(x, y)
    # r = np.hypot(hxy, z)
    # theta = np.arctan2(z, hxy)
    # phi = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def cart2sph_velocity(ux, uy, uz, xx, yy, zz, xc=0, yc=0, zc=0):
    """
    Transformation: cartesian to spherical (x, y, z, (ux, uy, uz)) -> (r, theta, phi, (ur, utheta, uphi))
    ... Returns a velocity field in the spherical basis
    Parameters
    ----------
    ux: numpy array, x-component of velocity field (Cartesian)
    uy: numpy array, y-component of velocity field (Cartesian)
    uz: numpy array, z-component of velocity field (Cartesian)
    xx: numpy array, x-coord (Cartesian)
    yy: numpy array, y-coord (Cartesian)
    zz: numpy array, z-coord (Cartesian)
    xc: float, x-coord of the origin of the spherical cordinate system
    yc: float, y-coord of the origin of the spherical cordinate system
    zc: float, z-coord of the origin of the spherical cordinate system

    Returns
    -------
    ur: numpy array, radial component of velocity field in the spherical basis
    utheta: numpy array, polar component of velocity field in the spherical basis
    uphi: numpy array, azimuthal component of velocity field in the spherical basis
    """
    xx_tmp, yy_tmp, zz_tmp = xx - xc, yy - yc, zz - zc
    R = np.sqrt(xx_tmp ** 2 + yy_tmp ** 2 + zz_tmp ** 2)
    Rxy = np.sqrt(xx_tmp ** 2 + yy_tmp ** 2)
    ur = (xx_tmp * ux + yy_tmp * uy + zz_tmp * uz) / R
    utheta = - (zz_tmp * (xx_tmp * ux + yy_tmp * uy) - Rxy ** 2 * uz) / (R ** 2 * Rxy)
    uphi = (yy_tmp * ux - xx_tmp * uy) / (Rxy ** 2)
    return ur, utheta, uphi


def sph2cart_velocity(ur, utheta, uphi, ttheta, pphi):
    """
    Transformation: spherical to cartesian (r, theta, phi, (ur, utheta, uphi)) -> (x, y, (ux, uy, uz))

    z = r cos theta
    y = r sin theta sin phi
    x = r sin theta cos phi

    Spherical coorrdinates:
    (r, theta, phi) = (radial distance, polar angle, azimuthal angle)
    r: radius
    theta: polar angle [-pi/2, pi/2] (angle from the z-axis)
    phi: azimuthal angle [-pi, pi] (angle on the x-y plane)

    http://www.astrosurf.com/jephem/library/li110spherCart_en.htm

    Parameters
    ----------
    ur: numpy array, radial component of velocity field in the spherical basis
    utheta: numpy array, polar component of velocity field in the spherical basis
    uphi: numpy array, azimuthal component of velocity field in the spherical basis
    ttheta: numpy array, polar angle [-pi/2, pi/2] (angle from the z-axis)
    pphi: numpy array, azimuthal angle [-pi, pi] (angle on the x-y plane)

    Returns
    -------
    ux: numpy array, x-component of velocity field (Cartesian)
    uy: numpy array, y-component of velocity field (Cartesian)
    uz: numpy array, z-component of velocity field (Cartesian)
    """
    st, ct = np.sin(ttheta), np.cos(ttheta)
    sp, cp = np.sin(pphi), np.cos(pphi)
    ux = ur * st * cp + utheta * ct * cp - uphi * sp
    uy = ur * st * sp + utheta * ct * sp + uphi * cp
    uz = ur * ct - utheta * st
    return ux, uy, uz


def sph2cart(r, theta, phi, xc=0, yc=0, zc=0):
    """
    Transformation from spherical to cartesian coordinates

    Parameters
    ----------
    r: radial distance
    theta: polar angle [-pi/2, pi/2] (angle from the z-axis)
    phi: azimuthal angle [-pi, pi] (angle on the x-y plane)
    xc: float, origin at (xc, yc, zc)
    yc: float, origin at (xc, yc, zc)
    zc: float, origin at (xc, yc, zc)

    Returns
    -------
    x, y, z: cartesian coordinates
    """
    x = r * np.sin(theta) * np.cos(phi) + xc
    y = r * np.sin(theta) * np.sin(phi) + yc
    z = r * np.cos(theta) + zc
    return x, y, z


def change_of_basis_mat_sph2cart(theta, phi):
    """
    Returns a change-of-basis matrix from a spherical basis to a cartesian basis

    Parameters
    ----------
    theta: float, polar angle [-pi/2, pi/2] (angle from the z-axis)
    phi: float, azimuthal angle [-pi, pi] (angle on the x-y plane)

    Returns
    -------
    Ms2c: 3d array, change-of-basis matrix from spherical to cartesian
    Example
    -------
    [ax, ay, az] = np.matmul(Ms2c, [ar, atheta, aphi]) # Convert from spherical to cartesian coordinates
    """
    Ms2c = np.asarray([[np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), -np.sin(phi)],
                       [np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), np.cos(phi)],
                       [np.cos(theta), -np.sin(theta), 0]])
    return Ms2c


def change_of_basis_mat_cart2sph(theta, phi):
    """
    Returns a change-of-basis matrix from a cartesian basis to a spherical basis
    Parameters
    ----------
    theta: float, polar angle [-pi/2, pi/2] (angle from the z-axis)
    phi: float, azimuthal angle [-pi, pi] (angle on the x-y plane)

    Returns
    -------
    Mc2s: 3d array, change-of-basis matrix from cartesian to spherical

    Example
    -------
    [ar, atheta, aphi] = np.matmul(Ms2c, [ax, ay, az]) # Convert from cartesian to spherical coordinates
    """
    Mc2s = np.asarray([[np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
                       [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
                       [-np.sin(phi), np.cos(phi), 0]])
    return Mc2s


def pol2cart_udata(r, theta, udata_pol, x0=0, y0=0):
    """
    Transformation: Polar coord to Cartesian coord (r, phi, (ur, uphi)) -> (x, y, (ux, uy))
    ... Returns polar coordiantes AND velocity field in the standard basis

    Parameters
    ----------
    r: numpy array, radial distance
    theta: numpy array, polar angle [-pi/2, pi/2] (angle from the z-axis)
    udata_pol: numpy array, velocity field in the polar basis

    Returns
    -------
    x, y, udata: cartesian coordinates and velocity field in the standard basis
    """
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0

    if udata_pol.shape[1:] == r.shape:
        udata = np.empty_like(udata_pol)
        udata[0, ...] = udata_pol[0, ...] * np.cos(theta) - udata_pol[1, ...] * np.sin(theta)
        udata[1, ...] = udata_pol[0, ...] * np.sin(theta) + udata_pol[1, ...] * np.cos(theta)

    else:
        udata_pol = fix_udata_shape(udata_pol)
        udata = np.empty_like(udata_pol)
        for t in range(udata_pol.shape[-1]):
            udata[0, ..., t] = udata_pol[0, ..., t] * np.cos(theta) - udata_pol[1, ..., t] * np.sin(theta)
            udata[1, ..., t] = udata_pol[0, ..., t] * np.sin(theta) + udata_pol[1, ..., t] * np.cos(theta)
    return x, y, udata


def cart2cyl(x, y, z):
    """
    Transformation: cartesian to cylindrical coord
    z = z
    y = rho sin phi
    x = rho cos phi

    Parameters
    ----------
    x: numpy array, cartesian x-coord
    y: numpy array, cartesian y-coord
    z: numpy array, cartesian z-coord

    Returns
    -------
    rho, phi, z: cylindrical coordinates
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi, z


def cyl2cart(rho, phi, z, x0=0, y0=0):
    """
    Transformation from cylindrical to cartesian coords
    Parameters
    ----------
    rho: numpy array, radial distance (cylindrical)
    phi: numpy array, polar angle [-pi, pi] (angle on the x-y plane)
    z: numpy array, z-coord
    x0: float, (x0, y0, 0) is the origin of the coordinate system
    y0: float, (x0, y0, 0) is the origin of the coordinate system

    Returns
    -------
    x, y, z: cartesian coordinates
    """
    x = rho * np.cos(phi) + x0
    y = rho * np.sin(phi) + y0
    return x, y, z


def cart2cyl_2d(xx, yy, a=0, b=0, x0=None, signed=False):
    """
    Transforms Cartesian (x, y) to Cylindrical (z', rho') coordinates
    ... This function is useful when an axisymmetry is present in the data.
    ... Cartesian-to-Cylindrical coordinate transformation is a mapping between R3 and R3.
        Axisymmetry makes a number of necessary variables to 2 from 3.
        This function transforms (x, y) to (z', rho')
        ... The axis of symmetry is defined by y = a*x + b
        ... If a=b=0, z=x, rho=y

    Parameters
    ----------
    xx: numpy array, x-coord
    yy: numpy array, y-coord
    a: float, the axis of symmetry is given by y = a*x + b
    b: float, the axis of symmetry is given by y = a*x + b
    x0: float, This shifts the origin of the cylindrical coordinate system along the symmetry axis.
    signed: bool, if True, the function returns the signed distance from the axis of symmetry.

    Returns
    -------
    zz, rrho: numpy arrays, cylindrical coordinates
    """

    def get_dst_from_pt_to_line(x0, y0, a, b, c, signed=False):
        """
        Returns a distance between a pt and a line

        Eq of a line: ax + by + c = 0
        Point is at (x0, y0)
        Parameters
        ----------
        x
        y
        a
        b
        c

        Returns
        -------

        """
        if not signed:
            d = np.abs((a * x0 + b * y0 + c)) / np.sqrt(a ** 2 + b ** 2)
        else:
            d = (a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)
        return d

    def get_pt_projected_onto_line(x0, y0, a, b, c):
        """
        Assume a line and a point, and project the point onto the line.
        ... In other words, draw a circle with a radius (x0, y0) with d which is a distance between the pt and the line.
            Then, this function returns the intersection of the circle with the line.
        ... Eq of a line: ax + by + c = 0
            Point is at (x0, y0)

        Parameters
        ----------
        x0
        y0
        a
        b
        c

        Returns
        -------

        """
        d = get_dst_from_pt_to_line(x0, y0, a, b, c, signed=True)
        # theta = np.arctan2(-a, b)
        # x1 = x0 - d * np.cos(theta)
        # y1 = y0 + d * np.sin(theta)

        x1 = x0 - a * d / np.sqrt(a ** 2 + b ** 2)
        y1 = y0 - b * d / np.sqrt(a ** 2 + b ** 2)

        return x1, y1

    def dot(X, Y, axis=None):
        '''Calculate the dot product of two arrays of vectors.

        '''
        X, Y = np.asarray(X), np.asarray(Y)
        if axis is None:
            return (X * Y).sum(-1)
        elif axis is not None:
            dimX = len(X.shape)
            dimY = len(Y.shape)
            if not (dimX == 1 or dimY == 1):
                raise ValueError('At least ne of the input vectors must be 1D!')
            elif dimX == 1 and dimY != 1:
                tmp = X
                X = Y
                Y = tmp
            else:
                pass

            dummy = X[..., np.newaxis]
            dummy = np.swapaxes(dummy, axis, -1)
            dp = np.dot(dummy, Y)
            # Get rid of the axis along which dot product was computed
            dp_sqd = np.squeeze(dp, axis=axis)
            return dp_sqd

    xmin, xmax = np.nanmin(xx), np.nanmax(xx)

    if x0 is None:
        x0 = (xmax + xmin) / 2.
    if np.isinf(a):
        a = 10 ** 8
    y0 = a * x0 + b

    R = np.stack((xx, yy), axis=2)
    alpha = np.asarray([1, a]) / np.sqrt(1 + a ** 2)
    beta = np.asarray([-a, 1]) / np.sqrt(1 + a ** 2)
    rrho = get_dst_from_pt_to_line(xx, yy, a, -1, b, signed=signed)

    if x0 is not None:
        x1, y1 = get_pt_projected_onto_line(0, 0, a, -1, b)
        r1 = np.asarray([x1 - x0, y1 - y0])
        z_offset = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        if dot(r1, alpha) < 0:
            z_offset *= -1
    else:
        z_offset = 0
    zz = dot(R, alpha) + z_offset

    return zz, rrho


def cart2cyl_2d_udata(udata, xx, yy, a=0, b=0, x0=None, signed=False):
    """

    Transforms a velocity field from the Cartesian (x, y) to Cylindrical (z', rho') basis
    ... This function is useful when an axisymmetry is present in the data.
    ... Cartesian-to-Cylindrical coordinate transformation is a mapping between R3 and R3.
        Axisymmetry makes a number of necessary variables to 2 from 3.
        This function transforms (x, y) to (z', rho')
        ... The axis of symmetry is defined by y = a*x + b
        ... If a=b=0, z=x, rho=y

    Parameters
    ----------
    xx: numpy array, x-coord
    yy: numpy array, y-coord
    a: float, the axis of symmetry is given by y = a*x + b
    b: float, the axis of symmetry is given by y = a*x + b
    x0: float, This shifts the origin of the cylindrical coordinate system along the symmetry axis.
    signed: bool, if True, the function returns the signed distance from the axis of symmetry.

    Returns
    -------
    udata_cyl: numpy arrays, velocity field in cylindrical coordinates

    """
    udata_cyl = np.empty_like(udata)

    R = np.stack((xx, yy), axis=2)
    alpha = np.asarray([1, a]) / np.sqrt(1 + a ** 2)
    beta = np.asarray([-a, 1]) / np.sqrt(1 + a ** 2)

    theta = np.arctan2(a, 1)
    udata_cyl[0, ...] = udata[0, ...] * np.cos(theta) + udata[1, ...] * np.sin(theta)
    udata_cyl[1, ...] = udata[0, ...] * np.sin(theta) - udata[1, ...] * np.cos(theta)

    return udata_cyl


def natural_sort(arr):
    """
    natural-sorts elements in a given array
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    e.g.-  arr = ['a28', 'a01', 'a100', 'a5']
    ... WITHOUT natural sorting,
     -> ['a01', 'a100', 'a28', 'a5']
    ... WITH natural sorting,
     -> ['a01', 'a5', 'a28', 'a100']


    Parameters
    ----------
    arr: list or numpy array of strings

    Returns
    -------
    sorted_array: natural-sorted
    """

    def atoi(text):
        'natural sorting'
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    return sorted(arr, key=natural_keys)


def find_nearest(array, value, option='auto'):
    """
    Find an element and its index closest to 'value' in 'array'

    Parameters
    ----------
    array: nd array
    value: float/int
    option: str, 'auto' or 'less', or 'greater'
        ... 'auto' will return the element closest to 'value'
        ... 'less' will return the element closest to 'value' that is LESS than 'value'
        ... 'greater' will return the element closest to 'value' that is GREATER than 'value'

    Returns
    -------
    idx: index of the array where the closest value to 'value' is stored in 'array'
    array[idx]: the closest value in 'array'
    """
    # get the nearest value such that the element in the array is LESS than the specified 'value'
    if option == 'less':
        array_new = copy.copy(array)
        array_new[array_new > value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    # get the nearest value such that the element in the array is GREATER than the specified 'value'
    if option == 'greater':
        array_new = copy.copy(array)
        array_new[array_new < value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    else:
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]


def low_pass_filter(data, fc, fs, order=5):
    """
    Apply low-pass filter to a 1D data (signal)
    ... the signal must be sampled evenly

    Parameters
    ----------
    data: 1d array-like
    fc: float, cutoff frequency
    fs: float, sampling frequency of data

    Returns
    -------
    filtered: 1d array, low-pass filtered signal
    """
    if fc > fs / 2:
        print('low_pass_filter: cutoff freq must be smaller than a half of the sampling frequency')

    w = fc / (fs / 2)  # Normalize the cut-offfrequency

    b, a = signal.butter(order, w, 'low')
    filtered = signal.filtfilt(b, a, data)  # Butterworth digital filter
    return filtered


def get_running_avg_1d(x, t, notebook=True):
    """
    Calculate the running average of a 1D array

    Parameters
    ----------
    x: 1d array
    t: float, number of time steps to average over
    notebook: bool, if True, it uses tqdm_notebook instead of tqdm to show a progress bar

    Returns
    -------
    y: 1d array, running average of x
    """
    if t == 1:
        return x
    else:
        if notebook:
            from tqdm import tqdm_notebook as tqdm
            print('Using tqdm_notebook. If this is a mistake, set notebook=False')
        else:
            from tqdm import tqdm

        y = np.zeros(len(x) - t)
        for i in tqdm(list(range(t))):
            y += np.roll(x, -i)[:-t]
        y /= float(t)

        if notebook:
            from tqdm import tqdm
        return y


def get_running_avg_nd(udata, t, axis=-1, notebook=True):
    """
    Calculate the running average of a nD array

    Parameters
    ----------
    udata: nd array, a vector field
    t: float, number of time steps to average over
    axis: int, axis along which to average
    notebook: bool, if True, it uses tqdm_notebook instead of tqdm to show a progress bar

    Returns
    -------
    vdata: nd array, running average of udata along the specified axis
    """
    if t == 1:
        return udata
    else:
        if notebook:
            from tqdm import tqdm_notebook as tqdm
            print('Using tqdm_notebook. If this is a mistake, set notebook=False')
        else:
            from tqdm import tqdm
        shape = udata.shape
        newshape = tuple(list(shape)[:-1] + [shape[-1] - t])
        vdata = np.zeros(newshape)
        for i in tqdm(list(range(t)), desc='Computing running average (nd)'):
            # if array contains nan, nans will be replaced by 0.
            vdata += np.nan_to_num(np.roll(udata, -i, axis=axis)[..., :-t])
        vdata /= float(t)

        if notebook:
            from tqdm import tqdm
        return vdata



def get_phase_average(x, period_ind=None,
                      time=None, freq=None, nbins=100,
                      axis=-1, return_std=True, use_masked_array=True):
    """
    Returns phase average of a ND array (generalization of get_average_data_from_periodic_data)
    ... Assume x is a periodic data, and you are interested in averaging data by locking the phase.
       This function returns time (a cycle), phase-averaged data, std of the data
    ... Two methods to do this
        1. This is easy IF data is evenly spaced in time
        ... average [x[0], x[period], x[period*2], ...],
           then average [x[1], x[1+period], x[1+period*2], ...], ...
        2. Provide a time array as well as data.
            ... For example, one can give unevenly spaced data (ND array) in time
               one can take phase average by taking a histogram appropriately
    ... Required arguments for each method:
        1. x, period_ind
        2. x, time, freq

    Parameters
    ----------
    x: ND array, data
        ... one of the array shape must must match len(time) if time is given
    period_ind: int
        ... period in the unit of index
    time: 1d array, default: None
        ... time of data
    freq: float
        ... frequency of the periodicity of the data
    nbins: int
        ... number of points to probe data in the period
    axis: int, default:-1
        ... axis number to specify the temporal axis of the data
    return_std: bool, default: True
        ... If False, it returns the standard error instead of standard deviation

    Returns
    -------
    t_p: time (a single cycle)
        ... For the method 1, it returns np.arange(nbins)
    x_pavg: phase-averaged data (N-1)D array
    x_perr: std of the data by phase averaging (N-1)D array or standard error
    """

    x = np.asarray(x)

    if time is not None:
        time = np.asarray(time) - np.nanmin(time)

    if freq is None and period_ind is None:
        raise ValueError('... freq OR period_ind must be given to compute a phase average. Exiting...')
    if freq is not None and period_ind is not None:
        raise ValueError(
            'Both freq and period_ind were given! This is invalid. Specify the period of the given data by specifying one of them!')
    if period_ind is not None and freq is None:
        # Phase average when phase is specified by indices
        # ... Handy if data is taken at constant time
        # ... residue = indices mod period_ind
        # ... it averages [x[residue], x[residue + period_ind], x[residue + period_ind*2], x[residue + period_ind*3], ...]

        t_p = np.arange(period_ind)

        shape_pavg = list(x.shape)
        del shape_pavg[axis]
        shape_pavg += [period_ind]
        x_pavg = np.empty(shape_pavg)
        x_perr = np.empty(shape_pavg)
        for i in range(period_ind):
            indices = range(i, x.shape[axis], period_ind)
            x_pavg[..., i] = np.nanmean(x.take(indices=indices, axis=axis))
            if return_std:
                x_perr[..., i] = np.nanstd(x.take(indices=indices, axis=axis))
            else:
                x_perr[..., i] = np.nanstd(x.take(indices=indices, axis=axis)) / np.sqrt(len(indices))
        x_pavg = np.swapaxes(x_pavg, axis, -1)
        x_perr = np.swapaxes(x_perr, axis, -1)

    if freq is not None and period_ind is None:
        time, x = np.asarray(time), np.asarray(x)
        time_mod = time % (1. / freq)

        period = 1. / freq
        dt = period / nbins
        t_p = np.arange(nbins) * dt + dt / 2.

        shape_pavg = list(x.shape)
        del shape_pavg[axis]
        shape_pavg += [nbins]
        x_pavg = np.empty(shape_pavg)
        x_perr = np.empty(shape_pavg)

        for i in range(nbins):
            tmin, tmax = t_p[i] - dt / 2., t_p[i] + dt / 2
            keep1, keep2 = time_mod >= tmin, time_mod < tmax
            keep = keep1 * keep2
            indices = np.arange(x.shape[axis])[keep]
            x_pavg[..., i] = np.nanmean(x.take(indices=indices, axis=axis), axis=axis)
            if return_std:
                x_perr[..., i] = np.nanstd(x.take(indices=indices, axis=axis), axis=axis)
            else:
                x_perr[..., i] = np.nanstd(x.take(indices=indices, axis=axis), axis=axis) / np.sqrt(len(indices))
        x_pavg = np.swapaxes(x_pavg, axis, -1)
        x_perr = np.swapaxes(x_perr, axis, -1)

    if use_masked_array:
        x_pavg = np.ma.masked_array(x_pavg)
        x_perr = np.ma.masked_array(x_perr)

    return t_p, x_pavg, x_perr


def get_phase_averaged_udata_from_path(dpath, freq, time, deltaT=None,
                                       x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                       t0=0, t1=None, notebook=True, use_masked_array=True):
    """
    Returns the phase-averaged udata (velocity field) without loading the entire udata from dpath

    Parameters
    ----------
    dpath: str, path to the udata (h5)
        ... the h5 file must contain ux and uy at /ux and /uy, respectively
    freq: float, frequency of the peridocity in data
    time: 1D array, time
        ... time unit must be the same as the inverse of the unit of 'freq'
    deltaT: float, default: None
        ... time window of a phase used for averaging
        ... data in [T, T+deltaT), [2T, 2(T+deltaT)], [3T, 3(T+deltaT)], ... will be considered to be at the same phase
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])

    Returns
    -------
    tp: 1D array, time of phase-averaged data
        ... tp = [0, period]
        ... phase = tp * freq
    udata_pavg: 2D array, phase-averaged udata
        ... This is a phase-dependent mean flow.
    """

    duration = get_udata_dim(dpath)[-1]
    if len(time) != duration:
        raise ValueError("get_phase_averaged_udata_from_path: the length of t must match the duration of the udata")

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    time = np.asarray(time[t0:t1])
    period = 1. / freq  # period T
    if t1 is None:
        t1 = duration
    if deltaT is None:
        deltaT = period
    time_mod_period = time % period
    nt = int(np.floor(period / deltaT))
    tp = time[:nt]
    try:
        dummy, xxx, yyy, zzz = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=0, t1=1,
                                                   return_xy=True, verbose=False)
    except:
        dummy, xxx, yyy = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=0, t1=1,
                                              return_xy=True, verbose=False)
    shape = dummy.shape[:-1] + (nt,)
    udata_pavg = np.zeros(shape)

    for i in tqdm(range(nt)):
        cond1 = time_mod_period >= i * deltaT
        cond2 = time_mod_period < (i + 1) * deltaT
        cond = cond1 * cond2  # If True, load data
        counter = 0
        for t in range(t0, t1):
            if cond[t]:
                udata_pavg[..., i] += \
                get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=t, t1=t + 1,
                                    verbose=False)[..., 0]
                counter += 1
        udata_pavg[..., i] /= float(counter)

    if use_masked_array:
        udata_pavg = np.ma.masked_array(udata_pavg)

    if notebook:
        from tqdm import tqdm
    return tp, udata_pavg


def write_hdf5_dict(filename, datadict, attrdict=None, overwrite=False, verbose=True):
    """
    Writes data in data_dict = {'varname0': var0, 'varname1': var1, ...} to a h5 file
    - A quick function to write multiple data to a h5 file

    Parameters
    ----------
    filepath: str
        ... file path where data will be stored. (Do not include extension- .h5)
    datadict: dict, a collection of data to be stored in the h5 file
        ... data_dict = {'varname0': var0, 'varname1': var1, ...}
        ... data will be stored in the h5 file under the group /varname0, /varname1, ...
            and can be accessed by
            ... with h5py.File(filepath, mode='r'):
                    data = hf[key][:]
    attrdict: dict, default: None, metadata of the data
        ... Attributes of data to be stored in the h5 file
        ... Organize the attributes of a variable like this:
            attrdict = {'varname0': {'attr0': value0, 'attr1': value1, ...}, ...}
            ... attrdict = {'piv000': {'int_win': 32, 'date': 01012020, 'software': 'DaVis'},
                            'piv001': {'int_win': 16, 'date': 01012020, 'software': 'PIVLab'},
                            ...}
    bool: overwrite, default: False
    verbose: bool, default: True
        ... print where the data is stored

    Returns
    -------
    None
    """
    filedir = os.path.split(filename)[0]
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    ext = '.h5'
    if not filename[-3:] == ext:
        filename = filename + ext

    with h5py.File(filename, mode='a') as hf:
        for key in datadict:

            if key in hf.keys() and overwrite:
                del hf[key]
            try:
                hf.create_dataset(key, data=datadict[key])
            except RuntimeError:
                if overwrite:
                    del hf[key]
                    hf.create_dataset(key, data=datadict[key])
                else:
                    if verbose:
                        print(key, ' already exists in the h5 file! It will NOT be overwriting the existing data')

        if attrdict is not None:
            for key in attrdict:
                for attrname, item in attrdict[key].items():
                    hf[key][attrname] = item

    if verbose:
        print('Data was successfully saved at ' + filename)


def read_simple_hdf5(datapath, grpname=None):
    """
    Returns a dictionary of data in a hdf5
    1. ASSUMED DATA ARCHITECTURE (DEFAULT)
    ... The hdf5 file must have a simple structure.
        ... /DataA
        ... /DataB
        ... /DataC
    ... The returned dictionary has a structure like following.
        datadit['DataA'] = values of DataA
        datadit['DataB'] = values of DataB ...
    2. OPTIONAL ARCHITECTURE
    ... It is possible to read a COLLECTION of simple data
        ... /grp0/DataA
        ... /grp0//DataB
        ... /grp0//DataC
        ... /grp1/DataA
        ... /grp1//DataB
        ... /grp1//DataC

    Parameters
    ----------
    datapath: str, path to the hdf5 file
    grpname: str, default: None
        ... 'grpname' is the name of the group in the h5 nameto be read.
    Returns
    -------
    datadict: dict
        ... datadict['DataA'] = values of /DataA
    """
    datadict = {}
    with h5py.File(datapath, 'r') as data:
        if grpname is not None:
            data = data[grpname]
        keys = list(data.keys())
        for key in keys:
            datadict[key] = data[key][...]
        print('Keys of the returning dictionary: ', keys)
    return datadict


def add_data2udatapath(udatapath, datadict, attrdict=None, grpname=None, overwrite=False, verbose=True):
    """
    Writes a data stored in a dictionary into a hdf5 at udatatapth
    ... datadict = {"name1": value1, "name2":value2, ...} will be stored like /name1, /name2 in the hdf5 file
    ... datadict = {"grp1": {"name1": value1, "name2":value2, ...},
                    "grp1": {"name1": value1, "name2":value2, ...}, ... }
        will be stored like /grp1/name1, /grp1/name2, /grp2/name1, /grp2/name2, ...

    Parameters
    ----------
    udatapath: str, a path to the hdf5 file
    datadict: dictionary, or a nested dictionary (up to level 2)
        ... data must be stored like {"name1": value1, "name2": value2, ...}
    grpname: str
        ... if given, it saves the datadict like
            /grpname/name1, /grpname/name2
    overwrite: bool, if True, it overwrite the data in the target hdf5 file
    verbose: bool, if True, it prints out details during saving the data

    Returns
    -------
    None
    """
    if grpname is not None:
        if verbose:
            print('add_data2udatapath(): data will be saved like /%s/names' % grpname)
            print('... the given datadict must not be a nested dictionary')
        datadict = {grpname: datadict}

    restricted_keys = ['ux', 'uy', 'uz', 'x', 'y', 'z']  # these keys will be ignored to protect original udata

    # check if datadict contains data that must not be overwritten
    new_keys = [key for key in datadict.keys() if key not in restricted_keys]

    if not os.path.exists(os.path.split(udatapath)[0]):
        os.makedirs(os.path.split(udatapath)[0])
    with h5py.File(udatapath, mode='a') as f:
        existing_keys = f.keys()
        for new_key in new_keys:
            if overwrite and type(datadict[new_key]) != dict:
                try:
                    if verbose:
                        print('add_data2udatapath(): Adding %s...' % new_key)
                    f.create_dataset(new_key, data=datadict[new_key])
                except:
                    del f[new_key]
                    f.create_dataset(new_key, data=datadict[new_key])
                    if verbose:
                        print('add_data2udatapath(): %s already exists. Overwriting...' % new_key)
            else:
                if new_key in existing_keys and type(datadict[new_key]) != dict:
                    if verbose:
                        print('add_data2udatapath(): %s already exists! Skipping...' % new_key)
                else:
                    if type(datadict[new_key]) == dict:
                        if not new_key in existing_keys:
                            grp = f.create_group('/%s/' % new_key)
                        else:
                            grp = f[new_key]
                        subkeys = datadict[new_key].keys()
                        for subkey in subkeys:
                            try:
                                grp.create_dataset(subkey, data=datadict[new_key][subkey])
                                if verbose:
                                    print('add_data2udatapath(): Adding /%s/%s...' % (
                                        new_key, subkey))
                            except:
                                if overwrite:
                                    del grp[subkey]
                                    grp.create_dataset(subkey, data=datadict[new_key][subkey])
                                    print('add_data2udatapath(): /%s/%s already exists. Overwriting...' % (
                                        new_key, subkey))
                                else:
                                    if verbose:
                                        print('add_data2udatapath(): /%s/%s already exists! Skipping...' % (
                                            new_key, subkey))
                    else:
                        f.create_dataset(new_key, data=datadict[new_key])

        if attrdict is not None:
            for key in attrdict:
                for attrname, item in attrdict[key].items():
                    if attrname not in f[key].attrs or overwrite:
                        print(f'add_data2udatapath(): {key}.attrs[\'{attrname}\'] already exists. Overwriting...')
                        f[key].attrs[attrname] = item
                    else:
                        print(f'add_data2udatapath(): {key}.attrs[\'{attrname}\'] already exists. Skipping...')


def convert_dat2h5files(dpath, savedir=None, verbose=False, overwrite=True, start=0):
    """
    Converts tecplot data files (.data format) to a set of hdf5 files
    ... Used for DaVis STB Lagrangian data

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
        savepath = os.path.join(savedir, os.path.split(dpath)[1])

        COLUMNS = ["x", "y", "z", "I", "u", "v", "w", "|V|", "trackID", "ax", "ay", "az",
                   "|a|"]  # vel and acc are in m/s or m/s^2

        skiprows = 6
        fn = start
        ln = 0  # Line count

        # initialization
        data_lists = [[] for column in COLUMNS]
        with open(dpath, 'r') as f:
            while fn < 500:
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


def get_udata_dim(udatapath):
    "Returns the array size of ux stored in the given path"
    with h5py.File(udatapath, 'r') as f:
        shape = f['ux'].shape
    return shape


def get_udata_phys_dim(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    Returns the physical dimensions of the velocity field stored in the standard format in the given path
    ... the h5 file must contain the following datasets:
        - ux at /ux
        - uy at /uz
        - uz at /uy
        - x at /x
        - y at /y
        - z at /z
     width, height, and the depth of the udata in physical dimensions (probably in mm)

    Parameters
    ----------
    udatapath: str, path to the h5 file containing the velocity field
    x0: int, default: 0
        ... index to specify volume of data (xx[y0:y1, x0:x1, z0:z1])
    x1: int, default: 0
        ... index to specify volume of data (xx[y0:y1, x0:x1, z0:z1])
    y0: int, default: 0
        ... index to specify volume of data (xx[y0:y1, x0:x1, z0:z1])
    y1: int, default: 0
        ... index to specify volume of data (xx[y0:y1, x0:x1, z0:z1])
    z0: int, default: 0
        ... index to specify volume of data (xx[y0:y1, x0:x1, z0:z1])
    z1: int, default: 0
        ... index to specify volume of data (xx[y0:y1, x0:x1, z0:z1])

    Returns
    -------
    w, h, (d- if applicable): float, physical dimensions of the velocity field (width, height, depth)
    """
    if x1 is None: x1 = -1
    if y1 is None: y1 = -1
    if z1 is None: z1 = -1

    xxx, yyy, zzz = read_data_from_h5(udatapath, ['x', 'y', 'z'])
    if zzz is not None:
        w, h, d = xxx[y0, x1, z0] - xxx[y0, x0, z0], yyy[y0, x0, z0] - yyy[y1, x0, z0], zzz[y0, x0, z1] - zzz[
            y0, x0, z0]
        w, h, d = np.abs(w), np.abs(h), np.abs(d)
        return w, h, d
    else:
        w, h = xxx[y0, x1] - xxx[y0, x0], yyy[y0, x0] - yyy[y1, x0]
        w, h = np.abs(w), np.abs(h)
        return w, h


def suggest_udata_dim2load(dpath, p=1., n=5, show=True, return_tuple=False, return_None=True):
    """
    Returns a dictionary of inds = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}
    which can be used to load udata via get_udata_from_path(..., **inds)
    ... Estimating a reasonable volume for analysis is crucial to reduce the computation time since inpainting data is the
    rate-limiting step most of the time.
    ... Estimating the reasonable volume in udata is done by counting the number of nans in the data.

    Parameters
    ----------
    dpath: str, path to a udata (h5)
    p: float, param to determine the reasonable volume to load, default:1
    ... domain: 0 < p < 2
    ... The higher p, it returns a bigger volume.
        ... usually, STB-generated udata contains slices with all nans along z.
        ... You do not want to inpaint this!
        ... So keep p around 1 to be reasonable.
    n: int, number of time slices used for the estimation, default: 5
    ... usually, the nan distribution does not vary much, so use only a couple of slices
    show: bool, default:True
    ... If True, show nan distribution along each axis
    return_tuple: bool, default: False
    ... If one wants (x0, x1, y0, y1, z0, z1) instead of a dictionary for some reason, set this True.
    ... Typically, one likes to pass this information to a function with the kwargs ("x0", "x1", etc.),
    so simply pass the default output (which is a dictionary) like **dictionary to unpack.

    Returns
    -------
    ind_dict: dict, indices that can be used to load udata
         ... ind_dict = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}
         ... e.g.- get_udata_from_path(..., **ind_dict)
    """
    try:
        height, width, depth, duration = get_udata_dim(dpath)
        dim = 3
    except:
        height, width, duration = get_udata_dim(dpath)
        dim = 2

    inc = int(duration / n)
    # fractional number of nans
    nx = count_nans_along_axis(dpath, axis='x', inc=inc)
    ny = count_nans_along_axis(dpath, axis='y', inc=inc)
    lx, ly = len(nx), len(ny)

    if dim == 3:
        nz = count_nans_along_axis(dpath, axis='z', inc=inc)
        lz = len(nz)
        z0, _ = find_nearest(nz[:int(lz / 2)], (np.nanmin(nz) + np.nanmax(nz)) / 2. * p)
        z1, _ = find_nearest(nz[int(lz / 2):], (np.nanmin(nz) + np.nanmax(nz)) / 2. * p)
        z1 += int(lz / 2)
    else:
        z0, z1 = 0, None

    x0, _ = find_nearest(nx[:int(lx / 2)], (np.nanmin(nx) + np.nanmax(nx)) / 2. * p)
    x1, _ = find_nearest(nx[int(lx / 2):], (np.nanmin(nx) + np.nanmax(nx)) / 2. * p)
    y0, _ = find_nearest(ny[:int(ly / 2)], (np.nanmin(ny) + np.nanmax(ny)) / 2. * p)
    y1, _ = find_nearest(ny[int(ly / 2):], (np.nanmin(ny) + np.nanmax(ny)) / 2. * p)

    if x1 == 0:
        x1 = width
    else:
        x1 += int(lx / 2)
    if y1 == 0:
        y1 = height
    else:
        y1 += int(ly / 2)

    if show:
        import tflow.graph as graph
        fig, ax = graph.plot(nx, label='x', subplot=121)
        fig, ax = graph.plot(ny, label='y', subplot=121)

        graph.axvline(ax, x=x0, color='C0')
        graph.axvline(ax, x=x1, color='C0')

        graph.axvline(ax, x=y0, color='C1')
        graph.axvline(ax, x=y1, color='C1')

        if dim == 3:
            fig, ax = graph.plot(nz, label='z', subplot=121, figsize=(17, 8))
            graph.axvline(ax, x=z0, color='C2')
            graph.axvline(ax, x=z1, color='C2')

        ax.legend()

        nx_new = count_nans_along_axis(dpath, axis='x', inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        ny_new = count_nans_along_axis(dpath, axis='y', inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

        fig, ax2 = graph.plot(nx_new, label='x', subplot=122, figsize=(17, 8))
        fig, ax2 = graph.plot(ny_new, label='y', subplot=122, figsize=(17, 8))
        graph.labelaxes(ax, 'index j', '# of nans / total')
        graph.labelaxes(ax2, 'index i', '# of nans / total')

        if dim == 3:
            nz_new = count_nans_along_axis(dpath, axis='z', inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
            fig, ax2 = graph.plot(nz_new, label='z', subplot=122, figsize=(17, 8))

        ax.legend()

    if dim == 3:
        print('... Suggested volume (x0, x1, y0, y1, z0, z1) = (%d, %d, %d, %d, %d, %d)' % (x0, x1, y0, y1, z0, z1))
    else:
        print('... Suggested volume (x0, x1, y0, y1, z0, z1) = (%d, %d, %d, %d, %d, None)' % (x0, x1, y0, y1, z0))

    if not return_None:
        if x1 is None: x1 = width
        if y1 is None: y1 = height
        if dim == 3:
            if z1 is None: z1 = depth
        else:
            if z1 is None: z1 = -1

    if return_tuple:
        return x0, x1, y0, y1, z0, z1
    else:
        # Return x0,... in a dictionary- one can pass this to get_udata_from_path(..., **ind_dict)
        ind_dict = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}
        return ind_dict

# functions to derive major quantities of turbulence from udata and save it into a hdf5 format
def derive_all(udata, dx, dy, savepath, udatapath='none', **kwargs):
    """
    A shortcut function to derive major quantities of turbulence

    Parameters
    ----------
    udata: nd array, velocity field
    dx: float, spacing along x
    dy: float, spacing along y
    savepath: str, path where the derived quantities are saved
    kwargs: kwargs that are passed to both derive_easy() and derive_hard()
        ... e.g. x0, x1, y0, y1, z0, z1, t0, t1

    Returns
    -------

    """
    print('... Derive quantities which require small computational power')
    derive_easy(udata, dx, dy, savepath, udatapath=udatapath, **kwargs)
    print('... Derive quantities which require high computational power')
    derive_hard(udata, dx, dy, savepath, **kwargs)


def derive_easy(udata, dx, dy, savepath, time=None, inc=1,
                dz=None, nu=1.004,
                x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                reynolds_decomp=False,
                udatapath='none',
                notebook=True, overwrite=True):
    """
    Function to derive quantities which are (relatively) computationally cheap
    ... energy spectra, energy, enstrophy, skewness etc.
    Parameters
    ----------
    udata: nd array, velocity field
    dx: float, spacing along x
    dy: float, spacing along y
    savepath: str, path where the derived quantities are saved
    time: 1d array, time in physical units
    inc: int, increment along time
        ... if inc=1, then it uses the whole time series to derive various quantities (default)
        ... if inc=n, then it uses every nth time point to derive various quantities (this speeds up the process)
    dz: float, spacing along z (3d data only)
    nu: float, kinematic viscosity
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    reynolds_decomp: bool, default: False
        ... if True, then it uses the fluctuating velocity instead of raw velocity to compute various quantities
    notebook: bool, default: True
        ... if True, then it uses the tqdm_notebook instead of tqdm to display a progress bar
    overwrite: bool, default: True
        ... if True, then it overwrites the existing data in savepath

    Returns
    -------
    None
    """
    if reynolds_decomp:
        savepath += '_rd'  # mark if user decided to do reynolds decomposition before the whole analysis

    keys = ['time', 'e_savg', 'e_savg_err', 'enst_savg', 'enst_savg_err', 'epsilon_sij',
            'e11', 'e22', 'e11_err', 'e22_err', 'k11', 'ek', 'ek_err', 'kr',
            'e11s', 'e11_errs_s', 'k11_s', 'e22s', 'e22_errs_s', 'k22_s', 'ek_s', 'ek_err_s', 'kr_s',
            'lambda_f_iso', 'lambda_g_iso', 're_lambda_iso',
            'L', 'u_L', 'tau_L', 'u_lambda_iso', 'tau_lambda_iso',
            'eta', 'u_eta', 'tau_eta',
            'skewness', 'kurtosis']
    derive = not is_data_derived(savepath + '.h5', keys)

    if derive or overwrite:
        udata = fix_udata_shape(udata)
        dim = len(udata)

        if dim == 2:
            udata = udata[:, y0:y1, x0:x1, t0:t1]
        elif dim == 3:
            if z1 is None:
                z1 = udata[0].shape[2]
            udata = udata[:, y0:y1, x0:x1, z0:z1, t0:t1]
        if time is None:
            time = np.arange(udata.shape[-1])[::inc]

        if reynolds_decomp:
            udata_m, udata_t = reynolds_decomposition(udata)
            vdata = udata_t[..., ::inc]
        else:
            vdata = udata[..., ::inc]

        # PROCESSES WHICH REQUIRE RELATIVELY MINOR COMPUTATIONS

        # <Energy>_space vs time
        e_savg, e_savg_err = get_spatial_avg_energy(vdata, x0=x0, x1=x1, y0=y1, z0=z0, z1=z1)

        # <Enstrophy>_space vs time
        enst_savg, enst_savg_err = get_spatial_avg_enstrophy(vdata, x0=x0, x1=x1, y0=y1, z0=z0, z1=z1, dx=dx, dy=dy,
                                                             dz=dz)

        # dissipation rate vs time
        epsilon_sij = get_epsilon_using_sij(vdata, dx=dx, dy=dy, dz=dz, nu=nu, x0=x0, x1=x1, y0=y1, z0=z0)

        # energy spectra
        (e11, e22), (e11_err, e22_err), k11 = get_1d_energy_spectrum(vdata, x0=x0, x1=x1, y0=y1, z0=z0, z1=z1, dx=dx,
                                                                     dy=dy, dz=dz)
        ek, ek_err, kr = get_energy_spectrum(vdata, x0=x0, x1=x1, y0=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz,
                                             window='flattop', notebook=notebook)

        # scaled energy spectra
        e11s, e11_errs_s, k11_s = scale_energy_spectrum(e11, k11, epsilon=epsilon_sij, nu=nu, e_k_err=e11_err)
        e22s, e22_errs_s, k22_s = scale_energy_spectrum(e11, k11, epsilon=epsilon_sij, nu=nu, e_k_err=e22_err)
        ek_s, ek_err_s, kr_s = scale_energy_spectrum(ek, kr, epsilon=epsilon_sij, nu=nu, e_k_err=ek_err)

        # isotropic formulae
        lambda_f_iso, lambda_g_iso = get_taylor_microscales_iso(vdata, epsilon_sij, nu=nu)
        re_lambda_iso = get_taylor_re_iso(vdata, epsilon_sij, nu=nu)

        # lengthscales
        L, u_L, tau_L = get_integral_scales_all(vdata, dx, dy, dz, nu=nu)
        lambda_g_iso, u_lambda_iso, tau_lambda_iso = get_taylor_microscales_all_iso(vdata, epsilon_sij, nu=nu)
        eta, u_eta, tau_eta = get_kolmogorov_scales_all(vdata, dx, dy, dz, nu=nu)

        # skewness, kurtosis
        skewness = get_skewness(vdata, x0=x0, x1=x1, y0=y1, z0=z0, z1=z1)
        kurtosis = get_kurtosis(vdata, x0=x0, x1=x1, y0=y1, z0=z0, z1=z1)

        # keys = ['time', 'e_savg', 'e_savg_err', 'enst_savg', 'enst_savg_err', 'epsilon_sij',
        #         'e11', 'e22', 'e11_err', 'e22_err', 'k11', 'ek', 'ek_err', 'kr',
        #         'e11s', 'e11_errs_s', 'k11_s', 'e22s', 'e22_errs_s', 'k22_s', 'ek_s', 'ek_err_s', 'kr_s',
        #         'lambda_f_iso', 'lambda_g_iso', 're_lambda_iso',
        #         'L', 'u_L', 'tau_L', 'u_lambda_iso', 'tau_lambda_iso',
        #         'eta', 'u_eta', 'tau_eta',
        #         'skewness', 'kurtosis']
        data = [time, e_savg, e_savg_err, enst_savg, enst_savg_err, epsilon_sij,
                e11, e22, e11_err, e22_err, k11, ek, ek_err, kr,
                e11s, e11_errs_s, k11_s, e22s, e22_errs_s, k22_s, ek_s, ek_err_s, kr_s,
                lambda_f_iso, lambda_g_iso, re_lambda_iso,
                # lambda_iso is computed by the isotropic formula using epsilon_sij
                L, u_L, tau_L, u_lambda_iso, tau_lambda_iso,
                eta, u_eta, tau_eta,
                skewness, kurtosis]

        datadict = {}
        for i, key in enumerate(keys):
            try:
                datadict[key] = data[i]
            except:
                pass
        # Write to a h5 file
        write_hdf5_dict(savepath, datadict, overwrite=overwrite)
        # Add udatapath as its attribtue
        with h5py.File(savepath + '.h5', 'a') as ff:
            ff.attrs['udatapth'] = udatapath


def derive_hard(udata, dx, dy, savepath, time=None, inc=1,
                dz=None, nu=1.004,
                x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                reynolds_decomp=False,
                coarse=1.0, coarse2=1.0,  # structure function parameters
                notebook=True, overwrite=True):
    """
    Function to derive quantities which are (relatively) computationally expensive
    ... structure function, two_point correlation function,
     taylor microscale as a curvature of the long.  autocorr. func. etc.

    Parameters
    ----------
    udata: nd array, velocity field
    dx: float, spacing along x
    dy: float, spacing along y
    savepath: str, path where the derived quantities are saved
    time: 1d array, time in physical units
    inc: int, increment along time
        ... if inc=1, then it uses the whole time series to derive various quantities (default)
        ... if inc=n, then it uses every nth time point to derive various quantities (this speeds up the process)
    dz: float, spacing along z (3d data only)
    nu: float, kinematic viscosity
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    t0 int, default: 0
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    t1 int, default: None
        ... index to specify temporal range of data used to compute time average (udata[..., t0:t1:inc])
    reynolds_decomp: bool, default: False
        ... if True, then it uses the fluctuating velocity instead of raw velocity to compute various quantities
    coarse: float, default: 1.0
        ... parameter for get_structure_function
    coarse2: float, default: 1.0
        ... parameter for get_structure_function
        ... The higher the value, the more data points are used to evaluate the statistics to get the structure function
    notebook: bool, default: True
        ... if True, then it uses the tqdm_notebook instead of tqdm to display a progress bar
    overwrite: bool, default: True
        ... if True, then it overwrites the existing data in savepath

    Returns
    -------
    None
    """

    if reynolds_decomp:
        savepath += '_rd'  # mark if user decided to do reynolds decomposition before the whole analysis

    keys = ['rr', 'Dxx', 'Dxx_err', 'rr_scaled', 'Dxx_scaled', 'Dxx_err_scaled',
            'r_long', 'f_long', 'f_err_long', 'r_tran', 'g_tran', 'g_err_tran',
            'lambda_f', 'lambda_g',
            'L11', 'L22',
            're_lambda',
            'epsilon_iso_auto']
    derive = not is_data_derived(savepath + '.h5', keys)  # returns True if data does not exist in the file at savepath

    if derive or overwrite:
        udata = fix_udata_shape(udata)
        dim = len(udata)

        if dim == 2:
            udata = udata[:, y0:y1, x0:x1, t0:t1]
        elif dim == 3:
            if z1 is None:
                z1 = udata[0].shape[2]
            udata = udata[:, y0:y1, x0:x1, z0:z1, t0:t1]
        if time is None:
            time = np.arange(udata.shape[-1])[::inc]

        if reynolds_decomp:
            udata_m, udata_t = reynolds_decomposition(udata)
            vdata = udata_t[..., ::inc]
        else:
            vdata = udata[..., ::inc]
        vdata = fix_udata_shape(vdata)

        xx, yy = get_equally_spaced_grid(vdata, spacing=dx)

        # try:
        # longitudinal structure function
        sfunc_results = get_structure_function(vdata, xx, yy, indices=('x', 'x'), roll_axis=1, nu=nu,
                                               x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=t0, t1=t1,
                                               coarse=coarse, coarse2=coarse2, notebook=notebook)
        rrs, Dijks, Dijk_errs, rrs_scaled, Dijks_scaled, Dijk_errs_scaled = sfunc_results

        # two-point autocorrelation function
        autocorrs = get_two_point_vel_corr_iso(vdata, xx, yy, return_rij=False)
        r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran = autocorrs

        # lengthscales
        lambda_f, lambda_g = get_taylor_microscales(r_long, f_long, r_tran, g_tran)
        L11, L22 = get_integral_scales(r_long, f_long, r_tran, g_tran)

        # get_taylor_re using lambda_f (derived from structure function)
        re_lambda = get_taylor_re(vdata, r_long, f_long, r_tran, g_tran, nu=nu)

        # dissipation rate using the isotropic formula and Taylor microscale from the  autocorrelation function
        # This tends to be really noisy due to bad estimtion of lambda
        epsilon_iso_auto = get_epsilon_iso(vdata, lambda_f=lambda_g, lambda_g=lambda_g)

        # keys = ['rr', 'Dxx', 'Dxx_err', 'rr_scaled', 'Dxx_scaled', 'Dxx_err_scaled',
        #         'r_long', 'f_long', 'f_err_long', 'r_tran', 'g_tran', 'g_err_tran',
        #         'lambda_f', 'lambda_g',
        #         'L11', 'L22',
        #         're_lambda',
        #         'epsilon_iso_auto']
        data = [rrs, Dijks, Dijk_errs, rrs_scaled, Dijks_scaled, Dijk_errs_scaled,
                r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran,
                lambda_f, lambda_g,
                L11, L22,
                re_lambda,
                epsilon_iso_auto]

        datadict = {}
        for i, key in enumerate(keys):
            datadict[key] = data[i]
        write_hdf5_dict(savepath, datadict, overwrite=overwrite)
        # except:
        #     print('... Failed: ', os.path.split(savepath)[1])
        #     pass


# a function to check whether quantities of your interest is derived
def is_data_derived(savepath, datanames, verbose=False, mode=None):
    """
    Check whether quantities of your interest is derived
    Returns True if the quantities exist in the h5 file, False otherwise

    Parameters
    ----------
    savepath: str, path to the h5 file
    datanames: list of str, names of the quantities of interest
    verbose: bool, whether to print out the names of the quantities that are not found
    mode: str, 'hard' or 'easy'
        ... if mode=='easy',
            it checks whether quantities of derive_easy() are in savepath
        ... if mode=='hard',
            it checks whether quantities of derive_hard() are in savepath

    Returns
    -------

    """
    if not os.path.exists(savepath):
        print('... is_data_derived: savepath does not exist! returning False')
        return False
    else:
        if mode == 'hard':
            datanames = ['rr', 'Dxx', 'Dxx_err', 'rr_scaled', 'Dxx_scaled', 'Dxx_err_scaled',
                         'r_long', 'f_long', 'f_err_long', 'r_tran', 'g_tran', 'g_err_tran',
                         'lambda_f', 'lambda_g',
                         'L11', 'L22',
                         're_lambda',
                         'epsilon_iso_auto']
        elif mode == 'easy':
            datanames = ['time', 'e_savg', 'e_savg_err', 'enst_savg', 'enst_savg_err', 'epsilon_sij',
                         'e11', 'e22', 'e11_err', 'e22_err', 'k11', 'ek', 'ek_err', 'kr',
                         'e11s', 'e11_errs_s', 'k11_s', 'e22s', 'e22_errs_s', 'k22_s', 'ek_s', 'ek_err_s', 'kr_s',
                         'lambda_f_iso', 'lambda_g_iso', 're_lambda_iso',
                         'L', 'u_L', 'tau_L', 'u_lambda_iso', 'tau_lambda_iso',
                         'eta', 'u_eta', 'tau_eta',
                         'skewness', 'kurtosis']

        fyle = h5py.File(savepath, 'r')
        result = True
        for dataname in datanames:
            if not dataname in fyle.keys():
                if verbose:
                    print('... cannot find ' + dataname)
                result = False
        fyle.close()
    return result


def run_vel_statistics(dpath, inc=1, overwrite=False, t0=0, t1=None):
    """
    Calculate the statistics of the velocity field stored in dpath, and save the results
    ... A helper for derive_easy()

    Parameters
    ----------
    dpath: str, path to the h5 file where velocity field is stored
    inc: int, increment of the time index to be used to compute the statistics
        ... inc==1: use all v-field data
        ... inc==n: use every n-th frame of the v-field data
    overwrite: bool, whether to overwrite the existing data in dpath
    t0: int, index of the first frame to be used to compute the statistics, [t0, t1)
    t1: int, index of the first frame to be used to compute the statistics, [t0, t1)

    Returns
    -------
    None
    """
    def compute_pdf(data, nbins=100, vmin=None, vmax=None):
        """Get a normalized histogram"""
        data = np.asarray(data)

        # Use data where values are between vmin and vmax
        if vmax is not None:
            cond1 = np.asarray(
                data) < vmax  # if nan exists in data, the condition always gives False for that data point
        else:
            cond1 = np.ones(data.shape, dtype=bool)
        if vmin is not None:
            cond2 = np.asarray(data) > vmin
        else:
            cond2 = np.ones(data.shape, dtype=bool)
        data = data[cond1 * cond2]

        # exclude nans from statistics
        pdf, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=True)
        # len(bins) = len(hist) + 1
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, pdf

    def compute_cdf(data, nbins=100):
        """compute cummulative probability distribution of data"""
        bins, pdf = compute_pdf(data, nbins=nbins)
        cdf = np.cumsum(pdf) * np.diff(bins, prepend=0)
        return bins, cdf

    # If one wants to compute time-avg data between [t0, t1], then save then under /grpname/datanames
    if not (t0 == 0 and t1 is None):
        t1 = get_udata_dim(dpath)[-1]
        grpname = 't0_%05d_t1_%05d' % (t0, t1)
    else:
        grpname = None

    with h5py.File(dpath, mode='a') as f:
        if not (t0 == 0 and t1 is None):
            if not grpname in f.keys():
                grp = f.create_group('/%s/' % grpname)
            keys = [key for key in f[grpname].keys()]
        else:
            keys = [key for key in f.keys()]
    if not all([target in keys for target in ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']]) or overwrite:
        x0, x1, y0, y1, z0, z1 = suggest_udata_dim2load(dpath, show=False, return_tuple=True, return_None=False)
        datadict = {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'z0': z0, 'z1': z1}
        add_data2udatapath(dpath, datadict, overwrite=overwrite, grpname=grpname)
    else:
        x0, x1, y0, y1, z0, z1 = read_data_from_h5(dpath, ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])

    # Velocity statistics
    if not all(
            [target in keys for target in
             ['abs_ui_median', 'abs_ui_avg', 'abs_ui_99', 'abs_ui_99p9', 'u_cutoff']]) or overwrite:
        udata, xx, yy = get_udata_from_path(dpath, inc=100, return_xy=True, t0=t0, t1=t1)  # sample udata
        abs_ui_median, abs_ui_avg = np.nanmedian(np.abs(udata)), np.nanmean(np.abs(udata))
        # bins, pdf = compute_pdf(np.abs(udata))
        bins, cdf = compute_cdf(np.abs(udata))
        u_cutoff = bins[find_nearest(cdf, 0.999)[0]]  # set u_cutoff at which only 0.1% will be rejected
        datadict = {'abs_ui_median': abs_ui_median,
                    'abs_ui_avg': abs_ui_avg,
                    'abs_ui_99': bins[find_nearest(cdf, 0.99)[0]],
                    # 99% of velocity component is less than this value
                    'abs_ui_99p9': bins[find_nearest(cdf, 0.999)[0]],
                    # 99.9% of velocity component is less than this value
                    'u_cutoff': u_cutoff,  # suggested value for u_cutoff for clean
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite, grpname=grpname)
    else:
        u_cutoff = read_data_from_h5(dpath, ['u_cutoff'])


def default_analysis_piv(dpath, inc=1, overwrite=False, time=None, t0=0, t1=None):
    """
    A function to perform default analysis on 2D PIV data.

    Parameters
    ----------
    dpath: str, path to the h5 file where a velocity field is stored
    inc: int, increment of the time index to be used to perform analysis
        ... inc==1: use all v-field data
        ... inc==n: use every n-th frame of the v-field data
    overwrite: bool, whether to overwrite existing analysis results
    time: 1d array, time that corresponds to the measurement
        ... if provided, it writes this data to the h5 file at /t
    t0: int, index of the first frame to be used to compute the statistics, [t0, t1)
    t1: int, index of the first frame to be used to compute the statistics, [t0, t1)
    Returns
    -------

    """

    def compute_pdf(data, nbins=100, vmin=None, vmax=None):
        """Get a normalized histogram"""
        data = np.asarray(data)

        # Use data where values are between vmin and vmax
        if vmax is not None:
            cond1 = np.asarray(
                data) < vmax  # if nan exists in data, the condition always gives False for that data point
        else:
            cond1 = np.ones(data.shape, dtype=bool)
        if vmin is not None:
            cond2 = np.asarray(data) > vmin
        else:
            cond2 = np.ones(data.shape, dtype=bool)
        data = data[cond1 * cond2]

        # exclude nans from statistics
        pdf, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=True)
        # len(bins) = len(hist) + 1
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, pdf

    def compute_cdf(data, nbins=100):
        """compute cummulative probability distribution of data"""
        bins, pdf = compute_pdf(data, nbins=nbins)
        cdf = np.cumsum(pdf) * np.diff(bins, prepend=0)
        return bins, cdf

    # If one wants to compute time-avg data between [t0, t1], then save then under /grpname/datanames
    if not (t0 == 0 and t1 is None):
        t1 = get_udata_dim(dpath)[-1]
        grpname = 't0_%05d_t1_%05d' % (t0, t1)
    else:
        grpname = None

    with h5py.File(dpath, mode='a') as f:
        if not (t0 == 0 and t1 is None):
            if not grpname in f.keys():
                grp = f.create_group('/%s/' % grpname)
            keys = [key for key in f[grpname].keys()]
        else:
            keys = [key for key in f.keys()]
    if not all([target in keys for target in ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']]) or overwrite:
        x0, x1, y0, y1, z0, z1 = suggest_udata_dim2load(dpath, show=False, return_tuple=True, return_None=False)
        datadict = {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'z0': z0, 'z1': z1}
        add_data2udatapath(dpath, datadict, overwrite=overwrite, grpname=grpname)
    else:
        x0, x1, y0, y1, z0, z1 = read_data_from_h5(dpath, ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])

    # Velocity statistics
    if not all(
            [target in keys for target in
             ['abs_ui_median', 'abs_ui_avg', 'abs_ui_99', 'abs_ui_99p9', 'u_cutoff']]) or overwrite:
        udata, xx, yy = get_udata_from_path(dpath, inc=100, return_xy=True, t0=t0, t1=t1)  # sample udata
        abs_ui_median, abs_ui_avg = np.nanmedian(np.abs(udata)), np.nanmean(np.abs(udata))
        bins, pdf = compute_pdf(np.abs(udata))
        bins, cdf = compute_cdf(np.abs(udata))
        u_cutoff = bins[find_nearest(cdf, 0.999)[0]]  # set u_cutoff at which only 0.1% will be rejected
        datadict = {'abs_ui_median': abs_ui_median,
                    'abs_ui_avg': abs_ui_avg,
                    'abs_ui_99': bins[find_nearest(cdf, 0.99)[0]],
                    # 99% of velocity component is less than this value
                    'abs_ui_99p9': bins[find_nearest(cdf, 0.999)[0]],
                    # 99.9% of velocity component is less than this value
                    'u_cutoff': u_cutoff,  # suggested value for u_cutoff for clean
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite, grpname=grpname)
    else:
        u_cutoff = read_data_from_h5(dpath, ['u_cutoff'])

    # Temporal/Spatial average quantities
    if not all([target in keys for target in
                ['etavg', 'esavg', 'esavg_err', 'enst_tavg', 'enst_savg', 'enst_savg_err', 'xc', 'yc', 'zc',
                 'xc_enst', 'yc_enst', 'zc_enst']]) or overwrite:
        udata, xx, yy = get_udata_from_path(dpath, t0=0, t1=1, return_xy=True, verbose=False)  # dummy
        print('... computing time-averaged energy...')
        etavg = get_time_avg_energy_from_udatapath(dpath, inc=inc, t0=t0, t1=t1)
        print('... computing time-averaged enstrophy...')
        enst_tavg = get_time_avg_enstrophy_from_udatapath(dpath, inc=inc, t0=t0, t1=t1)
        results_e = process_large_udata(dpath, func=get_spatial_avg_energy, inc=inc, clean=True,
                                        cutoff=u_cutoff, t0=t0, t1=t1)
        esavg, esavg_err = results_e
        results_enst = process_large_udata(dpath, func=get_spatial_avg_enstrophy, inc=inc,
                                           clean=True, cutoff=u_cutoff, xx=xx, yy=yy, t0=t0, t1=t1)
        enst_savg, enst_savg_err = results_enst

        # center of energy
        xc, yc, zc = np.nansum(xx * etavg) / np.nansum(etavg), \
                     np.nansum(yy * etavg) / np.nansum(etavg), \
                     np.nan
        # center of enstrophy
        xc_enst, yc_enst, zc_enst = np.nansum(xx * enst_tavg) / np.nansum(enst_tavg), \
                                    np.nansum(yy * enst_tavg) / np.nansum(enst_tavg), \
                                    np.nan

        datadict = {'xc': xc, 'yc': yc, 'zc': np.nan,  # Center of energy
                    'xc_enst': xc_enst, 'yc_enst': yc_enst, 'zc_enst': np.nan,  # Center of enstrophy
                    'etavg': etavg,  # time-averaged energy
                    'esavg': esavg,  # spatially averaged energy
                    'esavg_err': esavg_err,  # standard error of spatially averaged energy
                    'enst_tavg': enst_tavg,  # time-averaged enstrophy
                    'enst_savg': enst_savg,  # spatially averaged enstrophy
                    'enst_savg_err': enst_savg_err,  # standard error of spatially averaged enstrophy
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite, grpname=grpname)
    else:
        xc, yc, zc = read_data_from_h5(dpath, ['xc', 'yc', 'zc'])
        etavg, esavg = read_data_from_h5(dpath, ['etavg', 'esavg'])
        enst_tavg, enst_savg = read_data_from_h5(dpath, ['enst_tavg', 'enst_savg'])

    # mean flow
    if not 'udata_m' in keys or overwrite:
        udata_m = get_mean_flow_field_using_udatapath(dpath, inc=inc, t0=t0, t1=t1,
                                                      clean=True, cutoff=np.inf, method='nn', median_filter=False,
                                                      verbose=False,
                                                      notebook=True)
        add_data2udatapath(dpath, {'udata_m': udata_m}, overwrite=overwrite, grpname=grpname)

    # Radial profile
    if not all([target in keys for target in
                ['r_energy', 'eTimeThetaPhi_avg', 'eTimeThetaPhi_avg_err', 'r_blob_e']]) or overwrite:
        udata, xx, yy = get_udata_from_path(dpath, t0=0, t1=1, return_xy=True)  # sample udata
        rr, theta = cart2pol(xx - xc, yy - yc)
        radial_dist, eTimeThetaPhi_avg, eTimeThetaPhi_avg_err = get_binned_stats(rr,
                                                                                 etavg)  # radial, time-averaged energy distritbuion
        radial_dist_enst, enstTimeThetaPhi_avg, enstTimeThetaPhi_avg_err = get_binned_stats(rr,
                                                                                            enst_tavg)  # radial, time-averaged energy distritbuion

        datadict = {'r_energy': radial_dist,  # radial distance for "eTimeThetaPhi_avg"
                    'eTimeThetaPhi_avg': eTimeThetaPhi_avg,
                    # radial energy profile (averaged over polar and azimuthal angles
                    'eTimeThetaPhi_avg_err': eTimeThetaPhi_avg_err,  # standard error of eTimeThetaPhi_avg
                    'r_enstrophy': radial_dist_enst,  # radial distance for "enstTimeThetaPhi_avg"
                    'enstTimeThetaPhi_avg': enstTimeThetaPhi_avg,
                    # radial enstrophy profile (averaged over polar and azimuthal angles
                    'enstTimeThetaPhi_avg_err': enstTimeThetaPhi_avg_err,  # standard error of enstTimeThetaPhi_avg
                    'r_blob_e': np.trapz(radial_dist ** 3 * eTimeThetaPhi_avg, x=radial_dist) / np.trapz(
                        radial_dist ** 2 * eTimeThetaPhi_avg, x=radial_dist),
                    'r_blob_enst': np.trapz(radial_dist_enst ** 3 * enstTimeThetaPhi_avg,
                                            x=radial_dist_enst) / np.trapz(radial_dist ** 2 * enstTimeThetaPhi_avg,
                                            x=radial_dist_enst),
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite, grpname=grpname)
    # else:
    #     radial_dist, eTimeThetaPhi_avg, eTimeThetaPhi_avg_err = read_data_from_h5(dpath,
    #                                                                               ['r_energy', 'eTimeThetaPhi_avg',
    #                                                                                'eTimeThetaPhi_avg_err'])
    if time is not None:
        add_data2udatapath(dpath, {"t": time[t0:t1]}, overwrite=overwrite, grpname=grpname)


# STB helper
def default_analysis_stb(dpath, inc=1, overwrite=False, time=None):
    """
    A function which adds some basic results such as time-averaged energy (FOR 3D DATA)
    ... Suggestive indices of the volume to load:'x0', 'x1', 'y0', 'y1', 'z0', 'z1'
    ... Velocity statistics: 'abs_ui_median', 'abs_ui_avg', 'abs_ui_99', 'abs_ui_99p9', 'u_cutoff'
    ... Temporally/spatially averaged energy and enstrophy: 'etavg', 'esavg', 'esavg_err', 'enst_tavg', 'enst_savg', 'enst_savg_err'
    ... Center of energy: 'xc', 'yc', 'zc'
    ... Center of enstrophy: 'xc_enst', 'yc_enst', 'zc_enst'
    ... Radial energy profile (The center is (xc, yc, zc): r_energy', 'eTimeThetaPhi_avg', 'eTimeThetaPhi_avg_err
    ... Radial enstrophy profile (The center is (xc, yc, zc): r_energy', 'eTimeThetaPhi_avg', 'eTimeThetaPhi_avg_err

    Parameters
    ----------
    dpath: str, path to the h5 file where udata is stored.
        ... the h5 file must include 3D+1 array of 'ux', 'uy', 'uz' and 3D arrays of 'x', 'y', 'z'
        ... this is a default data format
    inc: int, default: 1
    ... increment used to compute time-sensitive quantities such as spatial-averaged energy
        ... If inc==10, for example, this function computes the spatially-averaged energy every 10 time steps.
    overwrite: bool, whether to overwrite existing analysis results
    time: 1d array, time that corresponds to the measurement
        ... if provided, it writes this data to the h5 file at /t

    Returns
    -------
    None
    """

    def compute_pdf(data, nbins=100, vmin=None, vmax=None):
        """Get a normalized histogram"""
        data = np.asarray(data)

        # Use data where values are between vmin and vmax
        if vmax is not None:
            cond1 = np.asarray(
                data) < vmax  # if nan exists in data, the condition always gives False for that data point
        else:
            cond1 = np.ones(data.shape, dtype=bool)
        if vmin is not None:
            cond2 = np.asarray(data) > vmin
        else:
            cond2 = np.ones(data.shape, dtype=bool)
        data = data[cond1 * cond2]

        # exclude nans from statistics
        pdf, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=True)
        # len(bins) = len(hist) + 1
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, pdf

    def compute_cdf(data, nbins=100):
        """compute cummulative probability distribution of data"""
        bins, pdf = compute_pdf(data, nbins=nbins)
        cdf = np.cumsum(pdf) * np.diff(bins, prepend=0)
        return bins, cdf

    with h5py.File(dpath, mode='r') as f:
        keys = [key for key in f.keys()]
    if not all([target in keys for target in ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']]) or overwrite:
        x0, x1, y0, y1, z0, z1 = suggest_udata_dim2load(dpath, show=False, return_tuple=True)
        datadict = {'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'z0': z0, 'z1': z1}
        add_data2udatapath(dpath, datadict, overwrite=overwrite)
    else:
        x0, x1, y0, y1, z0, z1 = read_data_from_h5(dpath, ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])

    # Velocity statistics
    if not all(
            [target in keys for target in
             ['abs_ui_median', 'abs_ui_avg', 'abs_ui_99', 'abs_ui_99p9', 'u_cutoff']]) or overwrite:
        udata, xxx, yyy, zzz = get_udata_from_path(dpath, inc=100, return_xy=True)  # sample udata
        abs_ui_median, abs_ui_avg = np.nanmedian(np.abs(udata)), np.nanmean(np.abs(udata))
        # bins, pdf = compute_pdf(np.abs(udata))
        bins, cdf = compute_cdf(np.abs(udata))
        u_cutoff = bins[find_nearest(cdf, 0.999)[0]]  # set u_cutoff at which only 0.1% will be rejected
        datadict = {'abs_ui_median': abs_ui_median,
                    'abs_ui_avg': abs_ui_avg,
                    'abs_ui_99': bins[find_nearest(cdf, 0.99)[0]],
                    # 99% of velocity component is less than this value
                    'abs_ui_99p9': bins[find_nearest(cdf, 0.999)[0]],
                    # 99.9% of velocity component is less than this value
                    'u_cutoff': u_cutoff,  # suggested value for u_cutoff for clean
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite)
    else:
        u_cutoff = read_data_from_h5(dpath, ['u_cutoff'])

    # Temporally averaged quantities
    if not all([target in keys for target in
                ['etavg', 'enst_tavg', 'xc', 'yc', 'zc', 'xc_enst', 'yc_enst', 'zc_enst']]) or overwrite:
        udata, xxx, yyy, zzz = get_udata_from_path(dpath, t0=0, t1=1, return_xy=True)  # sample udata
        etavg = get_time_avg_energy_from_udatapath(dpath, inc=inc)
        enst_tavg = get_time_avg_enstrophy_from_udatapath(dpath, inc=inc)

        # center of energy
        xc, yc, zc = np.nansum(xxx * etavg) / np.nansum(etavg), np.nansum(yyy * etavg) / np.nansum(
            etavg), np.nansum(zzz * etavg) / np.nansum(etavg)
        xc_enst, yc_enst, zc_enst = np.nansum(xxx * enst_tavg) / np.nansum(enst_tavg), np.nansum(
            yyy * enst_tavg) / np.nansum(enst_tavg), np.nansum(zzz * enst_tavg) / np.nansum(enst_tavg)

        datadict = {'xc': xc, 'yc': yc, 'zc': zc,  # Center of energy
                    'xc_enst': xc_enst, 'yc_enst': yc_enst, 'zc_enst': zc_enst,  # Center of enstrophy
                    'etavg': etavg,  # time-averaged energy
                    'enst_tavg': enst_tavg,  # time-averaged enstrophy
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite)
    else:
        xc, yc, zc = read_data_from_h5(dpath, ['xc', 'yc', 'zc'])
        etavg, enst_tavg = read_data_from_h5(dpath, ['etavg', 'enst_tavg'])

    # Spatially averaged quantities
    if not all([target in keys for target in
                ['esavg', 'esavg_err', 'enst_savg', 'enst_savg_err', ]]) or overwrite:
        results_e = process_large_udata(dpath, func=get_spatial_avg_energy, inc=inc, clean=True,
                                        cutoff=u_cutoff)
        esavg, esavg_err = results_e
        results_enst = process_large_udata(dpath, func=get_spatial_avg_enstrophy, inc=inc,
                                           clean=True, cutoff=u_cutoff, xx=xxx, yy=yyy, zz=zzz)
        enst_savg, enst_savg_err = results_enst

        datadict = {'esavg': esavg,  # spatially averaged energy
                    'esavg_err': esavg_err,  # standard error of spatially averaged energy
                    'enst_savg': enst_savg,  # spatially averaged enstrophy
                    'enst_savg_err': enst_savg_err,  # standard error of spatially averaged enstrophy
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite)
    else:
        xc, yc, zc = read_data_from_h5(dpath, ['xc', 'yc', 'zc'])
        esavg, enst_tavg = read_data_from_h5(dpath, ['esavg', 'enst_savg'])

    # Radial profile
    if not all([target in keys for target in ['r_energy', 'eTimeThetaPhi_avg', 'eTimeThetaPhi_avg_err']]) or overwrite:
        udata, xxx, yyy, zzz = get_udata_from_path(dpath, t0=0, t1=1, return_xy=True)  # sample udata
        rr, theta, phi = cart2sph(xxx - xc, yyy - yc, zzz - zc)
        radial_dist, eTimeThetaPhi_avg, eTimeThetaPhi_avg_err = get_binned_stats(rr,
                                                                                 etavg)  # radial, time-averaged energy distritbuion
        radial_dist_enst, enstTimeThetaPhi_avg, enstTimeThetaPhi_avg_err = get_binned_stats(rr,
                                                                                            enst_tavg)  # radial, time-averaged energy distritbuion

        datadict = {'r_energy': radial_dist,  # radial distance for "eTimeThetaPhi_avg"
                    'eTimeThetaPhi_avg': eTimeThetaPhi_avg,
                    # radial energy profile (averaged over polar and azimuthal angles
                    'eTimeThetaPhi_avg_err': eTimeThetaPhi_avg_err,  # standard error of eTimeThetaPhi_avg
                    'r_enstrophy': radial_dist_enst,  # radial distance for "enstTimeThetaPhi_avg"
                    'enstTimeThetaPhi_avg': enstTimeThetaPhi_avg,
                    # radial enstrophy profile (averaged over polar and azimuthal angles
                    'enstTimeThetaPhi_avg_err': enstTimeThetaPhi_avg_err,  # standard error of enstTimeThetaPhi_avg
                    }
        add_data2udatapath(dpath, datadict, overwrite=overwrite)
    # else:
    #     radial_dist, eTimeThetaPhi_avg, eTimeThetaPhi_avg_err = read_data_from_h5(dpath,
    #                                                                               ['r_energy', 'eTimeThetaPhi_avg',
    #                                                                                'eTimeThetaPhi_avg_err'])
    if time is not None:
        add_data2udatapath(dpath, {"t": time}, overwrite=overwrite)


# functions related to turbulence decay
def get_time_indices_for_selfsimilar_movie(time, t0, dt, exponent=-1, nmax=None):
    """
    Returns indices of a time array required to make a self-similar movie

    ... Let t_i be the time at frame i.
    To preserve the displacement between two frames when energy decays in a power law (E ~ (t-t0)^n),
    t_i must satisfy the following recurrence relation.

    t_{i+2} = {2t_{i+1}^m - 2t_{i}^m }^{1/m}

    where m = n/2 + 1.

    ... This function returns indices for the given array "time". To do so, one needs t0, t1, and n.
    ... t1 = t0 + dt
    ... exponent is "n".

    Parameters
    ----------
    time: 1d array
        ... time. this does not have to be evenly spaced.
        ... e.g. [0.0, 0.005, 0.010, 0.020, ...]
    t0: float/int
        ... the first
    dt: float/int
        ... t1 = t0 + dt. If not clear, read about the recurrence relation above
    exponent: exponent of energy decay, default=-1
        ... Some best estimate on this is around -1.2 for the initial period of decay.
    nmax: int, default=None
        ... number of elements of the output array

    Returns
    -------
    t_indices: 1d array, indices of a time array required to make a self-similar movie
    """

    def find_nearest(array, value, option='normal'):
        """
        Find an element and its index closest to 'value' in 'array'
        Parameters
        ----------
        array
        value

        Returns
        -------
        idx: index of the array where the closest value to 'value' is stored in 'array'
        array[idx]: value closest to 'value' in 'array'

        """
        # get the nearest value such that the element in the array is LESS than the specified 'value'
        if option == 'less':
            array_new = copy.copy(array)
            array_new[array_new > value] = np.nan
            idx = np.nanargmin(np.abs(array_new - value))
            return idx, array_new[idx]
        # get the nearest value such that the element in the array is GREATER than the specified 'value'
        if option == 'greater':
            array_new = copy.copy(array)
            array_new[array_new < value] = np.nan
            idx = np.nanargmin(np.abs(array_new - value))
            return idx, array_new[idx]
        else:
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

    t1 = t0 + dt
    tau0 = t0 - t0
    tau1 = t1 - t0
    taus = [tau0, tau1]

    n = exponent
    m = n / 2. + 1

    counter = 2
    t_indices = []
    if nmax is None:
        nmax = len(time)
    while counter < nmax:
        tau0, tau1 = taus[-2], taus[-1]
        tau2 = (2 * tau1 ** m - tau0 ** m) ** (1. / m)
        if tau2 < np.nanmax(time):
            taus.append(tau2)
        else:
            break
        counter += 1

    for i, tau in enumerate(taus):
        t_ind, _ = find_nearest(time, t0 + tau)
        t_indices.append(t_ind)
    return np.asarray(t_indices)


def get_suggested_inds(udatapath):
    """
    Returns a dictionary of the recommended indices of the udata stored in udatapath
    ... e.g. 
        inds = read_suggested_inds(udatapath)
        udata = read_udata_from_path(udatapath, **inds)
        ... This will load udata[y0:y1, x0:x1, z0:z1, :] onto the memory
        ... The motivation behind this is that STB sometimes results a lot of empty data especially when no particles can be tracked in the region.
            ... This is often the case due to the limited depth of the field that the STB may resolve. 
            ... suggest_udata_dim2load(udatapath) computes the statistics, and suggests the indices one should use so that one avoids loading 
            the empty data which could bias the data and cause problems for many analyses such as a Fourier analysis. 
        
    Parameters
    ----------
    udatapath: str, a path to the h5 file where udata is stored

    Returns
    -------
    inds: dict, the suggested indices of the udata
        ... inds = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}
        ... udata[y0:y1, x0:x1, z0:z1, :]

    """
    x0, x1, y0, y1, z0, z1 = read_data_from_h5(udatapath, ['x0', 'x1', 'y0', 'y1', 'z0', 'z1'])
    if x0 is None:
        print("read_suggested_inds: Cannot find the datasets (x0, x1, y0, y1,z0, z1)")
        inds = {"x0": 0, "x1": None, "y0": 0, "y1": None, "z0": 0, "z1": None}
    else:
        inds = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}
    return inds


############ THEORY/MODELS: ############
# Inviscid, incompressible flow
# stream function formulaton
## APPROACH1: Compute the contour integral between point A (reference) and B (observing pt)
# 2D flows
def compute_streamfunction_values(udata, xx, yy,
                                  x, y,
                                  xref=0, yref=0,
                                  contours=None, nctrs=4, nsample=100,
                                  nkinks=5, noise=None,
                                  return_contours=False):
    """
    Returns a streamfunction value at a point (x, y) for 2D incompressible flows
    with respect to the value at the reference (xref, yref)

    ... Computes a contour integral \int_A^B udy - vdx
    ... Requires bezier module

    Parameters
    ----------
    udata
    xx
    yy
    x
    y
    xref, float
    ... x coordinate of the reference point
    yref, float
    ... y coordinate of the reference point
    contours: 3d array with shape (number of contours, number of points on each contour, 2), default: None
        E.g. (x, y) on the 3rd contour is stored as (contours[2, :, 0], contours[2, :, 1])
        ... By default, it generates nctrs contours connecting (xref, yref) and (x, y)
            It generates non-selfintersecting, smooth conours. It could be non-smooth if (xref, yref) == (x, y)
    nctrs: int, must be greater than or equal to 1
        ... number of contours used to compute the streamfunction value at the given point
        ... the streamfunction values could be different depending on the contour due to discreteness of data
            ... divergent field will also result in such a discrepancy
            udata must be non-divergent in order for the streamfunction to exist
    nsample: int, default: 100
        ... number of sampled points on each contour to compute the integral,
    nkinks: int, must be greater or equal to 2
        ... this controls how wiggly contours are if the contours were not given by the user
    noise: array-like with two elements, default: None
        ... This controls the maximum amplitude of the wiggle.
    return_contours
        ... If True, it returns contours used to compute the integral.
    Returns
    -------
    psis: nd array with shape=(nctrs, udata.shape[-1)
    contours (optional): points on the contours used to compute the streamfunction value

    """
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]

    # Initialize
    psis = np.empty((nctrs, duration))
    psis[:] = np.nan

    # Warning
    if dim == 3:
        raise ValueError(
            '... Spatially 3D udata is provided. \n'
            'Potential formation of 3D flows is possible but is out of the scope of this function')

    if contours is None:
        contours = np.empty((nctrs, nsample, 2))
        for i in range(nctrs):
            contours[i, ...] = gen_arb_curves([xref, yref], [x, y], n=nkinks, npts=nsample, noise=noise)

    for i, contour in enumerate(contours):
        xs, ys = contours[i, :, 0], contours[i, :, 1]
        dx = np.diff(xs)
        dy = np.diff(ys)
        dx, dy = np.append(dx, 0), np.append(dy, 0)
        dl = np.stack((dx, dy)).T

        contour4int_func = contour[:, [1, 0]]  # y, x
        for t in range(duration):
            fs = interpolate_udata_at_instant_of_time(udata, xx, yy, t=t)
            ux_along_cntr = fs[0](contour4int_func)
            uy_along_cntr = fs[1](contour4int_func)
            psi = ux_along_cntr * dl[:, 1] - uy_along_cntr * dl[:, 0]
            psis[i, t] = np.nanmean(psi) * len(ux_along_cntr)

    if return_contours:
        contours = np.asarray(contours)
        return psis, contours
    else:
        return psis


def compute_streamfunction_direct(udata, xx, yy,
                                  xref=0, yref=0,
                                  inc=1,
                                  nctrs=1, nsample=100,
                                  nkinks=10, noise=None,
                                  return_sampled_grid=False,
                                  coordinate_system='cartesian',
                                  notebook=True):
    """
    Returns a streamfunction of a given udata in a ND array by computing
            \psi = \int_A^B udy - vdx  (cartesian)
            \psi = \int_A^B \rho uz drho - urho dz  (cyindrical)

    ... udata must be non-divergent

    Parameters
    ----------
    udata
    xx: 2d array, a positional grid (x)
        ... If coordinate_system == 'cylindrical', this will be interpreted as a grid of z
    yy: 2d array, a positional grid (y)
        ... If coordinate_system == 'cylindrical', this will be interpreted as a grid of rho=np.sqrt(x**2 + y**2)
    xref: float, default: 0
        ... reference point of a streamfunction
        ... The contour integral will be calculated between (xref, yref) and (xp, yp)
            (xp, yp) will be sampled from the given positinal grid.
    yref: float, default: 0
        ... reference point of a streamfunction
        ... The contour integral will be calculated between (xref, yref) and (xp, yp)
            (xp, yp) will be sampled from the given positinal grid.
    inc: int, sampling frequency of (xp, yp) out of the positional grid (xx and yy)
    nctrs: int, must be greater than or equal to 1
        ... number of contours used to compute the streamfunction value at the given point
        ... the streamfunction values could be different depending on the contour due to discreteness of data
            ... divergent field will also result in such a discrepancy
            udata must be non-divergent in order for the streamfunction to exist
    nsample: int, default: 100
        ... number of sampled points on each contour to compute the integral,
    nkinks: int, must be greater or equal to 2
        ... this controls how wiggly contours are if the contours were not given by the user
    noise: array-like with two elements, default: None
        ... This controls the maximum amplitude of the wiggle.
    return_sampled_grid: bool, default: False
        ... If True, this will return positional grids (xx_s, yy_s) used to sample the streamfunction values.
            This becomes handy if inc is not equal to 1. You can always plot the result as ax.pcolormesh(xx_s, yy_s, psi)
    coordinate_system: str, default: 'cartesian'
        ... the form of the contour integral to compute the stream function depends on the coordinate system
            psi = \int u dy - v dx (Cartesian, 2D)
            psi = \int \rho (u_z drho - u_rho dz) (Cylindrical, 3D-axisymmetric)
    notebook: bool, default: True
        ... If True, it uses tqdm.tqdm_notebook for a progress bar.
            (a natural choise if you run the code on Jupyter notebooks)
        ... If False, it uses tqdm.tqdm for a progress bar.

    Returns
    -------
    psi_master, psi_err_master, xx_sampled (Optional), yy_sampled (Optional)
    """
    if coordinate_system == 'cylindrical':
        print('compute_streamfunction_direct():')
        print('... coordinate_system is cylindrical')
        print('... udata=(uz, urho), (xx, yy) will be interpreted as (zz, rrho)')
        print('... (xp, yp, xref, yref) will be interpreted as (zp, rhop, zref, rhoref)')
        print('... yy = rrho  must be in an ascending order. (1) you must be passing rho = sqrt(xx **2 + yy**2)\n'
              'therefore, rho must be greater or equal to 0 !!!')

    if notebook:
        from tqdm import tqdm_notebook as tqdm
    udata = fix_udata_shape(udata)
    dim, height, width, duration = udata.shape
    x1d, y1d = xx[0, ::inc], yy[::inc, 0]
    xx_sampled, yy_sampled = np.meshgrid(x1d, y1d)

    psi_shape = (len(y1d), len(x1d), duration)

    psi_master = np.empty(psi_shape)
    psi_err_master = np.empty(psi_shape)

    for t in range(duration):
        for i, yp in enumerate(tqdm(y1d)):
            for j, xp in enumerate(x1d):
                if coordinate_system == 'cartesian':
                    psis, contours = compute_streamfunction_values(udata, xx, yy,
                                                                   xp, yp, xref, yref,
                                                                   nkinks=nkinks, noise=noise,
                                                                   nctrs=nctrs, nsample=nsample,
                                                                   return_contours=True)
                elif coordinate_system == 'cylindrical':
                    psis, contours = compute_streamfunction_values_cylindrical(udata, xx, yy,
                                                                               xp, yp, xref, yref,
                                                                               nkinks=nkinks, noise=noise,
                                                                               nctrs=nctrs, nsample=nsample,
                                                                               return_contours=True)
                psi_master[i, j, t] = np.nanmean(psis)
                psi_err_master[i, j, t] = np.nanstd(psis)

    if notebook:
        from tqdm import tqdm as tqdm
    if return_sampled_grid:
        return psi_master, psi_err_master, xx_sampled, yy_sampled
    else:
        return psi_master, psi_err_master


# 3D axisymmetric_flows
def compute_streamfunction_values_cylindrical(udata_cylindrical, zz, rrho,
                                              z, rho,
                                              zref=0, rhoref=0,
                                              contours=None, nctrs=4, nsample=100,
                                              nkinks=10, noise=None,
                                              return_contours=False):
    """
    Returns a streamfunction value at a point (x, y) for 3D, axisymmetric flows
    with respect to the value at the reference (xref, yref)

    ... Computes a contour integral \int_A^B rho (u_z drho - u_rho dz)
    ... Requires bezier module

    Parameters
    ----------
    udata_cylindrical: nd array, (u_z, u_rho)
    zz
    rrho
    z
    rho
    zref, float
    rhoref, float
    contours: 3d array with shape (number of contours, number of points on each contour, 2), default: None
        E.g. (x, y) on the 3rd contour is stored as (contours[2, :, 0], contours[2, :, 1])
        ... By default, it generates nctrs contours connecting (xref, yref) and (x, y)
            It generates non-selfintersecting, smooth conours. It could be non-smooth if (xref, yref) == (x, y)
    nctrs: int, must be greater than or equal to 1
        ... number of contours used to compute the streamfunction value at the given point
        ... the streamfunction values could be different depending on the contour due to discreteness of data
            ... divergent field will also result in such a discrepancy
            udata must be non-divergent in order for the streamfunction to exist
    nsample: int, default: 100
        ... number of sampled points on each contour to compute the integral,
    nkinks: int, must be greater or equal to 2
        ... this controls how wiggly contours are if the contours were not given by the user
    noise: array-like with two elements, default: None
        ... This controls the maximum amplitude of the wiggle.
    return_contours
        ... If True, it returns contours used to compute the integral.

    Returns
    -------
    psis: nd array with shape=(nctrs, udata.shape[-1)
    contours (optional): points on the contours used to compute the streamfunction value

    """
    udata = fix_udata_shape(udata_cylindrical)
    dim, duration = udata.shape[0], udata.shape[-1]

    # Initialize
    psis = np.empty((nctrs, duration))
    psis[:] = np.nan

    # Warning
    if dim == 3:
        raise ValueError(
            '... Spatially 3D udata is provided. \n'
            'Potential formation of 3D flows is possible but is out of the scope of this function')

    if contours is None:
        contours = np.empty((nctrs, nsample, 2))
        for i in range(nctrs):
            contours[i, ...] = gen_arb_curves([zref, rhoref], [z, rho], n=nkinks, npts=nsample, noise=noise)

    for i, contour in enumerate(contours):
        zs, rhos = contours[i, :, 0], contours[i, :, 1]
        dz = np.diff(zs)
        drho = np.diff(rhos)
        dz, drho = np.append(dz, 0), np.append(drho, 0)  # the destination of the contour will not be counted to the sum
        dl = np.stack((dz, drho)).T

        contour4int_func = contour[:, [1, 0]]  # y, x
        for t in range(duration):
            fs = interpolate_udata_at_instant_of_time(udata_cylindrical, zz, rrho, t=t)
            uz_along_cntr = fs[0](contour4int_func)
            urho_along_cntr = fs[1](contour4int_func)
            psi = rhos * (uz_along_cntr * dl[:, 1] - urho_along_cntr * dl[:, 0])
            psis[i, t] = np.nanmean(psi) * len(uz_along_cntr)

    if return_contours:
        contours = np.asarray(contours)  # in (z, rho)
        return psis, contours
    else:
        return psis


def psi2udata(psi,
              dx=1, dy=1,
              xy_orientations=np.asarray([1, 1]),
              xx=None, yy=None,
              coordinate_system='cartesian'):
    """
    Derives udata from streamfunction psi.
    returns v = dq/dx and u= -dq/dy, on U-grid

    Source: https://anaconda.org/bfiedler/n090_streamfunctionvorticity2d/notebook?version=2018.07.28.0933

    Parameters
    ----------
    psi: 2d array, streamfunction values
    dx: spacing in x-direction
    dy: spacing in y-direction
    xy_orientations: 2d array, (x-orientation, y-orientation)
        ... x-orientation: 1 if x increases from left to right, -1 if x decreases from left to right
        ... y-orientation: 1 if y increases from bottom to top, -1 if y decreases from bottom to top
    xx: 2d array, x-coordinates of psi
    yy: 2d array, y-coordinates of psi
    coordinate_system: string, default: 'cartesian', must be 'cartesian' or 'cylindrical'

    Returns
    -------
    udata: 2d array, velocity field (Cartesian)
    """
    # u = 0. * psi
    # v = 0. * psi
    #
    # u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dy)
    # u[0, 1:-1] = psi[1, 1:-1] / dy
    # u[-1, 1:-1] = -psi[-2, 1:-1] / dy
    #
    # v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dx)
    # v[1:-1, 0] = -psi[1:-1, 1] / dx
    # v[1:-1, -1] = +psi[1:-1, -2] / dx

    if coordinate_system == 'cylindrical':
        print('... psi=psi(z, rho)')
        print('... Note that Caucy-Riemann relation depends on the cooridnate system')
        print('... try psi2udata_cylindrical()')
    else:
        if xx is not None and yy is not None:
            xy_orientations = get_jacobian_xyz_ijk(xx, yy)
            dx, dy = get_grid_spacing(xx, yy)

        u = np.gradient(psi, dy, axis=0) * xy_orientations[0]
        v = -np.gradient(psi, dx, axis=1) * xy_orientations[1]

        udata = np.stack((u, v))

        return udata


def psi2udata_cylindrical(psi,
                          rrho=None, zz=None,
                          return_cartesian=False):
    """
    Derives udata from streamfunction psi in cylindrical basis.

    Source: https://anaconda.org/bfiedler/n090_streamfunctionvorticity2d/notebook?version=2018.07.28.0933

    Parameters
    ----------
    psi: 2d array, streamfunction values
    rrho: 2d array, rho-coordinates of psi
    zz: 2d array, z-coordinates of psi
    return_cartesian: bool, default: False, if True, returns udata in cartesian coordinates

    Returns
    -------
    udata: 2d array, velocity field (cylindrical)
        ... if return_cartesian=True: udata = (ux, uy)
        ... if return_cartesian=False: udata = (uz, urho)

    """

    urho = - 1. / rrho * np.gradient(psi, zz[0, :], axis=1)
    uz = 1. / rrho * np.gradient(psi, rrho[:, 0], axis=0)
    udata_cylindrical = np.stack((uz, urho))

    # udata_cylindrical = (uz, urho)
    ## now, we redefine the z-axis as the x-axis, and the rho axis as y as we initially did.
    ux = uz
    uy = copy.deepcopy(urho)
    drho = np.gradient(rrho[:, 0])  # If drho < 0, y<0. i.e. If drho < 0, dy = -drho
    # uy[drho > 0] *= -1
    udata_cartesian = np.stack((ux, uy))

    if return_cartesian:
        return udata_cartesian
    else:
        return udata_cylindrical


# Sample stream functions and udata
def get_streamfunction_about_a_rankine_vortex(xx, yy, x0=0, y0=0, gamma=1., a=1.):
    """
    Returns a streamfunction about a rankine vortex.
    Source: http://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/fprops/poten/node38.html

    Parameters
    ----------
    xx: 2d array, x-coordinates
    yy: 2d array, y-coordinates
    x0: float, default: 0, x-coordinate of vortex
    y0: float, default: 0, y-coordinate of vortex
    gamma: float, default: 1, circulation of vortex
    a: float, default: 1, radius of vortex

    Returns
    -------
    psi: 2d array, streamfunction values
    """
    rr, theta = cart2pol(xx - x0, yy - y0)
    psi = np.empty_like(rr)

    inside_core = rr <= a
    psi[inside_core] = - gamma / (4 * np.pi * a ** 2) * rr[inside_core] ** 2
    psi[~inside_core] = - gamma * a ** 2 / (4 * np.pi * a ** 2) * (1 + 2 * np.log(rr[~inside_core] / a))
    return psi


def get_streamfunction_about_a_lifting_cylinder(xx, yy, xc=0, yc=0, u=1, a=1, beta=-np.pi / 4):
    """
    Returns a stream function about a lifting cylinder

    Source: http://www-mdp.eng.cam.ac.uk/web/library/enginfo/aerothermal_dvd_only/aero/fprops/poten/node38.html

    Parameters
    ----------
    xx: 2d array, x-coordinates
    yy: 2d array, y-coordinates
    xc: float, default: 0, x-coordinate of cylinder
    yc: float, default: 0, y-coordinate of cylinder
    u: float, default: 1, magnitude of the stream velocity
    a: radius of cylinder
    beta: angle of the stream

    Returns
    -------
    psi: 2d array, streamfunction values
    """
    rr, theta = cart2pol(xx - xc, yy - yc)
    gamma = 4 * np.pi * u * a * np.sin(beta)
    psi = u * rr * (1 - a ** 2 / rr ** 2) * np.sin(theta) - gamma / 2 / np.pi * np.log(rr)
    return psi


def get_flow_about_a_lifting_cylinder(xx, yy, xc=0, yc=0, u=1, a=1, beta=-np.pi / 4, return_in_xy_basis=True):
    """
    Returns a flow (udata) about a lifting cylinder

    Parameters
    ----------
    xx: 2d array, x-coordinates
    yy: 2d array, y-coordinates
    xc: float, default: 0, x-coordinate of cylinder
    yc: float, default: 0, y-coordinate of cylinder
    u: float, default: 1, magnitude of the stream velocity
    a: radius of cylinder
    beta: angle of the stream
    return_in_xy_basis: bool, default: True, if True, returns udata in cartesian coordinates
        ... if return_in_xy_basis=False, returns udata in polar coordinates

    Returns
    -------
    psi: 2d array, streamfunction values
    """
    psi = get_streamfunction_about_a_lifting_cylinder(xx, yy, xc=xc, yc=yc, u=u, a=a, beta=beta)

    dx, dy = get_grid_spacing(xx, yy)
    udata = psi2udata(psi, dx, dy)
    udata_pol = cart2pol_udata(xx, yy, udata)

    if return_in_xy_basis:
        return udata
    else:
        return udata_pol


def get_flow_past_a_circular_cylinder(xx, yy, xc=0, yc=0, u=1., a=1., return_in_xy_basis=True):
    """
    Returns udata (ux, uy) of a flow past a circular cylinder in Stokes limit

    Parameters
    ----------
    xx: 2d array, x-coordinates
    yy: 2d array, y-coordinates
    xc: float, default: 0, x-coordinate of cylinder
    yc: float, default: 0, y-coordinate of cylinder
    U: float, default: 1, magnitude of the stream velocity
    a: radius of cylinder
    return_in_xy_basis: bool, default: True, if True, returns udata in cartesian coordinates
        ... if return_in_xy_basis=False, returns udata in polar coordinates

    Returns
    -------
    psi: 2d array, streamfunction values
    """
    k = u * a ** 2

    rr, ttheta = cart2pol(xx - xc, yy - yc)

    ur = np.cos(ttheta) * (u - k / rr ** 2)
    utheta = -np.sin(ttheta) * (u + k / rr ** 2)
    udata_pol = np.stack((ur, utheta))
    xx, yy, udata = pol2cart_udata(rr, ttheta, udata_pol, x0=xc, y0=yc)

    if return_in_xy_basis:
        return udata
    else:
        return udata_pol


def gen_bezier_curve(pts, n=100):
    """
    Generates a Bezier curve between two given points
    ... pts.shape = (n, 2)
    ... pts[0] = x, pts[1] = y

    REQUIRES berzier module

    Parameters
    ----------
    pts: 2d array,
        ... points used to draw bezier curves. Shape: (number of points, 2)
    n: int
        ... number of sampled points on the obtained bezier curve

    Returns
    -------
    bezier_curve: 2d array, shape=(n, 2)
        ... (x, y) = (bezier_curve[:, 0], bezier_curve[:, 1])

    """
    nodes = np.asarray(pts).T
    dim, npts = nodes.shape

    curve1 = bezier.Curve(nodes, degree=npts - 1)

    s = np.linspace(0, 1, n)
    bezier_curve = curve1.evaluate_multi(s).T
    return bezier_curve


def gen_arb_curves(pt1, pt2, n=5, npts=100, noise=None, allow_self_intersection=False,
                   n_attempt=100):
    """
    Generates n arbitrary curves connecting pt1 and pt2
    ... the obtained curve is not self-intersecting by default
        ... this can be changed via "allow_self_intersection"

    How it works:
    1. Sample equally-spaced points on the straight line connecting pt1 and pt2
    2. Add noise to each sampled points
    3. Draw a Benzier curve with disturbed points (this makes the curve smooth)
    4. If Step 3 results a self-intersecting curve, repeat 1-3 untile the curve is not self-intersecting
    5. Repeat 1-4 until you obtain n curves


    Parameters
    ----------
    pt1: 1d array-like, e.g.- [0., 0.,]
    pt2: 1d array-like, e.g.- [1., 1.,]
        ... pt1 and pt2 could be the same coordinate but one might need to adjust "noise" to obtain a desired curve
            However, this would make the curve non-smooth at pt1
    n: int, MUST be greater than or equal to 2
        ... number of points used to construct a Benzier curve
        ... The higher n is, the more wiggly/complex a curve becomes
        ... n=2: Line
            n=3: 1 kink
            n=4: 2 kinks ...
    npts: int
        ... number of sampled points on the output curves
            This essentially defines the number of kinks on the curve
    noise:
        ... noise is added to a straight lig
    allow_self_intersection: bool, default: False
        ... If False, do not allow self-intersection of the generated curves
    n_attempt: int, default: 100

    Returns
    -------
    bezier_curve: 2d array, shape=(n, 2)
    """

    def check_if_curve_self_intersects(pts, inc=1, verbose=False):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        pts = np.asarray(pts)
        n, dim = pts.shape

        self_intersects = False
        for i in range(0, n - inc, inc):
            if self_intersects:
                break
            A = pts[i, :]
            B = pts[i + inc, :]
            for j in range(0, n - inc, inc):
                if not j in list(range(i - inc, i + inc + 1)):
                    C = pts[j, :]
                    D = pts[j + inc, :]

                    self_intersects = intersect(A, B, C, D)
                    if self_intersects:
                        if verbose:
                            print(i, j, list(range(i - inc, i + inc)), self_intersects)
                        break
        return self_intersects

    if n < 2:
        raise ValueError('gen_arb_curves(): npts must be greater than or equal to 2!')

    pt1, pt2 = np.asarray(pt1), np.asarray(pt2)

    if noise is None:
        # noise = np.abs((pt2 - pt1) / n - 1)
        amp = np.sqrt(np.nansum((pt2 - pt1) ** 2)) / 2
        noise = [amp, amp]

    self_intersecting = True
    counter = 0
    while self_intersecting and counter < n_attempt:
        nodes = np.empty((n, 2))
        nodes[:, 0] = np.linspace(pt1[0], pt2[0], n)
        nodes[:, 1] = np.linspace(pt1[1], pt2[1], n)
        nodes[1:-1, 0] += (np.random.random(n - 2) - 0.5) * noise[0]
        nodes[1:-1, 1] += (np.random.random(n - 2) - 0.5) * noise[1]

        bezier_curve = gen_bezier_curve(nodes, n=npts)

        counter += 1

        if not allow_self_intersection:
            # Check if the obtained curve is self-intersecting
            self_intersecting = check_if_curve_self_intersects(bezier_curve)
        else:
            # If you don't care whether the obtained curve is self-intersecting, leave the while-loop
            break
    if counter == n_attempt:
        print(
            'gen_arb_curves(): Non-self-intersecting curve was not obtained with the given parameters after %d trials' % n_attempt)
        print('... Consider decreasing noise')
    return bezier_curve


def get_streamfunction_hill_spherical_vortex(xx, yy, x0=0, y0=0, u=1, a=1, gamma=None,
                                             reference_frame='lab',
                                             return_cylindrical_coords=False):
    """
    Returns ax streamfunction values for a Hill's spherical vortex in a lab frame

    Source: https://link.springer.com/chapter/10.1007/978-3-319-55164-7_34 (by Emmanuel Branlard, p.413)

    Hill's spherical vortex is an axisymmetric flow.
    ... Radial distance to the extremum of streamfunction inside the bubble is a/sqrt(2)=0.707a.
    ... In the 'lab' frame, psi=0 defines the vortex atmosphere.
    ...... The streamfunction has an extremeum at rho=a/sqrt(2)=0.707a.
    ...... The vorticity center (rho weighted by omega(rho, z, theta)) is at rho=0.75a.
    ...... The velocity at the center is -1.5u \hat{z}.
    ... In the 'vortex' frame, the vortex is at rest.
    ...... The velocity at the center is 0.
    ... In the 'wind' or 'absolute' frame, the environment is at rest. velocity outside the vortex decays to zero at infinity.
    ...... The velocity at the center is -2.5u \hat{z}

    Parameters
    ----------
    xx: 2d array, positional grid (x)
    yy: 2d array, positional grid (y)
    x0: float, default: 0
        ... center of the spherical vortex is at (x0, y0)
    y0: float, default: 0
        ... center of the spherical vortex is at (x0, y0)
    u: float
        ... stream velocity
        ... The center of the vortex moves at (-1.5u, 0)
        ... gamma = 5 * a * u # circulaton of gamma
    gamma: float, default: None
        ... if given, u will be overwritten
    a: float, >0
        ... radius of the spherical vortex
    reference_frame: str, default- 'lab'
        ... Choose reference_frame from [lab, wind, vortex]
        ... "lab" refers to the rest frame
        ... "vortex" refers to the frame of reference of the vortex
        ... "wind"/"absolute" refers to a frame of reference such that the -2.5u.
    return_cylindrical_coords: bool, default- False

    Returns
    -------
    psi: 2d array, streamfunction

    Examples
    --------
        import velocity as vel
        import graph
        n = 101
        L = 200
        a=50
        x, y = np.linspace(-L/2, L/2, n), np.linspace(0, L/2, n)
        XX, YY = np.meshgrid(x, y)
        dx, dy = vel.get_grid_spacing(XX, YY)

        psi, zz, rrho = vel.get_streamfunction_hill_spherical_vortex(XX, YY, x0=0, y0=0, gamma=1, a=50,
                                                           return_cylindrical_coords=True, reference_frame='lab')
        wdata = vel.psi2udata_cylindrical(psi, zz=zz, rrho=rrho, return_cartesian=False)
        omega = vel.curl(wdata, xx=XX, yy=YY)[..., 0]

        fig, ax, Q = graph.quiver(XX, YY, wdata[0, ...], wdata[1, ...], color='darkgreen', zorder=100, inc=5)
        graph.contour(XX, YY, psi, levels=[-1.5, -1, -0.5, 0], colors='k', linewidths=1)
        graph.color_plot(XX, YY, omega, ax=ax, cmap='bwr')
        # graph.color_plot(XX, YY, psi, ax=ax, cmap='bwr')
        graph.scatter([0], [a/np.sqrt(2)], ax=ax) # psi has an extremum at a/sqrt(2)
    """
    if gamma is not None:
        u = gamma / 5 / a
        print('get_streamfunction_hill_spherical_vortex():\n'
              '... gamma was given. Overwriting u to', u)
    u *= -1
    psi = np.empty_like(xx)

    # From now on, use the cylindrical coordinates (x, y, z) -> (rho, phi, z)
    # Input is (x, y). I treat the x as the z axis, and the np.abs(y) as rho.
    # For axisymmetric flows, phi dependence can be dropped.
    zz = xx - x0
    rrho = np.abs(yy - y0)

    # in cylindrical coordinate
    rrho = np.abs(rrho)
    rr = np.sqrt(zz ** 2 + rrho ** 2)
    cond = rr < a
    psi[cond] = - 3 / 4. * u * rrho[cond] ** 2 * (1 - rr[cond] ** 2 / a ** 2)
    psi[~cond] = 1 / 2. * u * rrho[~cond] ** 2 * (1 - a ** 3 / rr[~cond] ** 3)

    # gamma = 5 * a * u # circulaton of gamma
    uc = -1.5 * u  # velocity at the center of the vortex

    if reference_frame == 'lab':  # stream
        pass
    elif reference_frame in ['wind', 'absolute', 'vortex']:  # vortex moves at u when the outer fluid is at rest
        psi = psi - 1 / 2. * u * rrho ** 2
    # elif reference_frame == 'vortex':
    #     psi = psi - 1/2. * uc * rrho ** 2
    else:
        raise ValueError('... Choose reference_frame from [lab, vortex]')

    if return_cylindrical_coords:
        return psi, zz, rrho
    else:
        return psi


def get_streamfunction_vloop(xx, yy, x0=0, y0=0, gamma=1, R=1,
                             reference_frame='lab',
                             return_cylindrical_coords=False, verbose=True):
    """
    Returns a streamfunction of a vortex loop in cylindrical coordinates
    ... A loop consists of a vortex filament. i.e. the vorticity is concentrated on a point (zero cross-section)
    ... This makes the streamfunction on the filament to diverge.
    ... Similarly, the energy density diverges like rho*gamma^2*R*log(R/epsilon) as it approaches to the filament by epsilon.

    y(input) => rho in the cylindrical axis (distance from the symmetry axis)
    ^
    |
    |
    --------------------------> x(input) = z-axis in the cylindrical coordinates


    Parameters
    ----------
    xx: 2d array, positional grid
    yy: 2d array, positional grid
    x0: float, x-coordinate of the origin of the cylindrical coordinate
    y0: float, x-coordinate of the origin of the cylindrical coordinate
    gamma: float, circulation of a vortex loop
    R: float, radius of a vortex loop
    reference_frame: str, default- 'lab'
        ... Choose reference_frame from [lab, wind, vortex]
        ... "lab" refers to the rest frame
        ... "vortex" refers to the frame of reference of the vortex
    return_cylindrical_coords: bool, default- False

    Returns
    -------
    psi: 2d array, streamfunction(rrho, zz)
    rrho: 2d array
    zz: 2d array
    """
    zz = xx - x0
    rrho = np.abs(yy - y0)

    k2 = 4 * rrho * R / (zz ** 2 + (rrho + R) ** 2)
    k = np.sqrt(k2)

    c1 = special.ellipk(k2)  # complete elliptical integral of the first kind
    c2 = special.ellipe(k2)  # complete elliptical integral of the second kind

    psi_cylindrical = gamma / 2 / np.pi * np.sqrt(rrho * R) * ((2 / k - k) * c1 - 2 / k * c2)  # Lamb Sect 161 Eq.9

    if reference_frame == 'lab':
        pass
    elif reference_frame == 'vortex':
        # a = R / 86 # lamb, Sec.163. At this value, the center of the vortex is at the same speed as the center of the ring.
        a = R / 4.708751231442116
        uc = gamma / (4 * np.pi * R) * (np.log(8 * R / a) - 0.25)
        # uc = 0
        psi_cylindrical = psi_cylindrical - 1 / 2. * uc * rrho ** 2
    else:
        raise ValueError('... Choose reference_frame from [lab, vortex]')

    if return_cylindrical_coords:
        if verbose:
            print('get_streamfunction_vloop: Use psi2udata_cylindrical(psi, rrho, zz) to get a velocity field')
        # return psi_cylindrical, zz, yy-y0
        return psi_cylindrical, zz, rrho
    else:
        if verbose:
            print('get_streamfunction_vloop: Output in Cartesian coords is not implemented yet')
        sys.exit()


def get_streamfunction_thin_cored_vring(xx, yy, x0=0, y0=0, gamma=1., R=1., a=0.01,
                                        reference_frame='lab', uc=None,
                                        return_cylindrical_coords=True, verbose=True, notebook=True):
    """
    Returns a streamfunction of a thin-cored vortex ring in cylindrical coordinates

    Parameters
    ----------
    xx: 2d array, positional grid
    yy: 2d array, positional grid
    x0: float, x-coordinate of the origin of the cylindrical coordinate
    y0: float, x-coordinate of the origin of the cylindrical coordinate
    gamma: float, circulation of a vortex ring
    R: float, radius of a vortex ring
    a: float, cora radius of a vortex ring
    reference_frame: str, default- 'lab'
        ... Choose reference_frame from [lab, wind, vortex]
        ... "lab" refers to the rest frame
        ... "vortex" refers to the frame of reference of the vortex
    uc: float, self-induced velocity of a vortex ring, default: None
        ... If a<<r, uc = -gamma / (4 * np.pi * R) * (np.log(8 * R / a) - 0.25)
        ... For reference_frame=='vortex', it assumes that the induced velocity is
    return_cylindrical_coords: bool, default- False
    verbose: bool, default- True
    notebook: bool, default- True, whether to use tqdm_notebook instead of tqdm

    Returns
    -------
    psi: 2d array, streamfunction(zz, rrho)
    zz, rrho: 2d arrays, cylindrical coordinates, optional
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm

    gamma *= -1

    zz = xx - x0
    rrho = np.abs(yy - y0)
    deltaZ, deltaRho = zz[0, 1] - zz[0, 0], rrho[1, 0] - rrho[0, 0]
    insideCore = np.sqrt(zz ** 2 + (rrho - R) ** 2) <= a
    ii, jj = np.indices(zz.shape)
    psi = np.zeros_like(zz)

    # CORE STRUCTURE
    ## RIGID CORE: v=omega r (inside the core) i.e. omega=gamma/A
    ## i.e. omega=gamma/A
    A = np.pi * a ** 2
    omega = gamma / A
    #     omega = gamma / float(np.sum(insideCore)) # this might be better than omega = gamma / A.=> no sig. difference

    for i in tqdm(range(psi.shape[0]), desc='Streamfunction'):
        for j in range(psi.shape[1]):
            for k, l in zip(ii[insideCore], jj[insideCore]):  # k, l: source
                r1 = np.sqrt((zz[i, j] - zz[k, l]) ** 2 + (rrho[i, j] - rrho[k, l]) ** 2)
                r2 = np.sqrt((zz[i, j] - zz[k, l]) ** 2 + (rrho[i, j] + rrho[k, l]) ** 2)
                lambda_ = (r2 - r1) / (r2 + r1)
                c1 = special.ellipk(lambda_ ** 2)  # complete elliptical integral of the first kind
                c2 = special.ellipe(lambda_ ** 2)  # complete elliptical integral of the second kind
                psi_ = - 1 / np.pi * (c1 - c2) * (r1 + r2) * omega * deltaZ * deltaRho
                psi[i, j] += psi_

    ss = np.sqrt(zz[insideCore] ** 2 + (rrho[insideCore] - R) ** 2)
    psi[insideCore] = - 0.5 * omega * R * a ** 2 * (np.log(8 * R / a) - 1.5 - 0.5 * (ss / a) ** 2)

    # matching psi at the core interface
    if gamma < 0:
        psi0 = np.nanmax(psi[~insideCore])
        psi1 = np.nanmin(psi[insideCore])
    else:
        psi0 = np.nanmin(psi[~insideCore])
        psi1 = np.nanmax(psi[insideCore])
    #     psi[insideCore] *= psi0/psi1 * 1.1 # scale inside # this changes the circulation!
    psi[~insideCore] *= psi1 / psi0 * 1.045  # scale outside

    if notebook:
        from tqdm import tqdm as tqdm

    if reference_frame == 'lab':
        pass
    elif reference_frame == 'vortex':
        ucThinCoreApprox = -gamma / (4 * np.pi * R) * (
                    np.log(8 * R / a) - 0.25)  # this is only valid for a thin-cored a<<R (practially a/R<<0.01)
        if uc is None:
            uc = ucThinCoreApprox
            if verbose:
                print(f'... Ring velocity (thin-core approx.) is {ucThinCoreApprox}')
        else:
            print(
                f'... Ring velocity (thin-core approx.) is {ucThinCoreApprox}. uc={uc}. uc/ucThinCoreApprox={uc / ucThinCoreApprox:.4f}')
        psi = psi - 1 / 2. * uc * rrho ** 2
    else:
        raise ValueError('... Choose reference_frame from [lab, vortex]')

    if return_cylindrical_coords:
        if verbose:
            print(
                'get_streamfunction_thin_cored_vring: Use psi2udata_cylindrical(psi, rrho, zz) to get a velocity field')
        # return psi_cylindrical, zz, yy-y0
        return psi, zz, rrho
    else:
        if verbose:
            print(
                'get_streamfunction_thin_cored_vring: Output in Cartesian coords is not implemented yet. Returning None')
        return None


# 3D streamlines
## Sample n streamlines (randomly)
def sample_streamlines(udata, xx, yy, zz, npt=1,
                       xmin=0, xmax=None, ymin=0, ymax=None, zmin=0, zmax=None,
                       t0=0, t1=None, dt=0.01):
    """
    Returns n streamlines which go through n randomly chosen points inside xx, yy, zz
    ... How this works: Basically, seed particles at the instant of time, and track how they move by the instantaneous vel field.
    ... Streamlines are a family of curves that are instantaneously tangent to the velocity vector of the flow.

    Parameters
    ----------
    udata: nd array, shape (nx, ny, nz, 3)
    xx: nd array, shape (nx, ny, nz)
    yy: nd array, shape (nx, ny, nz)
    zz: nd array, shape (nx, ny, nz)
    npt: float, number of streamlines
        ... to be precise, this is the number of points through which streamlines are drawn.
    xmin: int
        ... The initial points were taken from inside the box whose diagonal vertices are
            (xx[ymin, xmin, zmin], yy[ymin, xmin, zmin], zz[ymin, xmin, zmin])
            and
            (xx[ymax, xmax, zmax], yy[ymax, xmax, zmax], zz[ymax, xmax, zmax])
    xmax: int, optional
    ymin: int, optional
    ymax: int, optional
    zmin: int, optional
    zmax: int, optional
    t0: int, time index, it gets the streamlines using udata[..., t0:t1]
    t1: int, time index, it gets the streamlines using udata[..., t0:t1]
    dt: float, param to control the accuracy of the 3D streamlines

    Returns
    -------
    strmlines_master: list
        ... length of the list is duration (t1-t0)
        ...... strmlines_master[10] are a collection of n streamlines at frame 10
        ... If t1-t0 == 1 (given udata is a snapshot of the velocity field):
            it returns strmlines_master[0]

    """

    def create_a_streamline(x0, y0, z0, funcs, xx, yy, zz,
                            dt=0.01, nmax=1000,
                            translations=[0, 0, 0],
                            spacing=1.):
        """
        Create a streamline when point coordinates are given
        All computations are done in the index space not physical space.
        Hence, input positions, velocity functions, spatial grids must be in the index space
        """

        def insideBox(x, y, z, xx, yy, zz):
            xmin, xmax = np.nanmin(xx), np.nanmax(xx)
            ymin, ymax = np.nanmin(yy), np.nanmax(yy)
            zmin, zmax = np.nanmin(zz), np.nanmax(zz)
            w, h, d = xmax - xmin, ymax - ymin, zmax - zmin
            return x > xmin and x < xmax and y > ymin and y < ymax and z > zmin and z < zmax

        def sort_two_arrays_using_order_of_first_array(arr1, arr2):
            """
            Sort arr1 and arr2 using the order of arr1
            e.g. a=[2,1,3], b=[4,1,9]-> a[1,2,3], b=[1,4,9]
            Parameters
            ----------
            arr1
            arr2

            Returns
            -------

            """
            arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))

            return np.asarray(arr1), np.asarray(arr2)

        step = 0
        f, g, h = funcs
        xs, ys, zs = [x0], [y0], [z0]
        steps = [step]
        while insideBox(x0, y0, z0, xx, yy, zz) and step < nmax:
            # Compute the position at the next time step
            ux, uy, uz = f([y0, x0, z0])[0], g([y0, x0, z0])[0], h([y0, x0, z0])[0]
            dx, dy, dz = ux * dt, uy * dt, uz * dt
            x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

            # Insert data
            xs.append(x1)
            ys.append(y1)
            zs.append(z1)
            steps.append(step + 1)

            # Update x0, y0, z0
            x0, y0, z0 = x1, y1, z1
            step += 1

        # Now go back in time
        x0, y0, z0 = xs[0], ys[0], zs[0]
        step = 0
        while insideBox(x0, y0, z0, xx, yy, zz) and np.abs(step) < nmax:
            # Compute the position at the next time step
            ux, uy, uz = f([y0, x0, z0])[0], g([y0, x0, z0])[0], h([y0, x0, z0])[0]
            dx, dy, dz = -ux * dt, -uy * dt, -uz * dt
            x1, y1, z1 = x0 + dx, y0 + dy, z0 + dz

            # Insert data
            xs.append(x1)
            ys.append(y1)
            zs.append(z1)
            steps.append(step - 1)

            # Update x0, y0, z0
            x0, y0, z0 = x1, y1, z1
            step -= 1
        _, xs = sort_two_arrays_using_order_of_first_array(steps, xs)
        _, ys = sort_two_arrays_using_order_of_first_array(steps, ys)
        steps, zs = sort_two_arrays_using_order_of_first_array(steps, zs)

        # alter array shape, and convert index into physical units
        strmline = np.stack((xs * spacing + translations[0],
                             ys * spacing + translations[1],
                             zs * spacing + translations[2]))
        return strmline

    Xmin, Xmax = np.nanmin(xx), np.nanmax(xx)
    Ymin, Ymax = np.nanmin(yy), np.nanmax(yy)
    Zmin, Zmax = np.nanmin(zz), np.nanmax(zz)
    dx, dy, dz = get_grid_spacing(xx, yy, zz)

    udata = fix_udata_shape(udata)
    dim, height, width, depth, duration = udata.shape

    yy_i, xx_i, zz_i = get_equally_spaced_grid(udata, spacing=1)  # spatial grid (indices- not in physical units)
    if xmax is None:
        xmax = width
    if ymax is None:
        ymax = height
    if zmax is None:
        zmax = depth
    if t1 is None:
        t1 = duration

    # 1D arrays of udata grid (indices)
    x, y, z = np.arange(width), np.arange(height), np.arange(depth)

    strmlines_master = []
    x0s_list, y0s_list, z0s_list = [], [], []
    for t in range(t0, t1):
        f = RegularGridInterpolator((y, x, z), udata[0, ..., t] / dx)  # ux (px/s)
        g = RegularGridInterpolator((y, x, z), udata[1, ..., t] / dy)  # uy (px/s)
        h = RegularGridInterpolator((y, x, z), udata[2, ..., t] / dz)  # uz (px/s)

        # starting positions (indices) of particles
        x0s = xmin + np.random.random(npt) * (xmax - xmin)
        y0s = ymin + np.random.random(npt) * (ymax - ymin)
        z0s = zmin + np.random.random(npt) * (zmax - zmin)

        # get n streamlines every frame
        strmlines = []
        for n, (x0, y0, z0) in enumerate(tqdm(zip(x0s, y0s, z0s), total=len(x0s))):
            strmline = create_a_streamline(x0, y0, z0, [f, g, h], xx_i, yy_i, zz_i,
                                           dt=dt,
                                           spacing=dx,
                                           translations=[Xmin, Ymin, Zmin])
            strmlines.append(strmline)
        # strmlines_master stores the all streamlines across frames
        strmlines_master.append(strmlines)

        x0s_list.append(x0s * dx + Xmin)
        y0s_list.append(y0s * dy + Ymin)
        z0s_list.append(z0s * dz + Zmin)

    if len(strmlines_master) == 1:
        return strmlines_master[0]
    else:
        return strmlines_master



# APPROACH2: Solve a poisson equation (This will not work in general!)
def get_streamfunction_poisson_solver(udata,
                                      omega=None, dx=1., dy=1.,
                                      xx=None, yy=None, verbose=True):
    """
    Computes the streamfunction by solving a poisson equation

    Parameters
    ----------
    udata: ndarray, velocity field
    omega: ndarray, vorticity field
    dx: float, grid spacing in x
    dy: float, grid spacing in y
    xx: ndarray, x grid
    yy: ndarray, y grid
    verbose: bool, print progress

    Returns
    -------
    psi
    """
    print('get_streamfunction_poisson_solver(): DEPRECATED. \n'
          '... It outputs correct results iff the omega=0 at the boundaries.')

    def normalize_positional_grid(xx):
        xmin, xmax = np.nanmin(xx), np.nanmax(xx)
        w = xmax - xmin
        xx_n = (xx - xmin) / w
        return xx_n

    udata = fix_udata_shape(udata)
    dim, ny, nx, duration = udata.shape

    # Probably it is healthy to check if the field is non-divergent
    # dx, dy = get_grid_spacing(xx, yy)
    # divergence = np.nanmean(div(udata, dx=dx, dy=dy), axis=(0, 1))
    xx_norm = normalize_positional_grid(xx)
    yy_norm = normalize_positional_grid(yy)

    # initialization
    psi = np.empty((ny, nx, duration))
    # omega_recovered = np.empty((ny, nx, duration)) # debugging purpose

    if xx is not None and yy is not None:
        dx, dy = get_grid_spacing(xx_norm, yy_norm)
    if omega is None:
        if verbose:
            print('... vorticity is computed using (dx, dy)=(%f, %f)' % (dx, dy))
            print('... Provide dx, dy if they are not correct')
            print('... It is also recommended to supply xx, yy to ensure that curl() works')
        omega = curl(udata, dx=dx, dy=dy, xx=xx_norm, yy=yy_norm)
    else:
        if verbose:
            print('... Solving for psi in \del^2 psi = omega')
            print('... vorticity is given by the user')
            print('... (dx, dy)=(%f, %f)' % (dx, dy))

    invlapl = poisson_fft_prep(nx, ny, dx, dy, lapl='discrete')  # lapl='discrete') #lapl='calculus' or lapl='discrete'
    for t in range(duration):
        psi[..., t] = poisson_fft(omega[..., t], invlapl)
        # omega_recovered[..., t] = laplacian(psi[..., t], dx, dy) # For debugging- this should be identical to omega at t=t
    return psi


## Poisson solver (2D)
def poisson_fft_prep(Nx, Ny, dx, dy, lapl='discrete'):
    """
    Returns the coefficients to multiply the vorticity Fourier amplitudes

    Source: https://anaconda.org/bfiedler/n090_streamfunctionvorticity2d/notebook?version=2018.07.28.0933

    Parameters
    ----------
    Nx: int, number of grid points in x
    Ny: int, number of grid points in x
    dx: float, grid spacing in x
    dy: float, grid spacing in y
    lapl: str, 'discrete'

    Returns
    -------
    invlapl: 2d array
        ... the coefficents for multiplying the vorticity Fourier amplitudes
    """
    # returns the coefficients to multiply the vorticity Fourier amplitudes
    L = dx * (Nx - 1)
    W = dy * (Ny - 1)

    Ka = np.arange(Nx - 2) + 1  # integer wavenumbers of the sine functions in the x-direction
    Ma = np.arange(Ny - 2) + 1  # integer wavenumbers of the sine functions in the y-direction
    ka = Ka * np.pi / L
    ma = Ma * np.pi / W

    lapl_op = np.zeros((Ny - 2, Nx - 2))
    if lapl == 'discrete':
        lapl_op[:] += (2 * np.cos(ka * dx) - 2) / dx ** 2  # add to every row
    else:  # the calculus Laplacian
        lapl_op[:] += -ka ** 2
    lapl_opT = lapl_op.T  # reverse columns and rows
    if lapl == 'discrete':
        lapl_opT[:] += (2 * np.cos(ma * dy) - 2) / dy ** 2  # add to every row
    else:  # the calculus Laplacian
        lapl_opT[:] += -ma ** 2
    lapl_op = lapl_opT.T  # reverse columns and rows
    invlapl = 1. / lapl_op  # the coefficents for multiplying the vorticity Fourier amplitudes
    return invlapl


def poisson_fft(vort, invlapl):
    """
    Solves for psi in del^2 psi = vort

    Source: https://anaconda.org/bfiedler/n090_streamfunctionvorticity2d/notebook?version=2018.07.28.0933

    Parameters
    ----------
    vort: 2d array
    invlapl: 2d array
        ... inverse laplacian operator
        ... can be obtained from poisson_fft_prep()

    Returns
    -------
    psi: 2d array
    """
    # solves for psi in del^2 psi = vort
    cv = vort[1:-1, 1:-1]  # central vorticity

    # convert gridded vorticity to gridded Fourier coefficients A_k,m
    cvt = fftpack.dst(cv, axis=1, type=1)
    cvt = fftpack.dst(cvt, axis=0, type=1)

    cpsit = cvt * invlapl  # Calculate B_k,m from A_k,m

    # convert array of Fourier coefficents for psi to gridded central psi
    cpsit = fftpack.idst(cpsit, axis=0, type=1)  # inverse transform
    cpsi = fftpack.idst(cpsit, axis=1, type=1)  # inverse transform

    sh = vort.shape
    psi = np.zeros(sh)  # we need 0 on boundaries, next line fills the center
    psi[...] = np.nan  # we need 0 on boundaries, next line fills the center
    psi[1:-1, 1:-1] = cpsi / (4 * (sh[0] - 1) * (sh[1] - 1))  # apply normalization convention of FFT
    return psi


def laplacian(p, dx, dy, il=None, ir=None, jb=None, jt=None):
    """
    Returns Laplacian of p, d^2p/dx^2 + d^2/dy^2.
    If needed, specify how to grab the image of a point outside
    the domain.  Otherwise, the d^2p/dx^2 or d^2/dy^2 term is not included
    on the boundary.

    Source: https://anaconda.org/bfiedler/n090_streamfunctionvorticity2d/notebook?version=2018.07.28.0933

    e.g. # Compute vorticity from a stream function
        laplacian(streamfunction, dx, dy)


    Parameters
    ----------
    p: 2d array, a scalar field
    dx: float, grid spacing in x
    dy: float, grid spacing in y
    il:
    ir
    jb
    jt

    Returns
    -------
    lapl: 2d array
    """
    rdx2 = 1. / (dx * dx)
    rdy2 = 1. / (dy * dy)
    lapl = np.zeros(p.shape)
    lapl[:, 1:-1] = rdx2 * (p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:])
    lapl[1:-1, :] += rdy2 * (p[:-2, :] - 2 * p[1:-1, :] + p[2:, :])
    if il in [-2, -1, 0, 1]:
        lapl[:, 0] += rdx2 * (p[:, il] - 2 * p[:, 0] + p[:, 1])
    if ir in [-2, -1, 0, 1]:
        lapl[:, -1] += rdx2 * (p[:, -2] - 2 * p[:, -1] + p[:, ir])
    if jb in [-2, -1, 0, 1]:
        lapl[0, :] += rdy2 * (p[jb, :] - 2 * p[0, :] + p[1, :])
    if jt in [-2, -1, 0, 1]:
        lapl[-1, :] += rdy2 * (p[-2, :] - 2 * p[-1, :] + p[jt, :])
    return lapl


# helpers for turbulent blob exp
def compute_form_no(stroke_length, orifice_d=25.6, piston_d=160., num_orifices=8, setting=None):
    """
    Returns a formation number with stroke length as an input
    ... Old box (small): orifice_d=20, piston_d=125
    ... New box (3D printed): orifice_d=25.6, piston_d=160

    Parameters
    ----------
    stroke_length: float/1d array, stroke length in mm
    orifice_d: float, orifice diameter in mm
    piston_d: float, piston diameter in mm
    num_orifices: int, number of orifices

    Returns
    -------
    LD: float/1d array, formation number
    """
    dp, do, N = piston_d, orifice_d, num_orifices
    if setting == 'medium':
        dp, do, N = 160, 25.6, 8.  # Setting 1
    elif setting == 'small':  # Setting 2
        dp, do, N = 56.7, 12.8, 8.
    LD = (dp / do) ** 2 * stroke_length / do / N
    return LD


def estimate_veff(sl, sv):
    """
    Returns effective velocity from a given commanded stroke length and velocity
    ... This function interpolates values based on the measurements in the past

    Parameters
    ----------
    sl: int/float/1d array
        - Commanded stroke length in mm
    sv: int/float/1d array
        - Commanded stroke velocity in mm/s

    Returns
    -------
    veff: 1d array
        - Effective velocity in mm/s
        - Velocity program factor * mean speed
        - Velocity program factor P = Mean of the second moment / Square of the mean of the first moment (average)
    """

    if isinstance(sl, np.ndarray):
        if sl.ndim == 0: sl = float(sl)
    if isinstance(sv, np.ndarray):
        if sv.ndim == 0: sv = float(sv)
    if type(sl) in [int, float]:
        sl, sv = [sl], [sv]
    pts2estimate = list(zip(sl, sv))
    # commanded velocity (mm)
    sl_cmd = [2.6, 2.6, 2.6, 2.6, 2.6,
              5.2, 5.2, 5.2, 5.2, 5.2, 5.2,
              7.800000000000001, 7.800000000000001, 7.800000000000001, 7.800000000000001, 7.800000000000001, 7.8,
              10.4, 10.4, 10.4, 10.4, 10.4,
              13.0, 13.0, 13.0, 13.0, 13.0, 13.0,
              15.600000000000001, 15.600000000000001, 15.600000000000001, 15.60000000000001, 15.600000000000001,
              15.600000000000001,
              18.2, 18.2, 18.2, 18.2, 18.2, 18.2,
              20.8, 20.8, 20.8, 20.8, 20.8, 20.8,
              23.400000000000002, 23.400000000000002]
    # commanded velocity (mm/s)
    vp_cmd_sorted = [100, 200, 300, 400, 1000,
                     50, 100, 200, 300, 400, 1000,
                     50, 100, 200, 300, 400, 1000,
                     100, 200, 300, 400, 1000,
                     50, 100, 200, 300, 400, 1000,
                     50, 100, 200, 300, 400, 1000,
                     50, 100, 200, 300, 400, 1000,
                     50, 100, 200, 300, 400, 1000,
                     400, 1000]
    # effective velocity
    veff_data_sorted = [101.49904052529494, 120.65398433821005, 132.64713389903653, 150.54889987470946,
                        152.11259965582218,
                        52.41700229303288, 104.41173411234168, 210.09654261275634, 269.4166774275767, 309.410059801625,
                        317.2246281837468,
                        49.42329705118604, 98.12842281108405, 195.83154238039648, 283.3434393869659, 398.3035669516888,
                        401.04887767,
                        101.94968594333488, 212.96098308225953, 374.8247002126497, 418.2011203579417, 420.,
                        52.031669318230065, 105.1845578538833, 213.6180029013978, 325.8482337883638, 405,
                        419.23942762100165,  # 335.94 instead of 405?
                        52.06532370156262, 103.19137134275651, 208.1194793168665, 325.628713002109, 442.99802519923526,
                        525.3650045528688,
                        52.236702955826594, 104.55176927251647, 207.09067744554528, 318.4794962481438,
                        440.9965102573683, 589.5473184534424,
                        52.51967045608249, 104.30639832142036, 206.27539423030547, 313.5499443061134,
                        435.22484204357414, 641.7059086973986,
                        425.65185384290123, 683.6223751807265]
    points = np.dstack((sl_cmd, vp_cmd_sorted))[0, ...]
    veff = interpolate.griddata(points, veff_data_sorted, pts2estimate)  # stroke length, commanded_velocity

    if len(veff) == 1:
        veff = veff[0]
    return veff


def isRingFormed(sl, sv, f, P=1.0, do=25.6, dp=160., n=8, ldmax=4.5, amax=1.8e4):
    """
    Returns if a commanded ring is formed in the turbulent blob experiment
    ... A ring is formed regardless of commanded stroke length and velocity; however, the piston may not move as commanded
    ... This function tells if a ring was formed cleanly
    ... Condition 1: L/D<=4... Ring is accompanied by a jet if L/D>4
    ... Condition 2: Piston moves with finite acceleration. Effective velocity, veff, is always less than a certain value.
    ... Condition 3: Oscillatory motion sets the minimum velocity of the piston needs to be.

    Parameters
    ----------
    sl: float/1d array, commanded stroke length in mm
    sv: float/1d array, commanded stroke velocity in mm/s
    f: float/1d array, driving frequency of the piston
    P: float, velocity shape factor <v^2>/<v>^2>=1. Experimentally, this is around 1.2.
    do: float, diameter of the orifice
    dp: float, diameter of the piston
    n: int, number of orifices of the chamber
    amax: float, maximum acceleration of the piston. Experimentally, this is 1.8e4 mm/s2. Relevant to Condition2.

    Returns
    -------
    isRingFormed: bool, True if the ring could be formed at the given driving frequency
    (the formed ring would have the same or similar properties as the rings formed by a corresponding pulse)
    False if the formed ring would have different properties from the rings formed by a corresponding pulse.
    """

    veff = estimate_veff(sl, sv)
    ld = compute_form_no(sl, orifice_d=do, piston_d=dp, num_orifices=n)

    cond1 = ld <= ldmax
    # cond2 = veff <= (P / 2. * (amax * do) ** 0.5 * ld ** 0.5) #This just sets the upper bound
    cond2 = True
    cond3 = veff > (P * f * do) * ld
    print(veff, (P * f * do) * ld)
    isRingFormed = cond1 * cond2 * cond3
    return isRingFormed


def smooth(x, window_len=11, window='hanning', log=False):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with a given signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    Parameters
    ----------
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Returns
    -------
        the smoothed signal

    Example
    -------
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)

        see also:

        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.filter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth() only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if log:
        x = np.log(x)

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    if not log:
        return y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)]
    else:
        return np.exp(y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)])


def smoothAlongAxis(data, n=5, window='hanning'):
    """Takes a 2D array, returns a smooth array along a axis using a specified kernel"""
    data_smt = np.empty_like(data)
    for i in range(data.shape[-1]):
        data_smt[:, i] = smooth(data[:, i], window_len=n, window=window)
    return data_smt


# Coarse-graining a field
def coarse_grain_udata(udata, nrows_sub, ncolumns_sub, overwrap=0.5, xx=None, yy=None, notebook=True):
    """
    Returns a coarse grained udata
    ... so far, this can handle only 2D udata
    Parameters
    ----------
    udata: nd array, velocity field
    nrows_sub: int, number of rows in the sub-grid
    ncolumns_sub: int, number of columns in the sub-grid
    overwrap: float, fraction of the sub-grid that is overlapped with the original grid, [0, 1]
    xx: nd array, x-coordinates of the original grid
    yy: nd array, y-coordinates of the original grid

    Returns
    -------
    udata_sub: nd array, coarse grained velocity field
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]
    if dim == 3:
        raise ValueError('coarse_grain_udata: 3D udata is not implemented yet')

    dummy = coarse_grain_2darr_overwrap(udata[0, ..., 0], nrows_sub, ncolumns_sub, overwrap=overwrap)
    shape = (dim,) + dummy.shape + (duration,)
    udata_c = np.empty(shape)
    for t in tqdm(range(duration)):
        for d in range(dim):
            udata_c[d, ..., t] = coarse_grain_2darr_overwrap(udata[d, ..., t], nrows_sub, ncolumns_sub,
                                                             overwrap=overwrap)

    if xx is not None and yy is not None:
        xx_c = coarse_grain_2darr_overwrap(xx, nrows_sub, ncolumns_sub, overwrap=overwrap)
        yy_c = coarse_grain_2darr_overwrap(yy, nrows_sub, ncolumns_sub, overwrap=overwrap)

    if notebook:
        from tqdm import tqdm

    if xx is not None and yy is not None:
        return udata_c, xx_c, yy_c
    else:
        return udata_c


def coarse_grain_2darr(arr, nrows_sub, ncolumns_sub):
    """
    Coarse-grain a 2D array
    ... Coarse-graining a MxN array means a following procedure.
        1. Divide an MxN array into blocks of (m x n) arrays
        ... m=nrows_sub, n=ncolumns_sub
        2. Replace each block by an average of the values in the block
    ... If you are not familiar with coarse-graining, consult with Kadanoff's block-spin renormalization group.

    Parameters
    ----------
    arr: 2d array
    nrows_sub: int, Number of rows of blocks (over which values are averaged)
    ncolumns_sub: int, Number of columns of blocks

    Returns
    -------
    arr_coarse: coarse-grained 2d arr

    """
    arr = np.asarray(arr)
    nrows, ncols = arr.shape

    # If the 2d array cannot be separated into blocks, then extend/pad the 2d array
    remainder_row = nrows % nrows_sub
    remainder_column = ncols % ncolumns_sub
    if not remainder_row == 0 or not remainder_column == 0:
        print('Shape is not an integer multiple of (nrows_sub, ncolumns_sub)!')
        print('Will extend the array with np.nan, and average...')
        nrows = int(np.ceil(arr.shape[0] / float(nrows_sub)) * nrows_sub)
        ncols = int(np.ceil(arr.shape[1] / float(ncolumns_sub)) * ncolumns_sub)
        arr = extend_2darray_fill(arr, (nrows, ncols), fill_value='np.nan')

    nrows_coarse, ncolumns_corarse = int(nrows / nrows_sub), int(ncols / ncolumns_sub)

    # make blocks from 2d array (nrows, ncols) -> (nblocks, nrows_sub, ncolumns_sub)
    arr_blocks = make_blocks_from_2d_array(arr, nrows_sub, ncolumns_sub)
    # Average inside the blocks, and reshape the array
    arr_coarse = np.nanmean(arr_blocks, axis=(1, 2)).reshape(nrows_coarse, ncolumns_corarse)

    return arr_coarse


def coarse_grain_2darr_overwrap(arr, nrows_sub, ncolumns_sub, overwrap=0.5):
    """
    Coarse-grain 2D arrays with overwrap (mimics how PIVLab processes a velocity field)

    arr= [[ 0  1  2  3  4  5]
         [ 6  7  8  9 10 11]
         [12 13 14 15 16 17]
         [18 19 20 21 22 23]
         [24 25 26 27 28 29]
         [30 31 32 33 34 35]]

        -> Make a new array. (nrows_sub=4, ncolumns_sub=4, overwrap=0.5)
    array([[  0.,   1.,   2.,   3.,   2.,   3.,   4.,   5.],
           [  6.,   7.,   8.,   9.,   8.,   9.,  10.,  11.],
           [ 12.,  13.,  14.,  15.,  14.,  15.,  16.,  17.],
           [ 18.,  19.,  20.,  21.,  20.,  21.,  22.,  23.],
           [ 12.,  13.,  14.,  15.,  14.,  15.,  16.,  17.],
           [ 18.,  19.,  20.,  21.,  20.,  21.,  22.,  23.],
           [ 24.,  25.,  26.,  27.,  26.,  27.,  28.,  29.],
           [ 30.,  31.,  32.,  33.,  32.,  33.,  34.,  35.]])

        -> Coarse-grain (output)
    array([[ 10.5,  12.5],
           [ 22.5,  24.5]])

    Parameters
    ----------
    arr: 2d array
    nrows_sub: int, Number of rows of blocks (over which values are averaged)
    ncolumns_sub: int, Number of columns of blocks
    overwrap: fraction of overwrap, default=0.5, [0, 1]

    Returns
    -------
    arr_coarse: coarse-grained 2d arr

    """
    nrows, ncols = np.array(arr).shape
    rowstep, colstep = int(nrows_sub * overwrap), int(ncolumns_sub * overwrap)
    # nrows_new, ncols_new = (nrows-1) * nrows_sub, (ncols-1) * ncolumns_sub
    # number of overwrapped regions
    nrow_ow, ncol_ow = int(np.ceil((nrows - nrows_sub) / (nrows_sub * (1 - overwrap)))), int(
        np.ceil((ncols - ncolumns_sub) / (ncolumns_sub * (1 - overwrap))))
    # shape of new array
    nrows_new, ncols_new = nrows_sub * (nrow_ow + 1), ncolumns_sub * (ncol_ow + 1)
    arr_new = np.empty((nrows_new, ncols_new))
    arr_new[...] = np.nan

    # Make a new array to coarse grain
    for i in range(0, nrows_new, nrows_sub):
        for j in range(0, ncols_new, ncolumns_sub):
            ii, jj = int(np.ceil(i * (1 - overwrap))), int(np.ceil(j * (1 - overwrap)))
            if i % nrows_sub == 0 and j % ncolumns_sub == 0:
                # print (i, j), (ii, jj)
                # print arr[ii:ii+nrows_sub, jj:jj+ncolumns_sub]
                try:
                    arr_new[i:i + nrows_sub, j: j + ncolumns_sub] = arr[ii:ii + nrows_sub, jj:jj + ncolumns_sub]
                except ValueError:
                    arr_new[i:i + nrows_sub, j: j + ncolumns_sub] = extend_2darray_fill(
                        arr[ii:ii + nrows_sub, jj:jj + ncolumns_sub], (nrows_sub, ncolumns_sub))
            else:
                # print (i, j), (ii, jj), 'skip'
                continue
            # print arr_new

    # Coarse-grain
    # Make blocks from 2d array (nrows, ncols) -> (nblocks, nrows_sub, ncolumns_sub)
    arr_blocks = make_blocks_from_2d_array(arr_new, nrows_sub, ncolumns_sub)
    # Average inside the blocks, and reshape the array
    nrows_coarse, ncolumns_corarse = int(nrows_new / nrows_sub), int(ncols_new / ncolumns_sub)
    arr_coarse = np.nanmean(arr_blocks, axis=(1, 2)).reshape(nrows_coarse, ncolumns_corarse)

    return arr_coarse


def extend_2darray_fill(arr, newarrshape, fill_value=np.nan):
    """
    Resize a 2d array while keeping the physical shape of the original array and fill the rest with something
    e.g.-
    arr =
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19]])

    -> arr_ext with newarrshape = (6,6)
        array([[  0.,   1.,   2.,   3.,   4.,  nan],
               [  5.,   6.,   7.,   8.,   9.,  nan],
               [ 10.,  11.,  12.,  13.,  14.,  nan],
               [ 15.,  16.,  17.,  18.,  19.,  nan],
               [ nan,  nan,  nan,  nan,  nan,  nan],
               [ nan,  nan,  nan,  nan,  nan,  nan]])

    Parameters
    ----------
    arr: 2d numpy array
    newarrshape: tuple, new array shape ... (nrows, ncols)
    fill_value: float, default=np.nan, value to fill the rest of the array with

    Returns
    -------
    arr_ext: extended/padded 2d array
    """
    arr = np.array(arr)
    shape = arr.shape
    arr_ext = np.full(newarrshape, fill_value)
    arr_ext[0:shape[0], 0:shape[1]] = arr
    return arr_ext


def make_blocks_from_2d_array(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where n * nrows * ncols = arr.size
    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    Parameters
    ----------
    arr: M x N list or numpy array
    nrows: int, number of rows in each block
    ncols: int, number of columns in each block

    Returns
    -------
    blocks: numpy array with shape (n, nrows, ncols)
    """

    arr = np.asarray(arr)
    h, w = arr.shape
    blocks = (arr.reshape(h // nrows, nrows, -1, ncols)
              .swapaxes(1, 2)
              .reshape(-1, nrows, ncols))
    return blocks


# Coarse-graining 3D
def coarse_grain_3darr(arr, nrow_sub, ncol_sub, ndep_sub, overwrap=0, showtqdm=True, notebook=True, verbose=False):
    """
    Coarse-grain a 3d array
    ... The idea is to split the original array into many subcells, and average over each subcell
    ... The cell size is (nrow_sub, ncol_sub, ndep_sub)

    Parameters
    ----------
    arr: 3d array to be coarse-grained
    nrow_sub: int, number of rows of the subcell
    ncol_sub: int, number of columns of the subcell
    ndep_sub: int, number of depths(steps) of the subcell
    overwrap: float, [0, 1)
    showtqdm: bool

    Returns
    -------
    new_arr: coarse-grained 3d array

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        if verbose:
            print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    try:
        height, width, depth = arr.shape
        duration = 1
        new_shape = (height, width, depth, duration)
        arr = np.reshape(arr, new_shape)
    except:
        height, width, depth, duration = arr.shape

    ii = [int(np.floor(i * (1 - overwrap) * nrow_sub)) for i in
          range(int(np.floor(height / (1 - overwrap) / nrow_sub)))]
    jj = [int(np.floor(i * (1 - overwrap) * ncol_sub)) for i in range(int(np.floor(width / (1 - overwrap) / ncol_sub)))]
    kk = [int(np.floor(i * (1 - overwrap) * ndep_sub)) for i in range(int(np.floor(depth / (1 - overwrap) / ndep_sub)))]
    new_h, new_w, new_d = len(ii), len(jj), len(kk)
    new_arr = np.empty((new_h, new_w, new_d, duration))

    for t in tqdm(range(duration), disable=~showtqdm, desc='coarse_grain_3darr: time loop'):
        for p, i in enumerate(tqdm(ii, disable=~showtqdm, desc='coarse_grain_3darr: row loop')):
            for q, j in enumerate(jj):
                for r, k in enumerate(kk):
                    new_arr[p, q, r, t] = np.nanmean(arr[i:i + nrow_sub, j:j + ncol_sub, k:k + ndep_sub, t])

    if notebook:
        from tqdm import tqdm
    if duration == 1:
        return new_arr[..., 0]
    else:
        return new_arr


def coarse_grain_3dudata(udata_3d, nrow_sub, ncol_sub, ndep_sub, overwrap=0, showtqdm=False, notebook=True):
    """

    Coarse-grain a 3d udata
    ... 3d udata has a shape (dim=3, nrows, ncols, ndeps, duration) or (dim=3, nrows, ncols, ndeps)
    ... The idea is to split the original array into many subcells, and average over each subcell
    ... The cell size is (nrow_sub, ncol_sub, ndep_sub)

    Parameters
    ----------
    udata_3d: 4d or 5d array, udata
        ... udata is just a nd array.
    nrow_sub: int, number of rows of the subcell
    ncol_sub: int, number of columns of the subcell
    ndep_sub: int, number of depths(steps) of the subcell
    overwrap: float, [0, 1)
    showtqdm: bool

    Returns
    -------
    udata_cg: coarse-grained ud

    """

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata_3d = fix_udata_shape(udata_3d)
    dim, height, width, depth, duration = udata_3d.shape
    ii = [int(np.floor(i * (1 - overwrap) * nrow_sub)) for i in
          range(int(np.floor(height / (1 - overwrap) / nrow_sub)))]
    jj = [int(np.floor(i * (1 - overwrap) * ncol_sub)) for i in range(int(np.floor(width / (1 - overwrap) / ncol_sub)))]
    kk = [int(np.floor(i * (1 - overwrap) * ndep_sub)) for i in range(int(np.floor(depth / (1 - overwrap) / ndep_sub)))]
    new_h, new_w, new_d = len(ii), len(jj), len(kk)
    new_shape = (dim, new_h, new_w, new_d, duration)
    udata_cg = np.empty(new_shape)

    for d in range(dim):
        udata_cg[d, ...] = coarse_grain_3darr(udata_3d[d, ...], nrow_sub, ncol_sub, ndep_sub,
                                              overwrap=overwrap, showtqdm=showtqdm, verbose=False)
    if notebook:
        from tqdm import tqdm
    return udata_cg


# HELPERS FOR Fourier Transform (DFT/FFT and CFT)
# 1D FFT and CFT
def fft_1d(gt):
    """
    Returns the shifted FFT of a 1D signal

    Parameters
    ----------
    gt: 1d array

    Returns
    -------
    gf: 1d complex array

    """
    return np.fft.fftshift(np.fft.fft(gt))


def cft_1d(g, f):
    """
    Numerically evaluate the Fourier Transform of a function 'g' for given frequencies

    Parameters
    ----------
    g: function
    f: array-like, frequencies at which Fourier coefficients are calculated

    Returns
    -------
    result: Fourier coefficients at given frequency

    """

    def complex_quad(g, a, b):
        """Return definite integral of complex-valued g from a to b,
        using Simpson's rule"""
        # 2501: Amount of used samples for the trapezoidal rule
        t = np.linspace(a, b, 2501)
        x = g(t)
        return integrate.simps(y=x, x=t)

    result = np.zeros(len(f), dtype=complex)

    # Loop over all frequencies and calculate integral value
    for i, ff in enumerate(f):
        # Evaluate the Fourier Integral for a single frequency ff,
        # assuming the function is time-limited to abs(t)<5
        result[i] = complex_quad(lambda t: g(t) * np.exp(-2j * np.pi * ff * t), -5, 5)
    return result


def FFTOutput2CFTOutput_1d(gf, t):
    """
    Approximates the FFT of a 1D signal to the Continuous Fourier Transform (CFT)

    Parameters
    ----------
    gf: 1d array, FFT output of a 1d sample g(t)
    t: 1d array, a variable (time, space, etc) of the original function g(t)

    Returns
    -------
    gf_cft: 1d array, Appriximated continuous fourier transform
    """
    t0 = t[0]
    dt = t[1] - t[0]
    n = len(gf)
    fs = 1 / dt

    f = np.linspace(-fs / 2, fs / 2, n, endpoint=False)
    gf_cft = gf * np.exp(-2j * np.pi * f * t0) * 1 / fs
    return gf_cft


def fourier_transform_1d(samples, fs, t0, return_freq=False):
    """Approximate the Fourier Transform of a time-limited
    signal by means of the discrete Fourier Transform.

    samples: signal values sampled at the positions t0 + n/Fs
    fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal

    Returns:
    FT of the underlying function of given samples
    frequency: 1d array, frequency in hz (optional)

    """
    f = np.linspace(-fs / 2, fs / 2, len(samples), endpoint=False)
    if return_freq:
        return np.fft.fftshift(np.fft.fft(samples)) / fs * np.exp(-2j * np.pi * f * t0), f
    else:
        return np.fft.fftshift(np.fft.fft(samples)) / fs * np.exp(-2j * np.pi * f * t0)


# ND FFT and ND CFT
def fft_nd(gx, axes=None):
    """
    Conducts FFT along specified axes
    ... If axes were not given, it FFT along all axes
    ... Returns shifted FFT output
        ... FFT is a fast DFT algorithm, meaning that it does computation in Frequency space (1/x)
        instead of the Angular Frequency (wavenumber) space.
        ... It is recommended to do the change of variables at the end of the process because it becomes increasingly
        hard to keep track of the factor of 2pi as the computation becomes more complex.

    Parameters
    ----------
    gx: ND-array, Signal
    axes: array-like,  e.g.- [0, 1]
        ... axes along which FFT is conducted
        ... Default is doing FFT along all axes
        ... If you want to do 1D FFT using a 3D data, you may provide a 3D data with axes=[0] etc.
    Returns
    -------
    np.fft.fftshift(np.fft.fftn(gx, axes=axes))
    """
    dim = len(gx.shape)
    if axes is None:
        axes = list(range(dim))
    return np.fft.fftshift(np.fft.fftn(gx, axes=axes), axes=axes)


def FFTOutput2CFTOutput_nd(gf, dxs, x0s, axes=None, return_freq=True):
    """
    Approximates the DFT output into Continuous Fourier Transform output
    ... Power -> Spectral Density
    ... The input "gf" is assumed to be the shifted, raw DFT output of a ND signal.
        i.e. The variable is frequency not angular frequency (wavenumber)

    Parameters
    ----------
    gf: ND array (complex)
        ... shifted, raw FFT output
        ... It assumes the output of the function "fft_nd(signal, axes)"
    dxs: array-like
        ... Sampling intervals of the signal in the spacial/temporal domain
            i.e. (x, y, z, t) not (1/Lx, 1/Ly, 1/Lz. 1/T)
        ... Must have the same length as gf
    x0s: array-like
        ... The minimum value of the spatial/temporal domain
            i.e. X0, Y0, Z0, T0- the edge coordinates of the data and the initial moment when the data was recorded
        ... Must have the same length as gf
    axes: array-like,  e.g.- [0, 1]
        ... axes along which FFT was conducted
        ... If None, it assumes that FFT was conducted along all axes of gx
    return_freq: bool, default: True
        ... If True, it returns a list of the frequencies along the axes FFT was conducted

    Returns
    -------

    """
    if axes is None:
        n = gf.size  # number of samples
        axes = tuple(range(len(gf.shape)))
    dim = len(axes)
    if len(dxs) != dim:
        raise ValueError('FFTOutput2CFTOutput_nd: dxs must have the same length as gk')
    if len(x0s) != dim:
        raise ValueError('FFTOutput2CFTOutput_nd: x0s must have the same length as gk')

    else:
        n = 1
        for d in range(dim):
            if d in axes:
                n *= gf.shape[d]
    ns = list(gf.shape)  # number of elements along each axis

    # Sampling Frequency
    fs = [1 / dx for dx in dxs]

    freqs = []
    for i, ind in enumerate(axes):
        newshape = [1] * len(gf.shape)
        newshape[ind] = ns[ind]
        # print(ind, dxs[i], x0s[i], fs[i], newshape, gf.shape, ns[ind])
        f = np.linspace(-fs[i] / 2, fs[i] / 2, ns[ind], endpoint=False).reshape(newshape)
        gf *= np.exp(-2j * np.pi * f * x0s[i]) * 1 / fs[i]
        freqs.append(f)

    if return_freq:
        return gf, freqs
    else:
        return gf


def fourier_transform_nd(gx, dxs, x0s, axes=None, return_freq=True, return_in='freq'):
    """
    Returns a (Continuous) Fourier Transform of a given signal using FFT (DFT)
    ... This returns the adjusted values of FFT as approximated CFT of the given signal.
        If sampling frequency were sufficiently high, there is a simple relation between DFT and CFT of the signal, and
        DFT can be used to approximate CFT of the signal.
    ... Since DFT is conveniently defined using Frequency (not angular frequency), this returns the CFT as a function of frequency, f.
        If one desires to convert the spectral density as a function of omega=2pi*f, you may set return_in='wavenumber'.

    Parameters
    ----------
    gx: ND signal
    dxs: array-like
        ... Sampling intervals of the signal in the spacial/temporal domain
            i.e. (y, x, z, t) not (1/Ly, 1/Lx, 1/Lz. 1/T)
        ... The order is (dy, dx, dz)
        ... Must have the same length as gf
    x0s: array-like
        ... The minimum value of the spatial/temporal domain
            i.e. Y0, X0, Z0, T0- the edge coordinates of the data and the initial moment when the data was recorded
        ... The order is (Y0, X0, Z0)
        ... Must have the same length as gf
    axes: array-like,  e.g.- [0, 1]
        ... axes along which FFT was conducted
        ... If None, it assumes that FFT was conducted along all axes of gx
    return_freq: bool, default: True
        ... If True, it returns a list of the frequencies along the axes FFT was conducted
    return_in: str, Choose from ['freq', 'wavenumber', 'angular frequency', 'ang freq', 'k']
        ... If return_in in ['wavenumber', 'angular frequency', 'ang freq', 'k'],
            it returns the CFT as a function of wavenumber instead of frequency (1/x)
    Returns
    -------
    gf_cft: Fourier transform of the signal "gx"
    freqs: list, corresponding frequencies along the axes
    """
    dim = len(gx.shape)
    if axes is None:
        axes = tuple(range(dim))
    gf_dft = fft_nd(gx, axes=axes)
    gf_cft, freqs = FFTOutput2CFTOutput_nd(gf_dft, dxs, x0s, axes=axes, return_freq=return_freq)
    if return_in in ['wavenumber', 'angular frequency', 'ang freq', 'k']:
        freqs = [freq * 2 * np.pi for freq in freqs]
        gf_cft /= (2 * np.pi) ** (len(axes) / 2)
    if return_freq:
        return gf_cft, freqs
    else:
        return gf_cft


def convertNDto1D(ef_nd, freqs, nkout=None, mode='linear', cc=1.):
    """
    Returns a shell-to-shell contribution from ND array
    Converts ND array into 1D array (primarily used for Energy spectrum computation)

    Parameters
    ----------
    ef_nd: ND array, assuming energy density in the frequency space
    freqs: ND array, frequency grid
        ... assuming np.stack((fxx, fyy)) or np.stack((fxx, fyy, fzz))
        ... fxx is a 2D/3D grid of 1/x.
    nkout: int
        ... Number of points to be sampled in the radial direction
    mode: str, ['linear', 'log']
        ... This determines whether sampling is evenly done in the linear or log scale.
        ... For energy spectrum computation, one might prefer to get statistics at frequency/wavenumber evenly spaced in the log scale
    Returns
    -------
    fr_binned, ef1d, ef1d_err
    """

    dim = len(ef_nd.shape)

    # Radial frequency
    fr = np.zeros_like(freqs[0])
    for i in range(dim):
        fr += freqs[i] ** 2
    fr = np.sqrt(fr)

    # number of bins
    if nkout is None:
        nkout = int(max(ef_nd.shape) * 1)
    fr_binned, efnd_binned, ef1d_err = get_binned_stats(fr.flatten(), ef_nd.flatten(),
                                                        n_bins=nkout, mode=mode)
    deltaf = (max(fr_binned) - min(fr_binned)) / (nkout - 1)

    if dim == 3:
        jacobian = 4 * np.pi * fr_binned ** 2
    elif dim == 2:
        jacobian = 2 * np.pi * fr_binned
    deltaf = fr_binned[1] - fr_binned[0]
    deltaf = 1
    ef1d = efnd_binned * jacobian * cc
    ef1d_err = ef1d_err * jacobian * cc
    return fr_binned, ef1d, ef1d_err


def get_kgrids(height, width, depth=None, dx=1., dy=1., dz=1., shift=True):
    """Returns 2d/3d wavenumber grids of an array which is FFT-ed"""
    if depth is None:
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dy)
        if shift:
            kx = np.fft.fftshift(kx)
            ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi  # Convert inverse length into wavenumber
        return np.asarray([kxx, kyy])
    else:
        kx = np.fft.fftfreq(width, d=dx)
        ky = np.fft.fftfreq(height, d=dy)
        kz = np.fft.fftfreq(depth, d=dz)
        if shift:
            kx = np.fft.fftshift(kx)
            ky = np.fft.fftshift(ky)
            kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
        return np.asarray([kxx, kyy, kzz])


# padding udata
def square_udata(udata, mode='edge', **kwargs):
    """
    Pad zeros to udata to make its spatial dimensions into a square or a cube
    ... Taking a spectrum of a rectangularly shaped array could cause further aliasing. In order to combat this, one may square/cubidize
    udata before taking the spectrum by padding zeros around the obtained field.
    Parameters
    ----------
    udata
    pad_value: float, value used to pad udata

    Returns
    -------
    squared_udata_padded_with_zero
    """
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]
    height, width = udata.shape[1], udata.shape[2]
    length = max(udata.shape[1:-1])

    i, j = int((length - udata.shape[1]) / 2), int((length - udata.shape[2]) / 2)
    if dim == 3:
        depth = udata.shape[3]
        k = int((length - udata.shape[3]) / 2)
        udata_padded = np.pad(udata, (
        (0, 0), (i, length - height - i), (j, length - width - j), (k, length - depth - k), (0, 0)),
                              mode=mode, **kwargs)
    else:
        udata_padded = np.pad(udata, ((0, 0), (i, length - height - i), (j, length - width - j), (0, 0)),
                              mode=mode, **kwargs)
    return udata_padded


### DEVELOPING
def get_energy_spectrum_nd(udata,
                           x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None,
                           window=None, correct_signal_loss=True, return_in='wavenumber',
                           dealiasing=True, padding_mode='edge', padding_kwargs={}):
    """
    Returns ukdata * np.conjugate(ukdata) where ukdata is the ND-Fourier Transform of udata
    ...
    Parameters
    ----------
    udata: nd array that contains a velocity field with a shape (dim, y, x, (z), duration)
    x0: int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    x1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    y1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z0 int, default: 0
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    z1 int, default: None
        ... index to specify volume of data to load (udata[:, y0:y1, x0:x1, z0:z1])
    dx: float, spacing of the x-grid
    dy: float, spacing of the y-grid
    dz: float, spacing of the z-grid
    window: str, 
    correct_signal_loss
    return_in: str, Choose from 'wavenumber' and 'freq'
        ... wavenumber:=2pi/L, freq:=1/L

    Returns
    -------
    ef_nd, np.asarray([fxx, fyy, fzz])
        ... If return_in=''wavenumber', it returns norm of ND-CFT of velocity field as a function of wavenumber (2pi/x) instead of frequency (1/x)
    """

    if dx is None or dy is None:
        print('ERROR: dx or dy is not provided! dx is grid spacing in real space.')
        print('... k grid will be computed based on this spacing! Please provide.')
        raise ValueError
    if x0 is None:
        x0 = 0
    if x1 is None:
        x1 = udata.shape[2]
    if y0 is None:
        y0 = 0
    if y1 is None:
        y1 = udata.shape[1]

    dim = udata.shape[0]
    udata = fix_udata_shape(udata)
    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
        volume = (x1 - x0) * dx * (y1 - y0) * dy
        if dealiasing:
            udata = square_udata(udata, mode=padding_mode, **padding_kwargs)
            x0 = y0 = 0
            x1 = y1 = udata.shape[1]
            volume = (x1 - x0) * dx * (y1 - y0) * dy
        dxs = [dy, dx]

        height, width, duration = udata[0].shape
        x0s = [- height / 2 * dy, -width / 2 * dx]  # will be used to approximate CFT of a field using DFT (y0, x0)


    elif dim == 3:
        if z1 is None:
            z1 = udata.shape[2]
        if dz is None:
            print('ERROR: dz is not provided! dx is grid spacing in real space.')
            print('... k grid will be computed based on this spacing! Please provide.')
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]
        volume = (x1 - x0) * dx * (y1 - y0) * dy * (z1 - z0) * dz
        if dealiasing:
            udata = square_udata(udata, mode=padding_mode, **padding_kwargs)
            x0 = y0 = z0 = 0
            x1 = y1 = z1 = udata.shape[1]
            volume = (x1 - x0) * dx * (y1 - y0) * dy * (z1 - z0) * dz

        dxs = [dy, dx, dz]

        height, width, depth, duration = udata[0].shape
        x0s = [- height / 2 * dy, -width / 2 * dx,
               -depth / 2 * dz]  # will be used to approximate CFT of a field using DFT (y0, x0, z0)
    # WINDOWING
    duration = udata.shape[-1]
    if window is not None or window != 'rectangle':
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata)
            for i in range(dim):
                udata_tapered[i, ...] = udata[i, ...] * windows
            # udata_tapered = udata * windows

        # PERFORM DFT on the windowed field
        ufdata, freqs = fourier_transform_nd(udata_tapered, dxs, x0s, axes=list(range(1, dim + 1)))

        energy, energy_tapered = get_energy(udata), get_energy(udata_tapered)
        signal_intensity_losses = np.nanmean(energy_tapered, axis=tuple(range(dim))) / np.nanmean(energy, axis=tuple(
            range(dim)))
    else:
        ufdata, freqs = fourier_transform_nd(udata, dxs, x0s, axes=list(range(1, dim + 1)))
        signal_intensity_losses = np.ones(duration)

    ##################################################
    # compute E(\vec{k})
    ##################################################
    ef_nd = np.zeros(ufdata[0].shape)

    for i in range(dim):
        ef_nd[...] += np.abs(ufdata[i, ...]) ** 2
    ef_nd /= 2. * volume  # Energy spectrum is defined via Energy DENSITY. Divide the array by volume

    if correct_signal_loss:
        for t in range(duration):
            # print signal_intensity_losses[t]
            ef_nd[..., t] = ef_nd[..., t] / signal_intensity_losses[t]
    if return_in != 'freq':
        # CHANGE OF VARIABLES FROM FREQ TO ANG FREQ (WAVENUMBER)
        for t in range(duration):
            ef_nd[..., t] = ef_nd[..., t] / (2 * np.pi) ** dim
        freqs = [freq * 2 * np.pi for freq in freqs]
    if dim == 2:
        fxx, fyy = np.meshgrid(freqs[1], freqs[0])
        return ef_nd, np.asarray([fxx, fyy])
    elif dim == 3:
        fxx, fyy, fzz = np.meshgrid(freqs[1], freqs[0], freqs[2])
        return ef_nd, np.asarray([fxx, fyy, fzz])


def get_energy_spectrum(udata, x0=0, x1=None, y0=0, y1=None,
                        z0=0, z1=None, dx=None, dy=None, dz=None, nkout=None,
                        window=None, correct_signal_loss=True, remove_undersampled_region=True,
                        cc=1, notebook=True, mode='linear',
                        dealiasing=True, padding_mode='edge', padding_kwargs={},
                        debug=False):
    """
    Returns an energy spectrum from velocity field data
    ... The algorithm implemented in this function is VERY QUICK because it does not use the two-point autorcorrelation tensor.
    ... Instead, it converts u(kx, ky, kz)u*(kx, ky, kz) into u(kr)u*(kr). (here * dentoes the complex conjugate)
    ... CAUTION: Must provide udata with aspect ratio ~ 1
    ...... The conversion process induces unnecessary error IF the dimension of u(kx, ky, kz) is skewed.
    ...... i.e. Make udata.shape like (800, 800), (1024, 1024), (512, 512) for accurate results.
    ... KNOWN ISSUES:
    ...... This function returns a bad result for udata with shape like (800, 800, 2)


    Parameters
    ----------
    udata: nd array
    epsilon: nd array or float, default: None
        dissipation rate used for scaling energy spectrum
        If not given, it uses the values estimated using the rate-of-strain tensor
    nu: flaot, viscosity
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str, a name of the windowing function
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.
    cc: float, default: 1.75
        A numerical factor to compensate for the signal loss due to approximations.
        ... cc=1.75 was obtained from the JHTD data.
    Returns
    -------
    e_k: numpy array
        Energy spectrum with shape (number of data points, duration)
    e_k_err: numpy array
        Energy spectrum error with shape (number of data points, duration)
    kk: numpy array
        Wavenumber with shape (number of data points, duration)

    """

    def convert_nd_spec_to_1d(e_ks, ks, nkout=nkout, cc=cc, mode='linear'):
        dim, duration = len(ks.shape), e_ks.shape[-1]
        if nkout is None:
            nkout = int(max(ks[0].shape) * 0.8)
        shape = (nkout, duration)
        e_k, e_k_err, kk = np.empty(shape), np.empty(shape), np.empty(nkout)
        for t in range(duration):
            kk, e_k[:, t], e_k_err[:, t] = convertNDto1D(e_ks[..., t], ks, nkout=nkout, mode=mode, cc=cc)
        return e_k, e_k_err, kk

    e_ks, ks = get_energy_spectrum_nd(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz,
                                      window=window, correct_signal_loss=correct_signal_loss, return_in='wavenumber',
                                      dealiasing=dealiasing, padding_mode=padding_mode, padding_kwargs=padding_kwargs)
    e_k, e_k_err, kk = convert_nd_spec_to_1d(e_ks, ks, nkout=nkout, cc=cc, mode=mode)

    if debug:
        udata = fix_udata_shape(udata)
        print("Check identity k = \int_0^\infty E(k)dk at t0=0")
        print("LHS =", np.trapz(e_k[:, 0], kk))
        print("(RHS (riemann sum), std )= ",
              get_spatial_avg_energy(udata[..., 0:1], x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1))

    return e_k, e_k_err, kk


def get_enstrophy_spectrum_nd(udata, sigma=None,
                              x0=0, x1=None, y0=0, y1=None,
                              z0=0, z1=None, xx=None, yy=None, zz=None,
                              window=None, correct_signal_loss=True, return_in='wavenumber',
                              dealiasing=True, padding_mode='edge', padding_kwargs={}):
    """
    Compute enstrophy spectrum of a 2/3D data.
        ... omegak * np.conjugate(omegak)
    Parameters
    ----------
    udata: nd array, velocity field (dim, height, width, (depth), duration)
    sigma: float, Standard deviation for Gaussian kernel
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    window: str, a name of the windowing function
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.
    return_in: str, default: 'wavenumber'
        ... the returning power will be in the units of 
             'wavenumber':(1/s^2) * (rad/m)^3
             'frequency': (1/s^2) * (1/m)^3 
        ... 'wavenumber': 2pi / L
        ... 'frequency': 1 / L
    dealiasing: if True, it pads the data with zeros to make the array square to mitigate aliasing effects..
    padding_mode: str, default: 'edge'
        ... arguments are passed to np.pad(array, pad_width, padding_mode, **kwargs)
    padding_kwargs:
        ... keyword arguments for np.pad()

    Returns
    -------
    ef_nd: nd array, enstrophy power density(dim, ky, kx, kz, duration)
        ... If return_in=''wavenumber', it returns norm of ND-CFT of velocity field as a function of wavenumber (2pi/x) instead of frequency (1/x)
    OPTIONAL:
    ky, kx, kz: nd array, wavenumber(ky, kx, kz) = 2pi/L
       or
    fy, fx, fz: nd array, frequency(fy, fx, fz) = 1/L
    """

    if xx is None or yy is None:
        print('ERROR: xx or yy is not provided! xx is a 2d/3dgrid')
        print(
            '... To compute curl, one should always provide a grid. spacing may not be sufficient as vorticity is a pseudo-vector.')
        print('... k grid will be computed based on the positional grid')
        raise ValueError
    if x0 is None:
        x0 = 0
    if x1 is None:
        x1 = udata.shape[2]
    if y0 is None:
        y0 = 0
    if y1 is None:
        y1 = udata.shape[1]

    dim = udata.shape[0]
    udata = fix_udata_shape(udata)
    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
        dx, dy = get_grid_spacing(xx, yy)
        volume = (x1 - x0) * dx * (y1 - y0) * dy
        if dealiasing:
            udata = square_udata(udata, mode=padding_mode, **padding_kwargs)
            x0 = y0 = 0
            x1 = y1 = udata.shape[1]
            volume = (x1 - x0) * dx * (y1 - y0) * dy  # Should the volume be the squared/cubic volume?
        dxs = [dy, dx]

        height, width, duration = udata[0].shape
        x0s = [- height / 2 * dy, -width / 2 * dx]  # will be used to approximate CFT of a field using DFT (y0, x0)

        omega = curl(udata, xx=xx, yy=yy)

    elif dim == 3:
        if z1 is None:
            z1 = udata.shape[2]
        if zz is None:
            print('ERROR: zz is not provided! dx is grid spacing in real space.')
            print('... k grid will be computed based on this spacing! Please provide.')
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        volume = (x1 - x0) * dx * (y1 - y0) * dy * (z1 - z0) * dz
        if dealiasing:
            udata = square_udata(udata, mode=padding_mode, **padding_kwargs)
            x0 = y0 = z0 = 0
            x1 = y1 = z1 = udata.shape[1]
            volume = (x1 - x0) * dx * (y1 - y0) * dy * (
                        z1 - z0) * dz  # Should the volume be the squared/cubic volume? This is a choice

        dxs = [dy, dx, dz]

        height, width, depth, duration = udata[0].shape
        x0s = [- height / 2 * dy, -width / 2 * dx,
               -depth / 2 * dz]  # will be used to approximate CFT of a field using DFT (y0, x0, z0)

        omega = curl(udata, xx=xx, yy=yy, zz=zz)

    # Gaussian blur
    if sigma is not None:
        if dim == 2:
            omega = gaussian_blur_scalar_field(omega, sigma=sigma)
        elif dim == 3:
            omega = gaussian_blur_vector_field(omega, sigma=sigma)

    # WINDOWING
    duration = udata.shape[-1]
    if window is not None:
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            omega_tapered = omega[...] * windows
            enst, enst_tapered = omega ** 2, omega_tapered ** 2

        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            omega_tapered = np.empty_like(omega)
            enst, enst_tapered = np.zeros(omega.shape[1:]), np.zeros(omega.shape[1:])
            for i in range(dim):
                omega_tapered[i, ...] = omega[i, ...] * windows
                enst += omega[i, ...] ** 2
                enst_tapered += omega_tapered[i, ...] ** 2

        # PERFORM DFT on the windowed field
        if dim == 2:
            omegafdata, freqs = fourier_transform_nd(omega_tapered, dxs, x0s, axes=list(range(0, dim)))
        elif dim == 3:
            omegafdata, freqs = fourier_transform_nd(omega_tapered, dxs, x0s, axes=list(range(1, dim + 1)))
        signal_intensity_losses = np.nanmean(enst_tapered, axis=tuple(range(dim))) / np.nanmean(enst, axis=tuple(
            range(dim)))
    else:
        if dim == 2:
            omegafdata, freqs = fourier_transform_nd(omega, dxs, x0s, axes=list(range(0, dim)))
        elif dim == 3:
            omegafdata, freqs = fourier_transform_nd(omega, dxs, x0s, axes=list(range(1, dim + 1)))
        signal_intensity_losses = np.ones(duration)

    ##################################################
    # compute P(\vec{k})- enstrophy spectrum
    ##################################################

    if dim == 2:
        enstf_nd = np.abs(omegafdata[...]) ** 2
    elif dim == 3:
        enstf_nd = np.zeros(udata.shape[1:])
        for i in range(dim):
            enstf_nd[...] += np.abs(omegafdata[i, ...]) ** 2
    enstf_nd /= volume  # Enstrophy spectrum is defined via Energy DENSITY. Divide a whole thing by volume

    if correct_signal_loss:
        for t in range(duration):
            # print signal_intensity_losses[t]
            enstf_nd[..., t] = enstf_nd[..., t] / signal_intensity_losses[t]
    if return_in == 'wavenumber':
        # CHANGE OF VARIABLES FROM FREQ TO ANG FREQ (WAVENUMBER)
        for t in range(duration):
            enstf_nd[..., t] = enstf_nd[..., t] / (2 * np.pi) ** dim
        freqs = [freq * 2 * np.pi for freq in freqs]
    if dim == 2:
        fxx, fyy = np.meshgrid(freqs[1], freqs[0])
        fs = np.asarray([fxx, fyy])

    elif dim == 3:
        fxx, fyy, fzz = np.meshgrid(freqs[1], freqs[0], freqs[2])
        fs = np.asarray([fxx, fyy, fzz])
    return enstf_nd, fs


def get_enstrophy_spectrum(udata, sigma=None,
                           x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, xx=None, yy=None, zz=None, nkout=None,
                           window=None, correct_signal_loss=True,
                           cc=1, notebook=True, mode='linear',
                           dealiasing=True, padding_mode='edge', padding_kwargs={},
                           debug=False):
    """
    Compute the enstrophy spectrum from a velocity field (udata)

    ... CAUTION: Must provide udata with aspect ratio ~ 1
    ...... The conversion process induces unnecessary error IF the dimension of u(kx, ky, kz) is skewed.
    ...... i.e. Make udata.shape like (800, 800), (1024, 1024), (512, 512) for accurate results.
    ... KNOWN ISSUES:
    ...... This function returns a bad result for a highly skewed udata with shape like (dim, 800, 800, 2, duration)

    Parameters
    ----------
    udata: ndarray, a velocity field
    sigma: float, the standard deviation of the Gaussian kernel
    x0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    x1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    y1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t0: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    t1: int
        index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
    dx: float
        spacing in x
    dy: float
        spacing in y
    dz: float
        spacing in z
    nkout: int, default: None
        number of bins to compute energy/dissipation spectrum
    notebook: bool, default: True
        Use tqdm.tqdm_notebook if True. Use tqdm.tqdm otherwise
    window: str, a name of the windowing function
        Windowing reduces undesirable effects due to the discreteness of the data.
        A wideband window such as 'flattop' is recommended for turbulent energy spectra.

        For the type of applying window function, choose from below:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)
    correct_signal_loss: bool, default: True
        If True, it would compensate for the loss of the signals due to windowing.
        Always recommended to obtain accurate spectral densities.
    cc: float, default: 1
        ... an arbitrary constant to scale the enstrophy spectrum
        ... this is used to mimic 1D enstrophy spectrum from a 3D spectrum.
    xx: ndarray, x-coordinates of the data, used to compute vorticity (Passng xx/yy/zz is recommended rather than passing dx, dy, dz)
    yy: ndarray, y-coordinates of the data, used to compute vorticity (Passng xx/yy/zz is recommended rather than passing dx, dy, dz)
    zz: ndarray, z-coordinates of the data, used to compute vorticity (Passng xx/yy/zz is recommended rather than passing dx, dy, dz)
    notebook
    mode: str, default: 'linear'
        ... 'linear' or 'log'
        ... sampling of the energy spectrum will be conducted with bins that are linearly/logarithmically spaced
    dealiasing: if True, it pads the data with zeros to make the array square to mitigate aliasing effects..
    padding_mode: str, default: 'edge'
        ... arguments are passed to np.pad(array, pad_width, padding_mode, **kwargs)
    padding_kwargs:
        ... keyword arguments for np.pad()
    debug: bool, default: False
        ... If True, it prints out the result of a sanity check based on the Parseval's theorem.
        ... It checks identity at t=0: <omega_i omega_i> = \int_0^\infty P(k)dk at t0=0

    Returns
    -------
    enst_k: 2d array, 3D enstrophy spectrum
        ... (nkout, duration)
    enst_k_err: 2d array, standard error of enstrophy spectrum
        ... (nkout, duration)
    kk: 1d array
        ... k-values of the enstrophy spectrum
    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    def convert_nd_spec_to_1d(enst_ks, ks, nkout=nkout, cc=cc, mode='linear'):
        dim, duration = len(ks.shape), enst_ks.shape[-1]
        if nkout is None:
            nkout = int(max(ks[0].shape) * 0.8)
        shape = (nkout, duration)
        enst_k, enst_k_err, kk = np.empty(shape), np.empty(shape), np.empty(nkout)
        for t in tqdm(range(duration)):
            kk, enst_k[:, t], enst_k_err[:, t] = convertNDto1D(enst_ks[..., t], ks, nkout=nkout, mode=mode, cc=cc)
        return enst_k, enst_k_err, kk

    enst_ks, ks = get_enstrophy_spectrum_nd(udata, sigma=sigma, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                            xx=xx, yy=yy, zz=zz,
                                            window=window, correct_signal_loss=correct_signal_loss,
                                            return_in='wavenumber',
                                            dealiasing=dealiasing, padding_mode=padding_mode,
                                            padding_kwargs=padding_kwargs )

    enst_k, enst_k_err, kk = convert_nd_spec_to_1d(enst_ks, ks, nkout=nkout, cc=cc, mode=mode)

    if debug:
        udata = fix_udata_shape(udata)
        avg_enst = get_spatial_avg_enstrophy(udata[..., 0:1], xx=xx, yy=yy, zz=zz,
                                             x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)[0]
        print("Check identity at t=0: <omega_i omega_i> = \int_0^\infty P(k)dk at t0=0")
        print("LHS =", avg_enst)
        print("(RHS (riemann sum), std )= ", np.trapz(enst_k[..., 0], kk))
    if notebook:
        from tqdm import tqdm as tqdm

    return enst_k, enst_k_err, kk


def get_confidence_levels_on_energy_spectrum(iw_wrt_eta, keta, alpha_min=0.1, slope=1.,
                                             alpha_max=1.0, simple=True):
    """
    Returns the alpha values which reflects the confidence levels of the energy spectrum obtained from
    a velocity field generated by PIV experiments
    ... Returned array can be used as alpha values. To plot values with varying alpha, use plt.scatter or
    graph.plot_with_varying_alphas. The latter allows to draw lines with different alpha values.

    ... path/to/module_dir/reference_data/error_functions_of_ek_args_iweta_keta.pkl
    contains an function which gives LOGARITHMIC ERROR of the spectrum (log10[Observed E(k) / True E(k)])
    with arguments (interrogation window size / eta, keta)

    Parameters
    ----------
    iw_wrt_eta: float
        ... interrogation window size with respect to Kolmogorov scale
            i.e. (interrogation window size) / (Kolmogorov scale)
        ... here we assume that the interrogation window was a square.
    keta: float
        ... dimensionless waveumber of the energy spectrum (k times Kolmogorov scale)
    simple: bool
        ... If True, the returning alphas will be
            [alpha_max, ..., alpha_max, alpha_min, ..., alpha_min, ]; alpha changes at keta = keta_c =2.*np.pi / (iw_wrt_eta * 2.)
    Returns
    -------
    alphas: list of alpha values

    """
    if simple:
        keta_c = 2. * np.pi / (iw_wrt_eta * 2.)
        alphas = np.ones_like(keta) * alpha_max
        alphas[keta > keta_c] = alpha_min
    else:
        # Get an error function about energy spectrum function
        dpath = os.path.join(os.path.join(moddirpath, 'reference_data'), 'error_functions_of_ek_args_iweta_keta.pkl')
        logErrFunction = read_pickle(dpath)
        logErr = logErrFunction(iw_wrt_eta, keta)
        alphas = 1 - slope * np.abs(logErr)
        alphas[alphas > 1] = 1.0
        alphas[alphas < alpha_min] = alpha_min
        alphas[np.isnan(alphas)] = alpha_min
    return alphas


def get_confidence_levels_on_structure_function(iw_wrt_eta, reta, alpha_min=0.1, slope=2.,
                                                alpha_max=1.0, simple=True):
    """
    Returns the alpha values which reflects the confidence levels of the second order longitudinal structure function
    obtained from a velocity field generated by PIV experiments
    ... Returned array can be used as alpha values. To plot values with varying alpha, use plt.scatter or
    graph.plot_with_varying_alphas. The latter allows to draw lines with different alpha values.

    ... path/to/module_dir/reference_data/error_functions_of_dll_args_iweta_keta.pkl
    contains an function which gives a SIGNED RELATIVE ERROR of the spectrum (log10[Observed E(k) / True E(k)])
    with arguments (interrogation window size / eta, keta)
    ... You should non-dimensionalize the distance r and the two-pt  correlation function by the most plausible dissipation rate!
        You have two choices in experiments.
        ... 1. Compute epsilon from the rate-of-strain tensor
            ... However, low resolution PIV could lead to poor estimation of epsilon
            2. Compute epsilon such that the structure function exhibits a plateau at y=2.1
            ... The dissipation rate estimated by this method is more robust than the first method. (It also has its limitaions)

    Parameters
    ----------
    iw_wrt_eta: float
        ... interrogation window size with respect to Kolmogorov scale
            i.e. (interrogation window size) / (Kolmogorov scale)
        ... here we assume that the interrogation window was a square.
    reta: float
        ... dimensionless distance of the second-order structure function (r / Kolmogorov scale)

    Returns
    -------
    alphas: list of alpha values
    """
    if simple:
        keta_c = iw_wrt_eta
        alphas = np.ones_like(reta) * alpha_max
        alphas[reta < keta_c] = alpha_min
    else:
        # Get an error function about the second-order longitudinal structure function
        dpath = os.path.join(os.path.join(moddirpath, 'reference_data'), 'error_functions_of_dll_args_iweta_reta.pkl')
        signedRelErrFunction = read_pickle(dpath)
        signedRelErr = signedRelErrFunction(iw_wrt_eta, reta)
        alphas = 1 - slope * np.abs(signedRelErr)
        alphas[alphas > 1] = 1.0
        alphas[alphas < alpha_min] = alpha_min
        alphas[np.isnan(alphas)] = alpha_min
    return alphas


# def get_1d_energy_spectrum(udata, k='kx', x0=0, x1=None, y0=0, y1=None,
#                            z0=0, z1=None, dx=None, dy=None, dz=None,
#                            window=None, correct_signal_loss=True, debug=False):
#     """
#     Returns 1D energy spectrum from velocity field data
#
#     Parameters
#     ----------
#     udata: nd array
#     k: str, default: 'kx'
#         string to specify the direction along which the given velocity field is Fourier-transformed
#     x0: int
#         index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
#     x1: int
#         index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
#     y0: int
#         index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
#     y1: int
#         index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
#     t0: int
#         index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
#     t1: int
#         index to specify a portion of data in which autocorrelation funciton is computed. Use data u[y0:y1, x0:x1, t0:t1].
#     dx: float
#         spacing in x
#     dy: float
#         spacing in y
#     dz: float
#         spacing in z
#     window: str
#         Windowing reduces undesirable effects due to the discreteness of the data.
#         A wideband window such as 'flattop' is recommended for turbulent energy spectra.
#
#         For the type of available window function, choose from below:
#         boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
#         kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
#         slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
#         tukey (needs taper fraction)
#     correct_signal_loss: bool
#         If True, it would compensate for the loss of the signals due to windowing.
#         Always recommended to obtain accurate spectral densities.
#
#     Returns
#     -------
#     eiis: numpy array
#         eiis[0] = E11, eiis[1] = E22
#         ... 1D energy spectra with argument k="k" (kx by default)
#     eii_errs: numpy array:
#         eiis[0] = E11_error, eiis[1] = E22_error
#     k: 1d numpy array
#         Wavenumber with shape (number of data points, )
#         ... Unlike get_energy_spectrum(...), this method NEVER outputs the wavenumber array with shape (number of data points, duration)
#     """
#     if x0 is None:
#         x0 = 0
#     if y0 is None:
#         y0 = 0
#     if x1 is None:
#         x1 = udata[0].shape[1]
#     if y1 is None:
#         y1 = udata[0].shape[0]
#
#     udata = fix_udata_shape(udata)
#     dim, duration = len(udata), udata.shape[-1]
#     if dim == 2:
#         ux, uy = udata[0, y0:y1, x0:x1, :], udata[1, y0:y1, x0:x1, :]
#         height, width, duration = ux.shape
#         udata_tmp = udata[:, y0:y1, x0:x1, :]
#     elif dim == 3:
#         ux, uy, uz = udata[0, y0:y1, x0:x1, z0:z1, :], udata[1, y0:y1, x0:x1, z0:z1, :], udata[2, y0:y1, x0:x1, z0:z1,:]
#         height, width, depth, duration = ux.shape
#         udata_tmp = udata[:, y0:y1, x0:x1, z0:z1, :]
#     else:
#         raise ValueError('... Error: Invalid dimension is given. Use 2 or 3 for the number of spatial dimensions. ')
#
#
#     if k == 'kx':
#         ax_ind = 1  # axis number to take 1D DFT
#         n = width
#         d = dx
#         if dim == 2:
#             ax_ind_for_avg = 0  # axis number(s) to take statistics (along y)
#         elif dim == 3:
#             ax_ind_for_avg = (0, 2)  # axis number(s) to take statistics  (along y and z)
#     elif k == 'ky':
#         ax_ind = 0  # axis number to take 1D DFT
#         n = height
#         d = dy
#         if dim == 2:
#             ax_ind_for_avg = 1  # axis number(s) to take statistics  (along x)
#         elif dim == 3:
#             ax_ind_for_avg = (1, 2)  # axis number(s) to take statistics  (along x and z)
#     elif k == 'kz':
#         ax_ind = 2  # axis number to take 1D DFT
#         n = depth
#         d = dz
#         ax_ind_for_avg = (0, 1)  # axis number(s) along which statistics is computed  (along x and y)
#     freq = np.fft.fftshift(np.fft.fftfreq(n, d=d))
#     deltaf = freq[1] - freq[0]
#     n_samples = ux.shape[ax_ind]
#
#     # Apply a hamming window to get lean FFT spectrum for aperiodic signals
#     if window is not None:
#         duration = udata.shape[-1]
#         if dim == 2:
#             xx, yy = get_equally_spaced_grid(udata, spacing=dx)
#             windows = get_window_radial(xx, yy, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1)
#             udata_tapered = np.empty_like(udata_tmp)
#             for i in range(dim):
#                 udata_tapered[i, ...] = udata_tmp[i, ...] * windows
#             ux, uy = udata_tapered
#         elif dim == 3:
#             xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
#             windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0,
#                                         z1=z1)
#             udata_tapered = np.empty_like(udata_tmp)
#             for i in range(dim):
#                 udata_tapered[i, ...] = udata_tmp[i, ...] * windows
#             ux, uy, uz = udata_tapered
#
#     if correct_signal_loss:
#         if window is not None:
#             if dim == 2:
#                 xx, yy = get_equally_spaced_grid(udata, spacing=dx)
#                 window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
#                 signal_intensity_loss = np.nanmean(window_arr)
#             elif dim == 3:
#                 xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
#                 window_arr = get_window_radial(xx, yy, wtype=window, x0=x0, x1=x1, y0=y0, y1=y1)
#                 signal_intensity_loss = np.nanmean(window_arr)
#         else:
#             signal_intensity_loss = 1.
#     print('deltaf, nsamples, d', deltaf, n_samples, d)
#     print('dx, 1/deltaf/n', d, 1/deltaf/n_samples)
#     # E11
#     ux_f = fft_nd(ux, axes=ax_ind)
#     FFTOutput2CFTOutput_nd(ux_f,  [d], [], axes=ax_ind, return_freq='wavenumber')
#
#     ux_k = np.fft.fftshift(np.fft.fft(ux, axis=ax_ind))
#
#     print('Parseval', np.nansum(ux**2)/(np.nansum(np.abs(ux_k * np.conj(ux_k)) * 2) / n_samples))
#     ux_k /= 2 * np.pi * deltaf * n_samples# convert to spectral density (Power per wavenumber)
#     e11_nd = np.abs(ux_k * np.conj(ux_k)) * 2 # e11 is defined as twice as the square of the 1D FT of u1
#     e11 = np.nanmean(e11_nd, axis=ax_ind_for_avg)
#     e11_err = np.nanstd(e11_nd, axis=ax_ind_for_avg)
#
#     # E22
#     uy_k = np.fft.fftshift(np.fft.fft(uy, axis=ax_ind))
#     uy_k /= 2 * np.pi * deltaf * n_samples # convert to spectral density
#     e22_nd = np.abs(uy_k * np.conj(uy_k)) * 2 # e22 is defined as twice as the square of the 1D FT of u2
#     e22 = np.nanmean(e22_nd, axis=ax_ind_for_avg)
#     e22_err = np.nanstd(e22_nd, axis=ax_ind_for_avg)
#
#
#     # Get an array for wavenumber
#     k = np.fft.fftfreq(n, d=d) * 2 * np.pi  # shape=(n, duration)
#     k = np.fft.fftshift(k)
#     deltak = k[1] - k[0]
#     if dim == 3:
#         # E33
#         uz_k = np.fft.fftshift(np.fft.fft(uz, axis=ax_ind))
#         uz_k /= 2 * np.pi *deltaf * n_samples  # convert to spectral density
#         e33_nd = np.abs(uz_k * np.conj(uz_k)) * 2 # e33 is defined as twice as the square of the 1D FT of u3
#         e33 = np.nanmean(e33_nd, axis=ax_ind_for_avg)
#         e33_err = np.nanstd(e33_nd, axis=ax_ind_for_avg)
#
#         # Must divide by 2pi because np.fft.fft() performs in the frequency space (NOT angular frequency space)
#         eiis, eii_errs = np.array([e11, e22, e33]), np.array([e11_err, e22_err, e33_err])
#     elif dim == 2:
#         # Must divide by 2pi^2 because np.fft.fft() performs in the frequency space (NOT angular frequency space)
#         eiis, eii_errs = np.array([e11, e22]), np.array([e11_err, e22_err])
#     else:
#         raise ValueError('... 1d spectrum: Check the dimension of udata! It must be 2 or 3!')
#
#
#     # Windowing causes the loss of the signal (energy.)
#     # ... This compensates for the loss.
#     if correct_signal_loss:
#         for i in range(dim):
#             eiis[i] /= signal_intensity_loss
#             eii_errs[i] /= signal_intensity_loss
#     if debug:
#         print('get_1d_energy_spectrum(): debug is set True. It will check the property \int_0^\infty Eii = 2 <ui ui>' )
#         for i in range(dim):
#             ui2_tavg = np.nanmean(udata[i, ...]**2, axis=tuple(range(dim)))[0]
#             k_i, eiis_i = clean_data_interp1d(eiis[i, ..., 0], k)
#             integral_i = np.trapz(eiis_i[..., 0], x=k_i[:, 0], axis=0)
#             print('...Frame0: <u%d squared> / integral of E_%d%d: ' % (i+1, i+1, i+1), ui2_tavg / integral_i )
#     return eiis, eii_errs, k

# Pressure- a simple poisson solver based on Euler's equation
def get_pressure(udata, xx, yy, zz, nu=1.003, notebook=True):
    """
    Returns pressure from a 3D+1 udata
    ... This is a poisson solver of an incompressible NS equation using FFT
    ... The solvalbility condition is F(p(x))|(kx=0, ky=0, kz=0) = pk(kx=0, ky=0, kz=0) = 0
    ...... The zeroth term of the Fourier decomposition of the pressure field is zero.

    Parameters
    ----------
    udata: 4d/5d array, v-field data with shape (3, h, w, d) or (3, h, w, d, t)
    xx: 3d array, positional grid about x
    yy: 3d array, positional grid about y
    zz: 3d array, positional grid about z
    nu: float, viscosity

    Returns
    -------
    pressure, 4d array with shape (h, w, d, t)

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    xmin, xmax, ymin, ymax, zmin, zmax = np.nanmin(xx), np.nanmax(xx), np.nanmin(yy), np.nanmax(yy), np.nanmin(
        zz), np.nanmax(zz)
    L_x, L_y, L_z = xmax - xmin, ymax - ymin, zmax - zmin
    _, nb_y, nb_x, nb_z, duration = udata.shape

    kx_list = np.arange(nb_x)
    tmp_mask = (kx_list > nb_x / 2.0)
    kx_list[tmp_mask] = kx_list[tmp_mask] - nb_x
    kx_list = kx_list * (2.0 * np.pi) / L_x

    ky_list = np.arange(nb_y)
    tmp_mask = (ky_list > nb_y / 2.0)
    ky_list[tmp_mask] = ky_list[tmp_mask] - nb_y
    ky_list = ky_list * (2.0 * np.pi) / L_y

    kz_list = np.arange(nb_z)
    tmp_mask = (kz_list > nb_z / 2.0)
    kz_list[tmp_mask] = kz_list[tmp_mask] - nb_z
    kz_list = kz_list * (2.0 * np.pi) / L_z

    #     kz_mesh, ky_mesh, kx_mesh = np.meshgrid(kz_list, ky_list, kx_list, indexing='ij')
    kx_mesh, ky_mesh, kz_mesh = np.meshgrid(kx_list, ky_list, kz_list)

    # initialization
    pressure = np.empty(udata.shape[1:])

    for t in tqdm(range(duration)):
        u_vec, v_vec, w_vec = udata[..., t]

        # duu_dx_hat = 1.0j*kx_mesh*np.fft.fftn(u_vec*u_vec)
        # duu_dx_hat = 1.0j*kx_mesh*np.fft.fftn(u_vec*u_vec)
        # duu_dx_hat = 1.0j*kx_mesh*np.fft.fftn(u_vec*u_vec)

        uu_hat = np.fft.fftn(u_vec * u_vec)
        uv_hat = np.fft.fftn(u_vec * v_vec)
        uw_hat = np.fft.fftn(u_vec * w_vec)
        vv_hat = np.fft.fftn(v_vec * v_vec)
        vw_hat = np.fft.fftn(v_vec * w_vec)
        ww_hat = np.fft.fftn(w_vec * w_vec)

        R_c_x_hat = - 1.0j * kx_mesh * uu_hat - 1.0j * ky_mesh * uv_hat - 1.0j * kz_mesh * uw_hat
        R_c_y_hat = - 1.0j * kx_mesh * uv_hat - 1.0j * ky_mesh * vv_hat - 1.0j * kz_mesh * vw_hat
        R_c_z_hat = - 1.0j * kx_mesh * uw_hat - 1.0j * ky_mesh * vw_hat - 1.0j * kz_mesh * ww_hat

        # du_dxdx_hat = -1.0*nu*kx_mesh*kx_mesh*np.fft.fftn(u_vec)

        R_v_x_hat = (-1.0 * nu * kx_mesh * kx_mesh - 1.0 * nu * ky_mesh * ky_mesh - 1.0 * nu * kz_mesh * kz_mesh) * np.fft.fftn(u_vec)
        R_v_y_hat = (-1.0 * nu * kx_mesh * kx_mesh - 1.0 * nu * ky_mesh * ky_mesh - 1.0 * nu * kz_mesh * kz_mesh) * np.fft.fftn(v_vec)
        R_v_z_hat = (-1.0 * nu * kx_mesh * kx_mesh - 1.0 * nu * ky_mesh * ky_mesh - 1.0 * nu * kz_mesh * kz_mesh) * np.fft.fftn(w_vec)
        RHS_hat = 1.0j * kx_mesh * (R_c_x_hat + R_v_x_hat) + 1.0j * ky_mesh * (
                R_c_y_hat + R_v_y_hat) + 1.0j * kz_mesh * (R_c_z_hat + R_v_z_hat)

        p_hat = RHS_hat / (-1.0 * kx_mesh * kx_mesh - 1.0 * ky_mesh * ky_mesh - 1.0 * kz_mesh * kz_mesh)
        p_hat[0, 0, 0] = 0.0  # solvability condition
        pressure[..., t] = np.real(np.fft.ifftn(p_hat))
    return pressure


# Hydrodynamic impulse and angular impulse
def get_impulse_density(udata, xx, yy, zz=None, rho=1e-3, crop=2):
    """
    It returns the density of hydrodynamic impulse
    hydrodynamic impulse is defined as
    ... P = 0.5 [ \int r \times \omega dV + \int r \times (n \times u) dS]
          = (Bulk term) + (Surface Term)
        ... It is best understood when this form is differentiated
            dP/dt = (bulk force like gravity) + (surface force: f_i = sigma_ij n_j where sigma is viscous stress tensor)

    Parameters
    ----------
    udata: nd array, a 3D velocity field (3, h, w, d, (t: optional))
    xx: 3d grid, x-coordinate of the udata
    yy: 3d grid, y-coordinate of the udata
    zz: 3d grid, z-coordinate of the udata
    rho: density of fluid, default: 0.001 (g/mm^3 for water)
    crop: int, greater or equal to 0
        ... Sets the values of omega near the edge to np.nan

    Returns
    -------
    hi: 5d array: hydrodynamic impulse density (3, y, x, z, t)

    """
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]
    hi = np.zeros_like(udata)
    # surface_terms = {}

    if zz is None:
        dx, dy = get_grid_spacing(xx, yy)
        omega = curl(udata, dx=dx, dy=dy, xx=xx, yy=yy)

        for t in range(duration):
            hi[0, ..., t] = yy * omega[..., t]
            hi[1, ..., t] = -xx * omega[..., t]

        bulk = np.zeros_like(hi).astype('bool')
        bulk[:, crop:-crop, crop:-crop, :] = True
        edges = ~bulk
        hi[edges] = np.nan
    else:
        rdata = np.stack((xx, yy, zz))
        rdata = np.repeat(rdata[..., np.newaxis], duration, axis=-1)
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        omega = curl(udata, dx=dx, dy=dy, dz=dz, xx=xx, yy=yy, zz=zz)
        hi = np.cross(rdata, omega, axis=0)
        bulk = np.zeros_like(hi).astype('bool')
        bulk[:, crop:-crop, crop:-crop, crop:-crop, :] = True
        edges = ~bulk
        hi[edges] = np.nan
    # # compute surface terms?
    # jacobian_diag = get_jacobian_xyz_ijk(xx, yy, zz=zz)

    # if zz is None:
    #     uz = np.zeros_like(udata[0, ...])
    #     zz = np.zeros_like(xx)
    #     rdata =  np.stack((xx, yy, zz))
    #     rdata = np.repeat(rdata[..., np.newaxis], duration, axis=-1)
    #     udata = np.stack((udata[0, ...], udata[1, ...], uz))
    #
    #     #surface1 # Right edge
    #     n1 = np.asarray([1, 0, 0]) * jacobian_diag[0]
    #     n1 = np.repeat(n1[..., np.newaxis], duration, axis=-1)
    #
    #     s1 = np.cross(rdata[:, :, -1, :], np.cross(n1, udata[:, :, -1, :], axis=0), axis=0)[:2, ...] * rho / 2
    #
    #     #surface2 # Left edge
    #     n2 = np.asarray([-1, 0, 0]) * jacobian_diag[0]
    #     n2 = np.repeat(n2[..., np.newaxis], duration, axis=-1)
    #     s2 = np.cross(rdata[:, :, 0, :], np.cross(n2, udata[:, :, 0, :], axis=0), axis=0)[:2, ...] * rho / 2
    #
    #     #surface3 # Bottom edge
    #     n3 = np.asarray([0, 1, 0]) * jacobian_diag[1]
    #     n3 = np.repeat(n3[..., np.newaxis], duration, axis=-1)
    #     s3 = np.cross(rdata[:, -1, :, :], np.cross(n3, udata[:, -1, :, :], axis=0), axis=0)[:2, ...] * rho / 2
    #
    #     #surface4 # Top edge
    #     n4 = np.asarray([0, -1, 0]) * jacobian_diag[1]
    #     n4 = np.repeat(n4[..., np.newaxis], duration, axis=-1)
    #     s4 = np.cross(rdata[:, 0, :, :], np.cross(n4, udata[:, 0, :, :], axis=0), axis=0)[:2, ...] * rho / 2
    #
    #     surface_terms["right"] = {"rdata": rdata[:, :, -1, 0],
    #                               "integrand": s1,
    #                               "dS": dy,
    #                               "n": n1}
    #     surface_terms["left"] = {"rdata": rdata[:, :, 0, 0],
    #                               "integrand": s2,
    #                               "dS": dy,
    #                               "n": n2}
    #     surface_terms["top"] = {"rdata": rdata[:, -1:, :, 0],
    #                               "integrand": s3,
    #                               "dS": dx,
    #                               "n": n3}
    #     surface_terms["bottom"] = {"rdata": rdata[:, 0, :, 0],
    #                               "integrand": s4,
    #                               "dS": dx,
    #                               "n": n4}
    # else:
    #     rdata = np.stack((xx, yy, zz))
    #     rdata = np.repeat(rdata[..., np.newaxis], duration, axis=-1)
    #
    #     # surface1 # Right Plane
    #     n1 = np.asarray([1, 0, 0]) * jacobian_diag[0]
    #     n1 = np.repeat(n1[..., np.newaxis], duration, axis=-1)
    #
    #     s1 = np.cross(rdata[:, :, -1, :, :], np.cross(n1, udata[:, :, -1, :, :], axis=0), axis=0) * rho / 2
    #
    #     # surface2 # Left Plane
    #     n2 = np.asarray([-1, 0, 0]) * jacobian_diag[0]
    #     n2 = np.repeat(n2[..., np.newaxis], duration, axis=-1)
    #     s2 = np.cross(rdata[:, :, 0, :, :], np.cross(n2, udata[:, :, 0, :, :], axis=0), axis=0) * rho / 2
    #
    #     # surface3 # Bottom Plane
    #     n3 = np.asarray([0, 1, 0]) * jacobian_diag[1]
    #     n3 = np.repeat(n3[..., np.newaxis], duration, axis=-1)
    #     s3 = np.cross(rdata[:, -1, :, :, :], np.cross(n3, udata[:, -1, :, :, :], axis=0), axis=0) * rho / 2
    #
    #     # surface4 # Top Plane
    #     n4 = np.asarray([0, -1, 0]) * jacobian_diag[1]
    #     n4 = np.repeat(n4[..., np.newaxis], duration, axis=-1)
    #     s4 = np.cross(rdata[:, 0, :, :, :], np.cross(n4, udata[:, 0, :, :, :], axis=0), axis=0) * rho / 2
    #
    #     # surface5 # Forward Plane
    #     n5 = np.asarray([0, 0, 1]) * jacobian_diag[2]
    #     n5 = np.repeat(n5[..., np.newaxis], duration, axis=-1)
    #     s5 = np.cross(rdata[:, :, :, -1, :], np.cross(n5, udata[:, :, :, -1, :], axis=0), axis=0) * rho / 2
    #
    #     # surface6 # Backward Plane
    #     n6 = np.asarray([0, 0, -1]) * jacobian_diag[2]
    #     n6 = np.repeat(n6[..., np.newaxis], duration, axis=-1)
    #     s6 = np.cross(rdata[:, :, :, 0, :], np.cross(n6, udata[:, :, :, 0, :], axis=0), axis=0) * rho / 2
    #     surface_terms["right"] = {"rdata": rdata[:, :, -1, :, 0],
    #                               "integrand": s1,
    #                               "dS": dy*dz,
    #                               "n": n1}
    #     surface_terms["left"] = {"rdata": rdata[:, :, 0, :, 0],
    #                              "integrand": s2,
    #                              "dS": dy*dz,
    #                              "n": n2}
    #     surface_terms["top"] = {"rdata": rdata[:, -1:, :, :, 0],
    #                             "integrand": s3,
    #                             "dS": dx*dz,
    #                             "n": n3}
    #     surface_terms["bottom"] = {"rdata": rdata[:, 0, :, :, 0],
    #                                "integrand": s4,
    #                                "dS": dx*dz,
    #                                "n": n4}
    #     surface_terms["forward"] = {"rdata": rdata[:, :, :, -1, 0],
    #                             "integrand": s5,
    #                             "dS": dx*dy,
    #                             "n": n5}
    #     surface_terms["backward"] = {"rdata": rdata[:, :, :, 0, 0],
    #                                "integrand": s6,
    #                                "dS": dx*dy,
    #                                "n": n6}
    hi *= rho
    return hi


def get_impulse(udata, xx, yy, zz=None, rho=1e-3, crop=2,
                keep=None, R=None, xc=0, yc=0, zc=0):
    """
    Returns the hydrodynamic impulse inside the region defined by xx, yy, zz
    ... Consider the momentum contained in a simply connected domain G.
        (total fluid momentum in G) = (hydrodynamic impulse in G) + (surface terms)
        ... Saffman "Vortex Dynamics" Ch. 3.3 Eq 13
    ... it returns an array of a shape (duration, )

    Parameters
    ----------
    udata: 4d/5d array, v-field data with shape (3, h, w, d) or (3, h, w, d, t)
    xx: 3d array, positional grid about x
    yy: 3d array, positional grid about y
    zz: 3d array, positional grid about z
    rho: fluid density, default 1e-3 g/mm3
    crop: int, crop the domain near the edge to remove the boundary effects
    keep: boolean array, indices of the points to keep
    R: float, if given, it gives the impulse inside a sphere of radius R centered at (xc, yc, zc)
    xc, yc, zc: float, if given, it gives the impulse inside a sphere of radius R centered at (xc, yc, zc)

    Returns
    -------
    hydrodynamic impulse, (optional:bulk term, surface term): 2d array (components, duration)
    """
    hi_density = get_impulse_density(udata, xx, yy, zz=zz, rho=rho, crop=crop)
    duration = hi_density.shape[-1]

    if keep is not None:
        for t in range(duration):
            if zz is None:
                for d in range(2):
                    hi_density[d, ~keep, t] = np.nan
            else:
                for d in range(3):
                    hi_density[d, ~keep, t] = np.nan
        num_valid = np.sum(keep)  # number of valid points per frame
    elif keep is None and R is not None:
        if zz is None:
            rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        else:
            rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2 + (zz - zc) ** 2)
        keep = rr < R

        for t in range(duration):
            if zz is None:
                for d in range(2):
                    hi_density[d, ~keep, t] = np.nan
            else:
                for d in range(3):
                    hi_density[d, ~keep, t] = np.nan
        num_valid = np.sum(keep)  # number of valid points
    else:
        if zz is None:
            num_valid = hi_density[:, :, 0].size
        else:
            num_valid = hi_density[0, :, :, :, 0].size

    if zz is None:
        dx, dy = get_grid_spacing(xx, yy)
        hi = np.nanmean(hi_density, axis=(1, 2)) * dx * dy * num_valid
    else:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        hi = np.nanmean(hi_density, axis=(1, 2, 3)) * dx * dy * dz * num_valid
    return hi


# Hydrodynamic impulse and angular impulse
def get_angular_impulse_density(udata, xx, yy, zz=None, rho=1e-3, crop=2):
    """
    It returns the density of (hydrodynamic) angular impulse

    ... It is the moment of the impulsive force which generates the motion from the rest
    ... Angular impulse is defined as
        A = 1/3 \int r \times (r \times \omega) dV (Lamb, 1932, Sect. 152, Saffman. Sect. 3.5, Eq.5)
          = - 1/2 \int r^2 omega dV - 1/6 r^2 r \times (omega \cdot n) dS (Batchelor, 1967, Sect. 7.2)
          =  \int r \times u dV  + 1/2 \int r^2 (n \times  u) dS (Saffman. Sect. 3.5, Eq.7)

        ... If all the vorticity is contained inside the control volume,
            A = \int r \times u dV

    Parameters
    ----------
    udata: 4d/5d array, v-field data with shape (3, h, w, d) or (3, h, w, d, t)
    xx: 3d array, positional grid about x
    yy: 3d array, positional grid about y
    zz: 3d array, positional grid about z
    rho: fluid density, default 1e-3 g/mm3
    crop: int, crop the domain near the edge to remove the boundary effects

    Returns
    -------
    ai: nd array, angular impulse density
    """
    udata = fix_udata_shape(udata)
    dim, duration = udata.shape[0], udata.shape[-1]
    surface_terms = {}

    if zz is None:
        ai = np.empty_like(udata[0, ...])
        dx, dy = get_grid_spacing(xx, yy)
        omega = curl(udata, dx=dx, dy=dy, xx=xx, yy=yy)

        for t in range(duration):
            ai[..., t] = -(xx ** 2 + yy ** 2) * omega[..., t] * rho / 3.
        bulk = np.zeros_like(ai).astype('bool')
        bulk[crop:-crop, crop:-crop, :] = True
        edges = ~bulk
        ai[edges] = np.nan
    else:
        ai = np.empty_like(udata)
        rdata = np.stack((xx, yy, zz))
        rdata = np.repeat(rdata[..., np.newaxis], duration, axis=-1)
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        omega = curl(udata, dx=dx, dy=dy, dz=dz, xx=xx, yy=yy, zz=zz)

        ai = np.cross(rdata, np.cross(rdata, omega, axis=0), axis=0) * rho / 3.

        bulk = np.zeros_like(ai).astype('bool')
        bulk[:, crop:-crop, crop:-crop, crop:-crop, :] = True
        edges = ~bulk
        ai[edges] = np.nan
    return ai


def get_angular_impulse(udata, xx, yy, zz=None, rho=1e-3, crop=2,
                        keep=None, R=None, xc=0, yc=0, zc=0):
    """
    Returns the total angular impulse inside the region defined by xx, yy, zz
    ... it returns an array of a shape (duration, )
    ... If return_bulk_surface is True, it returns hydrodynamic impulse, its bulk term, and surface term

    Parameters
    ----------
    udata: 4d/5d array, v-field data with shape (3, h, w, d) or (3, h, w, d, t)
    xx: 3d array, positional grid about x
    yy: 3d array, positional grid about y
    zz: 3d array, positional grid about z
    rho: fluid density, default 1e-3 g/mm3
    crop: int, crop the domain near the edge to remove the boundary effects
    keep: boolean array, indices of the points to keep
    R: float, if given, it gives the impulse inside a sphere of radius R centered at (xc, yc, zc)
    xc, yc, zc: float, if given, it gives the impulse inside a sphere of radius R centered at (xc, yc, zc)

    Returns
    -------
    hydrodynamic angular impulse: 2d array (components, duration)
    """
    ai_density = get_angular_impulse_density(udata, xx, yy, zz=zz, rho=rho, crop=crop)
    duration = ai_density.shape[-1]

    if keep is not None:
        for t in range(duration):
            if zz is None:
                ai_density[~keep, t] = np.nan
            else:
                for d in range(3):
                    ai_density[d, ~keep, t] = np.nan
        num_valid = np.sum(keep)  # number of valid points per frame
    elif keep is None and R is not None:
        if zz is None:
            rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        else:
            rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2 + (zz - zc) ** 2)
        keep = rr < R

        for t in range(duration):
            if zz is None:
                ai_density[~keep, t] = np.nan
            else:
                for d in range(3):
                    ai_density[d, ~keep, t] = np.nan
        num_valid = np.sum(keep)  # number of valid points
    else:
        if zz is None:
            num_valid = ai_density[:, :, 0].size
        else:
            num_valid = ai_density[0, :, :, :, 0].size

    if zz is None:
        dx, dy = get_grid_spacing(xx, yy)
        # for 2d, ai_density has a shape like (height, width, duration)
        ai = np.nanmean(ai_density, axis=(0, 1)) * dx * dy * num_valid
    else:
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        # for 2d, ai_density has a shape like (components, height, width, depth duration)
        ai = np.nanmean(ai_density, axis=(1, 2, 3)) * dx * dy * dz * num_valid
    return ai


def get_helicity_density(udata, xx, yy, zz, crop=2):
    """
    Returns helicity density u cdot omega
    ...
    Parameters
    ----------
    udata: (4d or 5d) array, 3d velocity field (components(ux, uy, uz), x, y, z, t)
    xx: 3d array
    yy: 3d array
    zz: 3d array
    crop: int, >0 or None
        ... it replaces the values near the edges to np.nan because omega often behaves ill near the edges
        ... helicity_density[crop:-crop, crop:-crop, crop:-crop, :] = np.nan[crop:-crop, crop:-crop, crop:-crop, :] = np.nan

    Returns
    -------
    helicity_density: 4d array (x, y, z, t)
    """

    udata = fix_udata_shape(udata)
    duration = udata.shape[-1]
    omega = curl(udata, xx=xx, yy=yy, zz=zz)

    helicity_density = np.zeros_like(udata[0, ...])
    for t in range(duration):
        for d in range(3):
            helicity_density[..., t] += udata[d, ..., t] * omega[d, ..., t]

    if crop is not None:
        bulk = np.zeros_like(helicity_density).astype('bool')
        bulk[crop:-crop, crop:-crop, crop:-crop, :] = True
        edges = ~bulk
        helicity_density[edges] = np.nan
    return helicity_density


def get_helicity(udata, xx, yy, zz, crop=2):
    """
    (For 3D velocity field)
    Returns a scalar field such that u dot omega

    Parameters
    ----------
    udata: 4d/5d array, v-field data with shape (3, h, w, d) or (3, h, w, d, t)
    xx: 3d array
    yy: 3d array
    zz: 3d array
    crop: int, >0 or None
        ... it replaces the values near the edges to np.nan because omega often behaves ill near the edges
        ... helicity_density[crop:-crop, crop:-crop, crop:-crop, :] = np.nan[crop:-crop, crop:-crop, crop:-crop, :] = np.nan

    Returns
    -------
    helicity: 1d array, shape (duration.)

    """
    helicity_density = get_helicity_density(udata, xx, yy, zz, crop=crop)
    dx, dy, dz = get_grid_spacing(xx, yy, zz)
    helicity = np.nanmean(helicity_density, axis=(0, 1, 2)) * xx.size * dx * dy * dz

    return helicity


def get_azimuthal_average(ef_nd, freqs, nkout=None, statistic='mean', mode='linear', cc=1.):
    """
    Computes an azimuthal average of an nd array ef_nd
    ...

    Parameters
    ----------
    ef_nd: nd array
    freqs: a list/tuple of nd array
        ... r = \sqrt(freqs[0]**2 + freqs[1]**2 + ... + freqs[-1]**2)
    nkout: number of bins used to take statistsics
    mode: str, choose from 'linear' or 'log'
        ... this determines the width of the bins to be evenly spaced in a linear space or logarithmic space
        ... i.e. bin edges will be like 0, 1, 2, 3, 4, ... for the linear mode
                 bin edges will be like 10^0, 10^1, 10^2, 10^3, ... for the logarithmic mode
    cc: constant to be multiplied to the output

    Returns
    -------
    fr_binned, ef1d, ef1d_err: radius r, azimuthal average of e(r, theta, phi), standard deviation as a function of r
    """
    dim = len(ef_nd.shape)

    # Radial frequency
    fr = np.zeros_like(freqs[0])
    for i in range(dim):
        fr += freqs[i] ** 2
    fr = np.sqrt(fr)

    # number of bins
    if nkout is None:
        nkout = int(max(ef_nd.shape) * 1)
    fr_binned, efnd_binned, ef1d_err = get_binned_stats(fr.flatten(), ef_nd.flatten(),
                                                        n_bins=nkout, mode=mode, statistic=statistic)
    ef1d = efnd_binned * cc
    ef1d_err = ef1d_err * cc
    return fr_binned, ef1d, ef1d_err


# Vector calculus with udata format
def dot(udata, vdata):
    """
    Returns a dot poduct of udata with vdata

    Parameters
    ----------
    udata: nd array with shape (dim, height, width, (depth), (duration))
    vdata: nd array with shape (dim, height, width, (depth), (duration))

    Returns
    -------
    scaData: nd array with (height, width, (depth), (duration))
    """
    udim, vdim = udata.shape[0], vdata.shape[0]
    if udim != vdim:
        raise ValueError('udata and vdata must have the same number of components')
    else:
        if len(udata.shape) < len(vdata.shape):
            udata, vdata = vdata, udata  # udata always a greater dimension
        if udata.shape == vdata.shape:  # preferred
            scaData = np.zeros_like(udata[0, ...])
            for d in range(udim):
                scaData += udata[d, ...] * vdata[d, ...]
            return scaData
        else:
            if udata.shape[:-1] != vdata.shape:
                raise ValueError(
                    'udata must have the same shape as vdata OR udata[..., 0] must have the shape as vdata')
            else:
                scaData = np.zeros_like(udata[0, ...])
                for d in range(udim):
                    for t in range(udata.shape[-1]):
                        scaData[..., t] += udata[d, ..., t] * vdata[d, ...]
                return scaData


def cross(udata, vdata):
    """
    Returns an cross poduct of udata with vdata

    Parameters
    ----------
    udata: nd array with shape (dim, height, width, (depth), (duration))
    vdata: nd array with shape (dim, height, width, (depth), (duration))

    Returns
    -------
    vecData: nd array with (dim, height, width, (depth), (duration)) if dim==3 OR (height, width, (depth), (duration)) if dim==2
    """
    udim, vdim = udata.shape[0], vdata.shape[0]
    if udim != vdim:
        raise ValueError('udata and vdata must have the same number of components')
    else:
        if len(udata.shape) < len(vdata.shape):
            udata, vdata = vdata, -udata  # udata always a greater dimension; flipping the order of u and v changes the sign
        if udim == 3:
            if udata.shape == vdata.shape:  # preferred (udata and vdata are both time-series with the same shape)
                vecData = np.empty_like(udata[...])
                vecData[0, ...] = udata[1, ...] * vdata[2, ...] - udata[2, ...] * vdata[1, ...]
                vecData[1, ...] = udata[2, ...] * vdata[0, ...] - udata[0, ...] * vdata[2, ...]
                vecData[2, ...] = udata[0, ...] * vdata[1, ...] - udata[1, ...] * vdata[0, ...]
                return vecData
            else:
                if udata.shape[:-1] != vdata.shape:
                    raise ValueError(
                        'udata must have the same shape as vdata OR udata[..., 0] must have the shape as vdata')
                else:
                    vecData = np.empty_like(udata[...])
                    for d in range(udim):
                        for t in range(udata.shape[-1]):
                            vecData[0, ..., t] = udata[1, ..., t] * vdata[2, ...] - udata[2, ..., t] * vdata[1, ...]
                            vecData[1, ..., t] = udata[2, ..., t] * vdata[0, ...] - udata[0, ..., t] * vdata[2, ...]
                            vecData[2, ..., t] = udata[0, ..., t] * vdata[1, ...] - udata[1, ..., t] * vdata[0, ...]
                    return vecData
        elif udim == 2:
            if udata.shape == vdata.shape:  # preferred (udata and vdata are both time-series with the same shape)
                vecData = udata[0, ...] * vdata[1, ...] - udata[1, ...] * vdata[0, ...]
                return vecData
            else:
                if udata.shape[:-1] != vdata.shape:
                    raise ValueError(
                        'udata must have the same shape as vdata OR udata[..., 0] must have the shape as vdata')
                else:
                    vecData = np.empty_like(udata[0, ...])
                    for d in range(udim):
                        for t in range(udata.shape[-1]):
                            vecData[..., t] = udata[0, ..., t] * vdata[1, ...] - udata[1, ..., t] * vdata[0, ...]
                    return vecData


def mag(udata):
    """
    Returns a norm of of udata

    Parameters
    ----------
    udata: nd array with a shape (dim, height, width, (depth), (duration))

    Returns
    -------
    umag: np.array with shape () or (duration, )
    """
    umag = np.zeros_like(udata[0, ...])
    dim = udata.shape[0]
    for d in range(dim):
        umag += udata[d, ...] ** 2
    umag = np.sqrt(umag)
    return umag


def norm(udata):
    """
    Returns a normalized udata (a vector stored at every location is normalized)

    Parameters
    ----------
    udata: nd array with a shape (dim, height, width, (depth), (duration))

    Returns
    -------
    normalized udata: : nd array with the same shape as udata
    """
    umag = mag(udata)
    return udata / umag


# FLUX ANALYSIS
def compute_energy_flux(udata, xx, yy, zz, xc, yc, zc, rho=0.000997, n=50, ntheta=100, nphi=100, flux_density=False):
    """
    Returns the energy flux (\int (Energy current density) dS in nW
    ... If flux_density is True, it returns the energy flux per unit area in nW / mm^2
    ... Units guide: [udata], [xx] = L/T, L.
        The dimension of the energy flux is [e_flux] = [rho * udata**3 * dS] = ML^2/T^3 = [W=J/s]
        ... This represents the amount of energy flow through the surface per unit time.
    ... If flux_density is True, this returns the energy flux density- simply energy flux divided by the surface area
        The dimension of the energy flux DENSITY is [e_flux density] = M/T^3 = [W] / L^2
        ... This represents the amount of energy flow through the surface per unit time.
    ... Sample case: [udata], [xx] = mm/s, mm
        ... The dimension of energy flux: ML^2/T^3 = g mm^2/s^3 = nW
        ... The dimension of energy flux density: nW / mm^2

    Parameters:
    _____________________
    n: number of gaussian surfaces considered
    """
    #     rho = 0.000997 #g/mm3
    udata = fix_udata_shape(udata)
    duration = udata.shape[-1]

    # Generate interpolating functions for ux, uy, uz
    y, x, z = yy[:, 0, 0], xx[0, :, 0], zz[0, 0, :]

    dx, dy, dz = get_grid_spacing(xx, yy, zz)
    l, w, d = np.max(yy) - np.min(yy), np.max(xx) - np.min(xx), np.max(zz) - np.min(zz)
    rmax = np.sqrt(l ** 2 + w ** 2 + d ** 2) / 2.
    rs = np.linspace(dx * 1, rmax, n)
    # xyz coordinates on the spherical surface with radius r and origin at (xc, yc, zc)
    theta = np.linspace(0, np.pi, ntheta)
    phi = np.linspace(0, 2 * np.pi, nphi)

    # INITIALIZATION
    e_flux = np.empty((len(rs), duration))
    for t in range(duration):
        f_ux = RegularGridInterpolator((y, x, z), udata[0, ..., t])  # ux interpolating function
        f_uy = RegularGridInterpolator((y, x, z), udata[1, ..., t])  # uy interpolating function
        f_uz = RegularGridInterpolator((y, x, z), udata[2, ..., t])  # uz interpolating function
        energy = get_energy(udata)[..., t]  # inpainted energy field
        f_e = RegularGridInterpolator((y, x, z), energy)  # energy interpolating function

        for j, r_float in enumerate(rs):
            r = np.asarray([r_float])
            # now make a 3D grid for radial distance, polar angle, and azimuthal angle
            ttheta, rr, pphi = np.meshgrid(theta, r, phi)  # shape [len(r), len(theta), len(phi)]
            x_, y_, z_ = sph2cart(rr, ttheta, pphi, xc=xc, yc=yc, zc=zc)

            try:
                ux_at_r = f_ux((y_, x_, z_))
                uy_at_r = f_uy((y_, x_, z_))
                uz_at_r = f_uz((y_, x_, z_))
                udata_at_r = np.stack((ux_at_r, uy_at_r, uz_at_r))

                # compute energy at r
                energy_at_r = get_energy(udata_at_r)
                #             energy_at_r = f_e((y_, x_, z_))

                # Get unit area vectors for each area element
                RR = np.stack((x_ - xc, y_ - yc, z_ - zc))
                RR_norm = np.sqrt((x_ - xc) ** 2 + (y_ - yc) ** 2 + (z_ - zc) ** 2)
                nx, ny, nz = (x_ - xc) / RR_norm, (y_ - yc) / RR_norm, (z_ - zc) / RR_norm
                nhat = np.stack((nx, ny, nz))

                # Now compute energy flux: flux psi_s = \int J_s \cdot \hat{n} dA
                dtheta, dphi = theta[1] - theta[0], phi[1] - phi[0]
                dA = r[0] ** 2 * np.sin(ttheta) * dtheta * dphi
                # manual cleaning
                #             energy_at_r = clean_data(energy_at_r, cutoff=1*10**5)
                e_flux_density_at_r = rho * energy_at_r * np.nansum(udata_at_r * nhat,
                                                                    axis=0)  # energy current = energy * velocity
                e_flux_at_r = np.nanmean(e_flux_density_at_r * dA) * e_flux_density_at_r.size
                if flux_density:
                    A = np.sum(dA)  # surface area of the gaussian sphere (should be 4pi*r^2)
                else:
                    A = 1.
                e_flux[j, t] = e_flux_at_r / A
            except ValueError:
                e_flux[j, t] = np.nan
                #             print('... a problem occured probably in ux_at_r, uy_at_r, or uz_at_r (r=%f)' % r_float)
                continue

    #         A = np.sum(dA) # surface area of the gaussian sphere (should be 4pi*r^2)
    return e_flux, rs


def compute_net_energy_current(udata, xx, yy, zz, xc, yc, zc, rho=0.000997, flux_density=False, maxr=None):
    """
    Returns the integral of energy flux (Net energy current inside the Gaussian sphere)
        Net energy current = \int_0^{R} (energy current density) \cdot dS dr    
    """
    eflux, rs = compute_energy_flux(udata, xx, yy, zz, xc, yc, zc, rho=rho, flux_density=False)
    duration = eflux.shape[-1]
    #     net_e_current = np.empty(duration)
    if maxr is not None:
        ind, _ = find_nearest(rs, maxr)
    else:
        ind = None
    eflux = clean_data(eflux, method='fill', fill_value=0)
    net_e_current = np.trapz(eflux[:ind, :], rs[:ind], axis=0)
    return net_e_current


def compute_mass_flux(udata, xx, yy, zz, xc, yc, zc, rho=0.000997, ntheta=100, nphi=100, flux_density=False):
    """
    Returns the energy flux (\int (mass current density) dS in M/T
    ... If flux_density is True, it returns the energy flux per unit area in M/(L^2 T)
    ... Units guide: [udata], [xx] = L/T, L.
        The dimension of the energy flux is [mass_flux] = [rho * udata * dS] = M/T
        ... This represents the amount of energy flow through the surface per unit time.
    ... If flux_density is True, this returns the energy flux density- simply energy flux divided by the surface area
        The dimension of the energy flux DENSITY is [mass_flux density] = M/(L^2 T)
        ... This represents the amount of energy flow through the surface per unit time.
    ... Sample case: [udata], [xx] = mm/s, mm
        ... The dimension of energy flux: M/T = g /s
        ... The dimension of energy flux density: M/(L^2 T)
    """
    #     rho = 0.000997 #g/mm3
    udata = fix_udata_shape(udata)
    duration = udata.shape[-1]

    # Generate interpolating functions for ux, uy, uz
    y, x, z = yy[:, 0, 0], xx[0, :, 0], zz[0, 0, :]

    dx, dy, dz = get_grid_spacing(xx, yy, zz)
    l, w, d = np.max(yy) - np.min(yy), np.max(xx) - np.min(xx), np.max(zz) - np.min(zz)
    rmax = np.sqrt(l ** 2 + w ** 2 + d ** 2) / 2.
    rs = np.linspace(dx * 1, rmax)
    # xyz coordinates on the spherical surface with radius r and origin at (xc, yc, zc)
    theta = np.linspace(0, np.pi, ntheta)
    phi = np.linspace(0, 2 * np.pi, nphi)

    # INITIALIZATION
    mass_flux = np.empty((len(rs), duration))
    for t in range(duration):
        for j, r_float in enumerate(rs):
            r = np.asarray([r_float])
            # now make a 3D grid for radial distance, polar angle, and azimuthal angle
            ttheta, rr, pphi = np.meshgrid(theta, r, phi)  # shape [len(r), len(theta), len(phi)]
            x_, y_, z_ = sph2cart(rr, ttheta, pphi, xc=xc, yc=yc, zc=zc)
            f_ux = RegularGridInterpolator((y, x, z), udata[0, ..., t])  # ux interpolating function
            f_uy = RegularGridInterpolator((y, x, z), udata[1, ..., t])  # uy interpolating function
            f_uz = RegularGridInterpolator((y, x, z), udata[2, ..., t])  # uz interpolating function

            try:
                ux_at_r = f_ux((y_, x_, z_))
                uy_at_r = f_uy((y_, x_, z_))
                uz_at_r = f_uz((y_, x_, z_))
                udata_at_r = np.stack((ux_at_r, uy_at_r, uz_at_r))

                #             energy_at_r = f_e((y_, x_, z_))

                # Get unit area vectors for each area element
                RR = np.stack((x_ - xc, y_ - yc, z_ - zc))
                RR_norm = np.sqrt((x_ - xc) ** 2 + (y_ - yc) ** 2 + (z_ - zc) ** 2)
                nx, ny, nz = (x_ - xc) / RR_norm, (y_ - yc) / RR_norm, (z_ - zc) / RR_norm
                nhat = np.stack((nx, ny, nz))

                # Now compute energy flux: flux psi_s = \int J_s \cdot \hat{n} dA
                dtheta, dphi = theta[1] - theta[0], phi[1] - phi[0]
                dA = r[0] ** 2 * np.sin(ttheta) * dtheta * dphi
                # manual cleaning
                #             energy_at_r = clean_data(energy_at_r, cutoff=1*10**5)
                mass_flux_density_at_r = rho * np.nansum(udata_at_r * nhat,
                                                         axis=0)  # energy current = energy * velocity
                mass_flux_at_r = np.nanmean(mass_flux_density_at_r * dA) * mass_flux_density_at_r.size
                if flux_density:
                    A = np.sum(dA)  # surface area of the gaussian sphere (should be 4pi*r^2)
                else:
                    A = 1.
                mass_flux[j, t] = mass_flux_at_r / A
            except ValueError:
                mass_flux[j, t] = np.nan
                #             print('... a problem occured probably in ux_at_r, uy_at_r, or uz_at_r (r=%f)' % r_float)
                continue

    #         A = np.sum(dA) # surface area of the gaussian sphere (should be 4pi*r^2)
    return mass_flux, rs


def compute_energy_flux_from_path(udatapath, xc, yc, zc, rho=0.000997, flux_density=False, maxr=None,
                                  x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                  t0=0, t1=None, inc=1, clean=True, cutoff=np.inf, verbose=False):
    if t1 is None:
        t1 = get_udata_dim(udatapath)[-1]

    e_flux = np.empty(len(range(t0, t1, inc)))
    for i, t in enumerate(tqdm(range(t0, t1, inc), desc='computing energy current: time')):
        udata, xx, yy, zz = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                                t0=t, t1=t + 1, inc=inc, return_xy=True, reverse_y=True,
                                                verbose=verbose)
        udata_i = clean_udata(udata, cutoff=cutoff, verbose=False, showtqdm=verbose)
        e_flux[i] = compute_energy_flux(udata_i, xx, yy, zz, xc, yc, zc, rho=rho)[0]
    return e_flux


def compute_energy_current_from_path(udatapath, xc, yc, zc, rho=0.000997, flux_density=False, maxr=None,
                                     x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                     t0=0, t1=None, inc=1, clean=True, cutoff=np.inf, verbose=False):
    if t1 is None:
        t1 = get_udata_dim(udatapath)[-1]

    net_e_current = np.empty(len(range(t0, t1, inc)))
    for i, t in enumerate(tqdm(range(t0, t1, inc), desc='computing energy current: time')):
        udata, xx, yy, zz = get_udata_from_path(udatapath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1,
                                                t0=t, t1=t + 1, inc=inc, return_xy=True, reverse_y=True,
                                                verbose=verbose)
        udata_i = clean_udata(udata, cutoff=cutoff, verbose=False, showtqdm=verbose)
        net_e_current[i] = compute_net_energy_current(udata_i, xx, yy, zz, xc, yc, zc, rho=rho, maxr=maxr)[0]
    return net_e_current


# RW HELPER
# pickle
def read_pickle(filename):
    """
    A wrapper to read a pickle file
    ... There was a change in the default encoding type during some version update of the pickle package
    ... This wrapper attempts to read a pickle file with different approaches.

    Parameters
    ----------
    filename: str, path to the pickle

    Returns
    -------
    obj: object stored in the given path

    """
    with open(filename, "rb") as pickle_in:
        try:
            obj = pickle.load(pickle_in)
        except UnicodeDecodeError:
            try:
                obj = pickle.load(pickle_in, encoding="bytes")
            except:
                # hmmm... something failed, Try pandas' reading funciton
                import pandas
                obj = pandas.read_pickle(filename)
    return obj


def read_data_from_h5(h5path, keys, return_dict=False, grpname=None, verbose=True):
    """
    Grabs data in a simply organized h5 file
    ... Return the data stored at /keys[0], /keys[1], ...

    Parameters
    ----------
    h5path: str, a path to a h5 file
    keys: list, a list of keys
        if

    Returns
    -------
    data_read: list of data stroed in a h5 file- [data0, data1, data2, ...]
    or
    datadict: dictionary of data- {name0: data0, name1: data1, name2: data2, ...}
    """
    if h5path[-3:] != '.h5':
        h5path += '.h5'
    if keys == 'all':
        keys = get_h5_keys(h5path)
    if not return_dict:
        data_read = []
        with h5py.File(h5path, mode='r') as f:
            for key in keys:
                try:
                    if grpname is None:
                        val = f[key][...]
                    elif grpname in f.keys():
                        val = f[grpname][key][...]
                    else:
                        if verbose:
                            print('... /%s does not exist in the given h5' % grpname)
                        sys.exit()
                    data_read.append(val)
                except:
                    if verbose:
                        print('read_data_from_h5: %s does not exist in %s' % (key, h5path))
                    data_read.append(None)
        return data_read
    else:
        datadict = {}
        with h5py.File(h5path, mode='r') as f:
            for key in keys:
                try:
                    if grpname is None:
                        val = f[key][...]
                    elif grpname in f.keys():
                        val = f[grpname][key][...]
                    else:
                        if verbose:
                            print('... /%s does not exist in the given h5' % grpname)
                        sys.exit()
                    datadict[key] = val
                except:
                    if verbose:
                        print('read_data_from_h5: %s does not exist in %s' % (key, h5path))
                    datadict[key] = None
        return datadict


def remove_data_from_h5(h5path, keys, ):
    """Removes datasets from a h5 file"""
    with h5py.File(h5path, mode='a') as f:
        for key in keys:
            del f[key]


def merge_simple_hdf5s(path2masterh5, paths2h5s2add, overwrite=False):
    """
    This function merges the datasets in H5FILES to H5MASTER
    ... datasets in H5FILES must be located under the top directory (/DATASET1, /DATASET2, ...)
    ... If H5MASTER already contains the dataset(s), this function does not overwrite the existing data unless specified.

    Parameters
    ----------
    path2masterh5: str, a path to the master h5 file (where the data in paths2h5s2add will be added)
    paths2h5s2add: list of str, paths where the data to be added to the master h5 are stored.
    overwrite: bool, default: False. If True, the data in path2masterh5 will be overwritten by the data in paths2h5s2add if it exists.

    Returns
    -------
    None

    """
    hfMaster = h5py.File(path2masterh5, mode='a')
    master_keys = hfMaster.keys()
    for i, path2add in enumerate(paths2h5s2add):
        hfExtra = h5py.File(path2add, mode='r')
        keys = hfExtra.keys()

        for key in keys:
            if not key in master_keys or overwrite:
                print('... adding {0} from {1} to {2}'.format(key, path2add, path2masterh5))
                if overwrite:
                    print('......overwriting {0}'.format(key))
                    del hfMaster[key]
                h5py.h5o.copy(hfExtra.id, bytes(key, encoding="utf-8"), hfMaster.id, bytes(key, encoding="utf-8"))
        hfExtra.close()
    hfMaster.close()
    print('... merging complete')


def sort_n_arrays_using_order_of_first_array(list_of_arrays, element_dtype=tuple):
    """
    Sort a list of N arrays by the order of the first array in the list
    e.g. a=[2,1,3], b=[1,9,8], c=['a', 'b', 'c']
        [a, b, c] -> [(1, 2, 3), (9, 1, 8), ('b', 'a', 'c')]

    Parameters
    ----------
    list_of_arrays: a list of lists/1D-arrays
    element_dtype: data type, default: tuple
        ... This argument specifies the data type of the elements in the returned list
        ... The default data type of the element is tuple because this functon utilizes sorted(zip(...))
        ... E.g. element_dtype=np.ndarray
                -> [a, b, c] -> [np.array([1, 2, 3]),
                                 np.array([9, 1, 8],
                                 np.array(['b', 'a', 'c'], dtype='<U1']

    Returns
    -------
    list_of_sorted_arrays: list of sorted lists/1D arrays
    """

    list_of_sorted_arrays = list(zip(*sorted(zip(*list_of_arrays))))
    if element_dtype == list:
        list_of_sorted_arrays = [list(a) for a in list_of_sorted_arrays]
    elif element_dtype == np.ndarray:
        list_of_sorted_arrays = [np.asarray(a) for a in list_of_sorted_arrays]
    return list_of_sorted_arrays


# turbulence project helpers
def compute_ring_radius_from_master_curve(l, dp=160., do=25.6, N=8, lowerGamma=2.2, setting='medium'):
    """
    Given a stroke length, this function returns a radius of a vortex ring in a vortex ring collider
    ... well tested over time
    ... units: mm
    e.g.-
        r = compute_ring_radius_from_master_curve(8.2, setting='small')
        v = compute_ring_velocity_from_master_curve(8.2, estimate_veff(8.2, 200), setting='small')

    Parameters
    ----------
    l: float/array
        ... stroke length in mm
    dp: float
        .... piston diameter in mm
    do float
        ... orfice diameter in mm
    N: int/float
        ... number of oricies
    lowerGamma: float
        ... experimental constant, default: 2.2
        ... This represents the ratio of semi-major to semi-minor radius when a vortex bubble is approximated as an ellipsoid
    setting: str
        ... vortex ring collider versions
        ... choose from 'medium' or 'small'
        ... 'medium' refers to the chamber configuration to create D=10cm blobs
        ... 'small' refers to the chamber configuration to create D=5cm blobs

    Returns
    -------
    radius: float/array, radius of the rings
    """
    if setting == 'medium':
        dp, do, N = 160, 25.6, 8.  # Setting 1
    elif setting == 'small':  # Setting 2
        dp, do, N = 56.7, 12.8, 8.
    lstar = 1. / N * (dp / do) ** 2 * l / do
    radius = do / lowerGamma * lstar ** (1 / 3)
    return radius


def compute_ring_velocity_from_master_curve(l, veff, dp=160., do=25.6, N=8, setting='medium'):
    """
    Given a stroke length, this function returns self-induced velocity of a vortex ring in a vortex ring collider
    ... well tested over time
    ... units: mm/s
    e.g.-
        r = compute_ring_radius_from_master_curve(8.2, setting='small')
        v = compute_ring_velocity_from_master_curve(8.2, estimate_veff(8.2, 200), setting='small')
    Parameters
    ----------
    l: float/array
        ... stroke length in mm
    veff: float/array
        ... effective piston velocity- (<vp>^2 / <vp> )
        ... this quantitty can be estimated from estimate_veff(commanded_stroke_length, commanded_stroke_velocity)

    dp: float
        .... piston diameter in mm
    do float
        ... orfice diameter in mm
    N: int/float
        ... number of oricies
    lowerGamma: float
        ... experimental constant, default: 2.2
        ... This represents the ratio of semi-major to semi-minor radius when a vortex bubble is approximated as an ellipsoid
    setting: str
        ... vortex ring collider versions
        ... choose from 'medium' or 'small'
        ... 'medium' refers to the chamber configuration to create D=10cm blobs
        ... 'small' refers to the chamber configuration to create D=5cm blobs

    Returns
    -------
    velocity: float/array, velocity of the ring
    """
    if setting == 'medium':
        dp, do, N = 160, 25.6, 8.
    elif setting == 'small':
        dp, do, N = 56.7, 12.8, 8.
    lstar = 1. / N * (dp / do) ** 2 * l / do

    prefactor = 0.34 * np.tanh(0.9 * lstar)  # model of the characteristic function
    velocity = prefactor * veff / N * (dp / do) ** 2

    return velocity


def compute_ring_circulation_from_master_curve(l, veff, dp=160., do=25.6, N=8, setting='medium'):
    """
    Given a stroke length, this function returns circulation of a vortex ring in a vortex ring collider
    ... well tested over time
    ... units: mm2/s
    e.g.-
        r = compute_ring_radius_from_master_curve(8.2, setting='small')
        v = compute_ring_velocity_from_master_curve(8.2, estimate_veff(8.2, 200), setting='small')
    Parameters
    ----------
    l: float/array
        ... stroke length in mm
    veff: float/array
        ... effective piston velocity- (<vp>^2 / <vp> )
        ... this quantitty can be estimated from estimate_veff(commanded_stroke_length, commanded_stroke_velocity)

    dp: float
        .... piston diameter in mm
    do float
        ... orfice diameter in mm
    N: int/float
        ... number of oricies
    lowerGamma: float
        ... experimental constant, default: 2.2
        ... This represents the ratio of semi-major to semi-minor radius when a vortex bubble is approximated as an ellipsoid
    setting: str
        ... vortex ring collider versions
        ... choose from 'medium' or 'small'
        ... 'medium' refers to the chamber configuration to create D=10cm blobs
        ... 'small' refers to the chamber configuration to create D=5cm blobs

    Returns
    -------
    velocity: float/array, velocity of the ring
    """
    if setting == 'medium':
        dp, do, N = 160, 25.6, 8.
    elif setting == 'small':
        dp, do, N = 56.7, 12.8, 8.
    lstar = 1. / N * (dp / do) ** 2 * l / do
    geomFactor = (do * veff / N * (dp / do) ** 2)
    circulation = 0.43794512 * geomFactor * lstar ** (2 / 3.)  # 0.43794512 is obtained from the master plot

    return circulation


def estimate_ring_energy(sl, sv, dp=160., do=25.6, N=8, lowerGamma=2.2, setting='medium', rho=1e-3,
                         veff=None,
                         a=1., alpha=None, beta=None, core_type='viscous',
                         circulation_option='master curve',
                         model='thin_core',
                         verbose=False):
    """
    Given a stroke length, this function estimates energy of a vortex ring in a vortex ring collider in nJ
    ... units: nJ

    Parameters
    ----------
    sl: float/1d array, stroke length in mm
    sv: float/1d array, stroke velocity in mm/s
    dp: float, piston diameter
    do: float, orifice diameter
    N: number of orifices
    lowerGamma: float
        ... experimental constant, default: 2.2
        ... This represents the ratio of semi-major to semi-minor radius when a vortex bubble is approximated as an ellipsoid
    setting: str, options are 'medium' and 'small'
        ... if setting == 'small', it uses dp, do, N = 56.7, 12.8, 8.
        ... if setting == 'medium', it uses dp, do, N = 160, 25.6, 8.
    rho: float, mass density of medium in g/mm3, default: 1e-3 (water)
    a: float, a core size of a vortex ring
    alpha: float, a vortex ring core parameter, see Sullivan et. al. 2008
    beta: float, a vortex ring core parameter, see Sullivan et. al. 2008
    core_type: str, defalut: 'viscous'
        ... if specified, it uses the values of alpha and beta for each core model- taken from Sullivan et. al. 2008
        ... options are 'viscous', 'solid_const_volume', 'hollow_const_volume','hollow_const_pressure', 'hollow_surface_tension'

    Returns
    -------
    energy: vortex ring energy in nJ
    """
    veff = estimate_veff(sl, sv)  # psss in (mm, mm/s) -> returns in mm

    if alpha is None and beta is None:
        if core_type == 'viscous':
            alpha, beta = 2.04, 0.558
        elif core_type == 'solid_const_volume':
            alpha, beta = 1.75, 0.25
        elif core_type == 'hollow_const_volume':
            alpha, beta = 2., 0.5
        elif core_type == 'hollow_const_pressure':
            alpha, beta = 1.5, 0.5
        elif core_type == 'hollow_surface_tension':
            alpha, beta = 1., 0.
        elif core_type == 'NLSE_solution':
            alpha, beta = 1.615, 0.615

    radius = compute_ring_radius_from_master_curve(sl, dp=dp, do=do, N=N, lowerGamma=lowerGamma,
                                                   setting=setting)  # in mm
    vel = compute_ring_velocity_from_master_curve(sl, veff, dp=dp, do=do, N=N, setting=setting)  # in mm/s
    if circulation_option == 'model':
        circulation = 4 * np.pi * radius * vel / (np.log(8. * radius / a) - beta)  # in mm2/s
    elif circulation_option == 'master curve' or 'mc':
        circulation = compute_ring_circulation_from_master_curve(sl, veff, dp=dp, do=do, N=N, setting=setting)

    if model == 'spherical':
        energy = 10 / 7. * np.pi * radius ** 3 * vel ** 2
    else:
        energy = 0.5 * rho * circulation ** 2 * radius * (np.log(8. * radius / a) - alpha)  # in mm= gmm2/s2 = nJ

    if verbose:
        # ring_properties = {'core size(given by a user)': a,
        #                    'core param alpha': alpha,
        #                    'core param beta': beta,
        #                    'radius in mm': radius,
        #                    'self-induced vel in mm/s': vel,
        #                    'circulation in mm2/s': circulation,
        #                    'circulation approx- 4piRV': 4*np.pi*radius*vel,
        #                    'energy in nJ': energy,
        #                    'energy in nJ with Gamma=4piRV':  0.5 * rho * (4*np.pi*radius*vel) ** 2 * radius * (np.log(8. * radius / a) - alpha),
        #                    'Impulse in gmm /s': rho * circulation * np.pi * radius**2,
        #                    'Impulse with Gamma=4piRV in gmm /s': rho * (4*np.pi*radius*vel)  * np.pi * radius ** 2
        #                    }
        ring_properties = {'core size(given by a user)': a,
                           'core param alpha': alpha,
                           'core param beta': beta,
                           'radius in mm': radius,
                           'self-induced vel in mm/s': vel,
                           'circulation in m2/s': circulation * 1e-6,
                           'circulation approx- 4piRV in m2/s': 4 * np.pi * radius * vel * 1e-6,
                           'energy in mJ': energy * 1e-6,
                           'energy in mJ with Gamma=4piRV': 0.5 * rho * (4 * np.pi * radius * vel) ** 2 * radius * (
                                       np.log(8. * radius / a) - alpha) * 1e-6,
                           'Impulse in kg m /s': rho * circulation * np.pi * radius ** 2 * 1e-6,
                           'Impulse with Gamma=4piRV in kg m /s': rho * (
                                       4 * np.pi * radius * vel) * np.pi * radius ** 2 * 1e-6
                           }
        for key in ring_properties.keys():
            print(key, ring_properties[key])
    if type(energy) == int or float:
        return energy
    else:
        if len(energy) == 1:
            return energy[0]
        else:
            return energy


def estimate_ringVRratio(sl, sv, dp=160., do=25.6, norfices=8, lowerGamma=2.2, setting='medium', method='master_curve',
                         return_err=False, return_V_R=False):
    """
    Estimates the V/R ratio of a ring created in a vortex ring collider
    ... V/R ratio: ring velocity / ring radius
    ... Two methods are possible.
        1. By Using a master curve to infer ring velocity and radius
        2. By interpolating the experimental measurements in the past

    Parameters
    ----------
    sl: float/array
        commanded stroke length in mm
    sv: float/array
        commanded stroke velocity in mm/s
    dp: float
        .... piston diameter in mm
    do float
        ... orfice diameter in mm
    norifices: int/float
        ... number of oricies
    lowerGamma: float
        ... experimental constant, default: 2.2
        ... This represents the ratio of semi-major to semi-minor radius when a vortex bubble is approximated as an ellipsoid
    setting: str
        ... vortex ring collider versions
        ... choose from 'medium' or 'small'
        ... 'medium' refers to the chamber configuration to create D=10cm blobs
        ... 'small' refers to the chamber configuration to create D=5cm blobs
    method: str, choose from "master_curve" and "measurement"
    return_err: bool
        ... Only applicable to method=="master_curve"

    Returns
    -------
    vrRatio, (vrRatio_err): float/arrays, float/arrays
        ... Self-induced velocity / radius of a vortex ring(s) created by (stroke length, stroke velocity)
        ... Error of V/R ratio due to the discrepancy between the rings created at the top and bottom orifices
    """
    if setting == 'medium':
        dp, do, norfices = 160, 25.6, 8.
    elif setting == 'small':
        dp, do, norfices = 56.7, 12.8, 8.
    veff = estimate_veff(sl, sv)

    if method == 'master_curve':  # from data collapse
        radius = compute_ring_radius_from_master_curve(sl, dp=dp, do=do, lowerGamma=lowerGamma, setting=setting)
        velocity = compute_ring_velocity_from_master_curve(sl, veff, dp=dp, do=do, setting=setting)
        vrRatio = velocity / radius
        if return_V_R:
            return vrRatio, velocity, radius
        else:
            return vrRatio
    elif method == 'measurement' and setting == 'medium':  # from past experimental results
        # get module location
        mod_loc = os.path.abspath(__file__)
        pdir, filename = os.path.split(mod_loc)
        ringDataDir = os.path.join(os.path.join(pdir, 'reference_data'), 'vortex_ring_data_setting_medium')

        # load interpolating functions which takes arguments (L/D, veff)
        f_vrT = read_pickle(os.path.join(ringDataDir, 'f_vrT.pkl'))  # velocity of a top ring
        f_vrB = read_pickle(os.path.join(ringDataDir, 'f_vrB.pkl'))  # velocity of a bottom ring
        f_drT = read_pickle(os.path.join(ringDataDir, 'f_drT.pkl'))  # diameter of a top ring
        f_drB = read_pickle(os.path.join(ringDataDir, 'f_drB.pkl'))  # diameter of a bottom ring
        # f_gammarT = read_pickle(os.path.join(ringDataDir, 'f_gammarT.pkl')) # circulation of a bottom ring
        # f_gammarB = read_pickle(os.path.join(ringDataDir, 'f_gammarB.pkl')) # circulation of a bottom ring

        # get formation number (L/D)
        ld = compute_form_no(sl, orifice_d=do, piston_d=dp, num_orifices=norfices)
        vrT, vrB = f_vrT(ld, veff), f_vrB(ld, veff)
        rrT, rrB = f_drT(ld, veff) / 2., f_drB(ld, veff) / 2.
        vr_avg, rr_avg = (vrT + vrB) / 2., (rrT + rrB) / 2.
        deltaVr, deltaRr = np.abs(vrT - vr_avg), np.abs(rr_avg - rrT)
        vrRatio = vr_avg / rr_avg
        vrRatio_err = vrRatio * np.sqrt((deltaVr / vr_avg) ** 2 + (deltaRr / rr_avg) ** 2)

        if return_err:
            if return_V_R:
                return vrRatio, vrRatio_err, vr_avg, rr_avg
            else:
                return vrRatio, vrRatio_err
        else:
            if return_V_R:
                return vrRatio, vr_avg, rr_avg
            else:
                return vrRatio


def estimate_cirulation_vring_collider(sl, sv):
    """
    Returns the circulation of a vortex ring created in the vortex ring collider (setting 1- 10cm blob creation)

    Parameters
    ----------
    sl: float/array
        commanded stroke length in mm
    sv: float/array
        commanded stroke velocity in mm/s

    Returns
    -------
    gammaT, gammaB: tuple of float values/arrays, circulation of the top and the bottom rings

    """
    print('... estimate_cirulation_vring_collider: Only applicable to the piston setting1 (~10cm blob creation)')
    # get module location
    mod_loc = os.path.abspath(__file__)
    pdir, filename = os.path.split(mod_loc)
    ringDataDir = os.path.join(os.path.join(pdir, 'reference_data'), 'vortex_ring_data_setting_medium')
    f_gammarT = read_pickle(os.path.join(ringDataDir, 'f_gammarT.pkl'))  # circulation of a bottom ring
    f_gammarB = read_pickle(os.path.join(ringDataDir, 'f_gammarB.pkl'))  # circulation of a bottom ring
    dp, do, norfices = 160, 25.6, 8.  # chamber parameters
    # compute L/D and veff
    ld = compute_form_no(sl, orifice_d=do, piston_d=dp, num_orifices=norfices)
    veff = estimate_veff(sl, sv)

    gammaT, gammaB = f_gammarT(ld, veff), f_gammarB(ld, veff)
    return gammaT, gammaB


def estimate_velocity_vring_collider(sl, sv):
    """
    Returns the velocity of a vortex ring created in the vortex ring collider (setting 1- 10cm blob creation)

    Parameters
    ----------
    sl: float/array
        commanded stroke length in mm
    sv: float/array
        commanded stroke velocity in mm/s

    Returns
    -------
    vrT, vrB: tuple of float values/arrays, velocity of the top and the bottom rings

    """
    print('... estimate_velocity_vring_collider: Only applicable to the piston setting1 (~10cm blob creation)')
    # get module location
    mod_loc = os.path.abspath(__file__)
    pdir, filename = os.path.split(mod_loc)
    ringDataDir = os.path.join(os.path.join(pdir, 'reference_data'), 'vortex_ring_data_setting_medium')
    f_vrT = read_pickle(os.path.join(ringDataDir, 'f_vrT.pkl'))  # velocity of a top ring
    f_vrB = read_pickle(os.path.join(ringDataDir, 'f_vrB.pkl'))  # velocity of a bottom ring
    dp, do, norfices = 160, 25.6, 8.  # chamber parameters
    # compute L/D and veff
    ld = compute_form_no(sl, orifice_d=do, piston_d=dp, num_orifices=norfices)
    veff = estimate_veff(sl, sv)
    vrT, vrB = f_vrT(ld, veff), f_vrB(ld, veff)
    return vrT, vrB


def estimate_radius_vring_collider(sl, sv):
    """
    Returns the radius of a vortex ring created in the vortex ring collider (setting 1- 10cm blob creation)

    Parameters
    ----------
    sl: float/array
        commanded stroke length in mm
    sv: float/array
        commanded stroke velocity in mm/s

    Returns
    -------
    rrT, rrB : tuple of float values/arrays, radius of the top and the bottom rings

    """
    print('... estimate_radius_vring_collider: Only applicable to the piston setting1 (~10cm blob creation)')
    # get module location
    mod_loc = os.path.abspath(__file__)
    pdir, filename = os.path.split(mod_loc)
    ringDataDir = os.path.join(os.path.join(pdir, 'reference_data'), 'vortex_ring_data_setting_medium')
    f_drT = read_pickle(os.path.join(ringDataDir, 'f_drT.pkl'))  # diameter of a top ring
    f_drB = read_pickle(os.path.join(ringDataDir, 'f_drB.pkl'))  # diameter of a bottom ring
    dp, do, norfices = 160, 25.6, 8.  # chamber parameters
    # compute L/D and veff
    ld = compute_form_no(sl, orifice_d=do, piston_d=dp, num_orifices=norfices)
    veff = estimate_veff(sl, sv)
    rrT, rrB = f_drT(ld, veff) / 2., f_drB(ld, veff) / 2.
    return rrT, rrB


# h5 helpers
def show_h5_keys(dpath):
    """Displays a list of dataset names in h5- only the datasets stored under the top"""
    with h5py.File(dpath, mode='r') as f:
        print(f.keys())


def show_h5_subkeys(dpath, key):
    """Displays a list of dataset names in h5- only the datasets stored under the top"""
    with h5py.File(dpath, mode='r') as f:
        try:
            print(f[key].keys())
        except KeyError:
            print(f'{key} does not exist in {dpath}')


def get_h5_keys(dpath):
    """Returns a list of dataset names in h5- only the datasets stored under the top
    ... /Dataset1, /Dataset2, ... -> Returns ["Dataset1", "Dataset2"]
    """
    with h5py.File(dpath, mode='r') as f:
        keys = [key for key in f.keys()]
    return keys


def get_h5_subkeys(dpath, key):
    """Returns a list of dataset names in h5- only the datasets stored under /key
    ... key/Dataset1, key/Dataset2, ... -> Returns ["Dataset1", "Dataset2"]
    """
    with h5py.File(dpath, mode='r') as f:
        subkeys = [subkey for subkey in f[key].keys()]
    return subkeys


# labeler
def suggest_name2write(filepath):
    """
    Returns a new filepath with a version number if a file with a given path already exists.
    Otherwise, it returns the given string

    Parameters
    ----------
    filepath: str

    Returns
    -------
    newfilepath: new filepath with a version number if the file already exists
    """
    pdir, filename = os.path.split(filepath)
    filepaths = glob.glob(os.path.join(pdir, '*'))
    filepath_wo_ext, ext = os.path.splitext(filepath)
    verNo = 1
    if not os.path.exists(filepath):
        return filepath
    while filepath in filepaths:
        newfilepath = filepath_wo_ext + '_%02d' % verNo + ext
        verNo += 1
        if not os.path.exists(newfilepath):
            return newfilepath
        if verNo == 100:
            raise ValueError('suggest_name2write: there are at least 100 versions of the file. Choose a different name')


# miscellaneous but useful functions
def get_mask_for_nan_and_inf(U):
    """
    Returns a mask for nan and inf values in a multidimensional array U

    Parameters
    ----------
    U: nd array

    Returns
    -------
    mask: nd array, boolean
        ... True if an element is nan or inf, False otherwise.
    """
    U = np.array(U)
    U_masked_invalid = ma.masked_invalid(U)
    return U_masked_invalid.mask


def resample(x, y, n=100, mode='linear'):
    """
    Resample x, y
    ... this is particularly useful to crete a evenly spaced data in log from a linearly spaced data, and vice versa

    Parameters
    ----------
    x: 1d array
    y: 1d array
    n: int, number of points to resample
    mode: str, options are "linear" and "log"

    Returns
    -------
    x_new, y_rs: 1d arrays of new x and new y
    """
    # x, y = copy.deepcopy(x_), copy.deepcopy(y_)
    x, y = np.array(x), np.array(y)  # np.array creates a new object unlike np.asarray

    # remove nans and infs
    hidex = get_mask_for_nan_and_inf(x)
    hidey = get_mask_for_nan_and_inf(y)
    keep = ~hidex * ~hidey
    x, y = x[keep], y[keep]

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if mode == 'log':
        if xmax < 0:
            raise ValueError('... log sampling cannot be performed as the max. value of x is less than 0')
        else:
            if xmin > 0:
                keep = [True] * len(x)
            else:
                keep = x > 0  # ignore data points s.t. x < 0
            xmin = np.nanmin(x[keep])
            logx = np.log10(x[keep])
            logxmin, logxmax = np.log10(xmin), np.log10(xmax)
            logx_new = np.linspace(logxmin, logxmax, n, endpoint=True)
            x_new = 10 ** logx_new
            flog = interpolate.interp1d(logx, y[keep])
            y_rs = flog(logx_new)
            return x_new, y_rs
    elif mode == 'loglog':
        if xmax < 0:
            raise ValueError('... log sampling cannot be performed as the max. value of x is less than 0')
        else:
            if xmin > 0:
                keep = [True] * len(x)
            else:
                keep = x > 0  # ignore data points s.t. x < 0
        xmin = np.nanmin(x[keep])
        logx = np.log10(x[keep])
        logxmin, logxmax = np.log10(xmin), np.log10(xmax)
        logx_new = np.linspace(logxmin, logxmax, n, endpoint=True)
        x_new = 10 ** logx_new
        flog = interpolate.interp1d(logx, np.log10(y[keep]))
        y_rs = 10 ** flog(logx_new)
        return x_new, y_rs
    else:
        x_new = np.linspace(xmin, xmax, n, endpoint=True)
        #         y_rs = scipy.signal.resample(y, n)
        f = interpolate.interp1d(x, y)
        y_rs = f(x_new)

        return x_new, y_rs


def resample2d(x, ys, n=100, mode='linear', return_x_in_2d=False):
    """
    Resample x, y
    ... this is particularly useful to crete a evenly spaced data in log from a linearly spaced data, and vice versa

    Parameters
    ----------
    x: 1d array
    ys: 2d array
    n: int, number of points to resample
    mode: str, options are "linear" and "log"

    Returns
    -------
    x_new, y_rs: 1d arrays of new x and new y
    """
    x, ys = np.array(x), np.array(ys)  # np.array creates a new object unlike np.asarray
    _, duration = ys.shape
    new_shape = (n, duration)
    x_resampled = np.empty(new_shape)
    y_resampled = np.empty(new_shape)

    for t in range(duration):
        y = ys[:, t]
        x_resampled[:, t], y_resampled[:, t] = resample(x, y, n=n, mode=mode)
    if return_x_in_2d:
        return x_resampled, y_resampled
    else:
        return x_resampled[:, 0], y_resampled


def find_n_largest_values(arr, n):
    """Returns n largest values in a given array"""
    return np.partition(arr, -n)[-n:]


def find_n_smallest_values(arr, n):
    """Returns n smallest values in a given  array"""
    return np.partition(arr, n)[:n]


# IMAGE ANALYSIS
def getHuMoments(qty, method='binary', thd=100, log=True):
    """
    Returns Hu Moments of a 2D array

    Parameters
    ----------
    qty: 2d array
    method: 'binary' or 'thresholding'
    thd: int/float, value for binarization or thresholding
    log:

    Returns
    -------
    huMoments_: 1d array of hu moments (h0-h6), h6 flips a sign under reflection
    """

    if method == 'binary':
        cv2Method = cv2.THRESH_BINARY
    elif method == 'thresholding':
        cv2Method = cv2.THRESH_TOZERO
    img = qty / np.nanmax(qty) * 255
    _, img = cv2.threshold(img, thd, 255, cv2Method)  # THRESH_TOZERO, THRESH_BINARY
    moments_ = cv2.moments(img)
    huMoments_ = cv2.HuMoments(moments_)
    huMoments_ = np.asaray([foo[0] for foo in huMoments_])
    if log:
        for i in range(0, 7):
            huMoments_[i] = -1 * math.copysign(1.0, huMoments_[i]) * math.log10(abs(huMoments_[i]))
    return huMoments_


def computeImageDistanceUsingHuMoments(img1, img2, method=cv2.CONTOURS_MATCH_I2, ):
    """
    Returns the image distance between two images
    Parameters
    ----------
    img1: 2d array- the elements must be between 0 and 255.
    img2: 2d array- the elements must be between 0 and 255.
    method: this method gets passed to cv2.matchShapes (e.g.- cv2.CONTOURS_MATCH_I1, cv2.CONTOURS_MATCH_I2, cv2.CONTOURS_MATCH_I3)

    Returns
    -------
    imgDist- a scalar quantity which represents a distance between two images based on the (logarithmic) Hu moments.
    """
    imgDist = cv2.matchShapes(img1, img2, method, 0)
    return imgDist
