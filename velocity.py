from scipy.stats import binned_statistic
from tqdm import tqdm
import numpy as np
import re
import h5py
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import minimize
import numpy.ma as ma
from scipy import signal
from scipy import interpolate
from scipy.interpolate import griddata
from scipy import ndimage
from scipy.stats import multivariate_normal
import itertools
import os, copy, sys, re  # fundamentals
import time as time_mod
import ilpm.vector as vec  # irvinelab codes

import warnings

warnings.simplefilter('ignore', RuntimeWarning)
# For plotting
import matplotlib.pyplot as plt
import tflow.graph as graph
import subprocess

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
h5py, tqdm, ilpm.vector

author: takumi matsuzawa
"""


########## Fundamental operations ##########
def get_duidxj_tensor(udata, dx=1., dy=1., dz=1.):
    """
    Assumes udata has a shape (d, nrows, ncols, duration) or  (d, nrows, ncols)
    ... one can easily make udata by np.stack((ux, uy))

    Parameters
    ----------
    udata: numpy array with shape (ux, uy) or (ux, uy, uz)
        ... assumes ux/uy/uz has a shape (nrows, ncols, duration) or (nrows, ncols, nstacks, duration)
        ... can handle udata without temporal axis

    Returns
    -------
    sij: numpy array with shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
        ... idea is... sij[spacial coordinates, time, tensor indices]
            e.g.-  sij(x, y, t) = sij[y, x, t, i, j]
        ... sij = d ui / dxj
    """
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    if shape[0] == 2:
        ux, uy = udata[0, ...], udata[1, ...]
        try:
            dim, nrows, ncols, duration = udata.shape
        except:
            dim, nrows, ncols = udata.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], duration))

        duxdx = np.gradient(ux, dx, axis=1)
        duxdy = np.gradient(ux, dy, axis=0)
        duydx = np.gradient(uy, dx, axis=1)
        duydy = np.gradient(uy, dy, axis=0)
        sij = np.zeros((nrows, ncols, duration, dim, dim))
        sij[..., 0, 0] = duxdx
        sij[..., 0, 1] = duxdy
        sij[..., 1, 0] = duydx
        sij[..., 1, 1] = duydy
    elif shape[0] == 3:
        dim = 3
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
        duxdx = np.gradient(ux, dx, axis=1)
        duxdy = np.gradient(ux, dy, axis=0)
        duxdz = np.gradient(ux, dz, axis=2)
        duydx = np.gradient(uy, dx, axis=1)
        duydy = np.gradient(uy, dy, axis=0)
        duydz = np.gradient(uy, dz, axis=2)
        duzdx = np.gradient(uz, dx, axis=1)
        duzdy = np.gradient(uz, dy, axis=0)
        duzdz = np.gradient(uz, dz, axis=2)

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
    elif shape[0] > 3:
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
    gij: 5d or 6d numpy array, anti-symmetric part of rate-of-strain tensor.
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


def reynolds_decomposition(udata):
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
        u_mean[i] = np.nanmean(udata[i], axis=dim)  # axis=dim is always the time axis in this convention
        for t in range(udata.shape[-1]):
            u_turb[i, ..., t] = udata[i, ..., t] - u_mean[i]
    return u_mean, u_turb


########## vector operations ##########
def div(udata):
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
    sij = get_duidxj_tensor(
        udata)  # shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    dim = len(sij.shape) - 3  # spatial dim
    div_u = np.zeros(sij.shape[:-2])
    for d in range(dim):
        div_u += sij[..., d, d]
    return div_u


def curl(udata, dx=1., dy=1., dz=1.):
    """
    Computes curl of a velocity field using a rate of strain tensor
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = np.stack((ugrid1, vgrid1))
        omega = vec.curl(udata)
    Parameters
    ----------
    udata: (ux, uy, uz) or (ux, uy)
    dx, dy, dz: float, spatial spating of a 2D/3D grid

    Returns
    -------
    omega: numpy array
        shape: (height, width, duration) (2D) or (height, width, duration) (2D)

    """
    sij = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz)
    dim = len(sij.shape) - 3  # spatial dim
    eij, gij = decompose_duidxj(sij)
    if dim == 2:
        omega = 2 * gij[..., 1, 0]  # checked. this is correct.
    elif dim == 3:
        # sign might be wrong
        omega1, omega2, omega3 = 2. * gij[..., 2, 1], 2. * gij[..., 0, 2], 2. * gij[..., 1, 0]
        # omega1, omega2, omega3 = -2. * gij[..., 2, 1], 2. * gij[..., 0, 2], -2. * gij[..., 1, 0]
        omega = np.stack((omega1, omega2, omega3))
    else:
        print('Not implemented yet!')
        return None
    return omega


def curl_2d(ux, uy, dx=1., dy=1.):
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

    # duxdx = np.gradient(ux, axis=1)
    duxdy = np.gradient(ux, dy, axis=0)
    duydx = np.gradient(uy, dx, axis=1)
    # duydy = np.gradient(uy, axis=0)

    omega = duydx - duxdy

    return omega


########## Elementary analysis ##########
def get_energy(udata):
    """
    Returns energy(\vec{x}, t) of udata
    ... Assumes udata is equally spaced data.

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    energy: nd array
        energy
    """
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = udata.shape[0]
    energy = np.zeros(shape[1:])
    for d in range(dim):
        energy += udata[d, ...] ** 2
    energy /= 2.
    return energy


def get_enstrophy(udata, dx=1., dy=1., dz=1.):
    """
    Returns enstrophy(\vec{x}, t) of udata
    ... Assumes udata is equally spaced data.

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
        enstrophy
    """
    dim = udata.shape[0]
    omega = curl(udata, dx=dx, dy=dy, dz=dz)
    shape = omega.shape  # shape=(dim, nrows, ncols, nstacks, duration) if nstacks=0, shape=(dim, nrows, ncols, duration)
    if dim == 2:
        enstrophy = omega ** 2 / 2.
    elif dim == 3:
        enstrophy = np.zeros(shape[1:])
        for d in range(dim):
            enstrophy += omega[d, ...] ** 2
        enstrophy /= 2.
    return enstrophy


def get_time_avg_energy(udata):
    """
    Returns a time-averaged-energy field
    ... NOT MEAN FLOW ENERGY

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    energy_avg:
        time-averaged energy field
    """
    dim = udata.shape[0]
    energy = get_energy(udata)
    energy_avg = np.nanmean(energy, axis=dim)
    return energy_avg


def get_time_avg_enstrophy(udata, dx=1., dy=1., dz=1.):
    """
    Returns a time-averaged-enstrophy field
    ... NOT MEAN FLOW ENSTROPHY

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    enstrophy_avg: nd array
        time-averaged enstrophy field
    """
    dim = udata.shape[0]
    enstrophy = get_enstrophy(udata, dx=dx, dy=dy, dz=dz)
    enstrophy_avg = np.nanmean(enstrophy, axis=dim)
    return enstrophy_avg


def get_spatial_avg_energy(udata, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    Return energy averaged over space

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    energy_vs_t: 1d numpy array
        average energy in a field at each time
    energy_vs_t_err
        standard deviation of energy in a field at each time
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
    energy = get_energy(udata)
    energy_vs_t = np.nanmean(energy, axis=tuple(range(dim)))
    energy_vs_t_err = np.nanstd(energy, axis=tuple(range(dim)))
    return energy_vs_t, energy_vs_t_err


def get_spatial_avg_enstrophy(udata, x0=0, x1=None, y0=0, y1=None,
                              z0=0, z1=None, dx=1., dy=1., dz=1.):
    """
    Return enstrophy averaged over space

    Parameters
    ----------
    udata: nd array

    Returns
    -------
    enstrophy_vs_t: 1d numpy array
        average enstrophy in a field at each time
    enstrophy_vs_t_err
        standard deviation of enstrophy in a field at each time
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

    enstrophy = get_enstrophy(udata, dx=dx, dy=dy, dz=dz)
    enstrophy_vs_t = np.nanmean(enstrophy, axis=tuple(range(dim)))
    enstrophy_vs_t_err = np.nanstd(enstrophy, axis=tuple(range(dim)))
    return enstrophy_vs_t, enstrophy_vs_t_err


def get_turbulence_intensity_local(udata):
    """
    Turbulence intensity is defined as u/U where
    u = sqrt((ux**2 + uy**2 + uz**2)/3) # characteristic turbulent velocity
    U = sqrt((Ux**2 + Uy**2 + Uz**2))   # norm of the rms velocity

    Note that this is ill-defined for turbulence with zero-mean flow !

    Parameters
    ----------
    udata

    Returns
    -------
    ti_local: nd array
        turbulent intensity field (scaler field)

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


########## Energy spectrum, Dissipation spectrum ##########

# def fft_nd(field, dx=1, dy=1, dz=1):
#     """
#     Parameters
#     ----------
#     field: np array, (height, width, depth, duration) or (height, width, duration)
#     dx: spacing along x-axis
#     dy: spacing along x-axis
#     dz: spacing along x-axis
#
#     Returns
#     -------
#     field_fft
#     np.asarray([kx, ky, kz])
#
#     """
#     dim = len(field.shape) - 1
#     n_samples = 1
#     for d in range(dim):
#         n_samples *= field.shape[d]
#
#     field_fft = np.fft.fftn(field, axes=list(range(dim)))
#     field_fft = np.fft.fftshift(field_fft, axes=list(range(dim)))
#     field_fft /= n_samples# Divide the result by the number of samples (this is because of discreteness of FFT)
#
#     if dim == 2:
#         height, width, duration = field.shape
#         kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
#         ky = np.fft.fftfreq(height, d=dy)
#         kx = np.fft.fftshift(kx)
#         ky = np.fft.fftshift(ky)
#         kxx, kyy = np.meshgrid(kx, ky)
#         kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber
#
#         return field_fft, np.asarray([kxx, kyy])
#
#     elif dim == 3:
#         height, width, depth, duration = field.shape
#         kx = np.fft.fftfreq(width, d=dx)
#         ky = np.fft.fftfreq(height, d=dy)
#         kz = np.fft.fftfreq(depth, d=dz)
#         kx = np.fft.fftshift(kx)
#         ky = np.fft.fftshift(ky)
#         kz = np.fft.fftshift(kz)
#         kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
#         kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
#         return field_fft, np.asarray([kxx, kyy, kzz])
#
def fft_nd(udata, dx=1, dy=1, dz=1,
           x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
           window=None, return_kgrid=True):
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


def get_energy_spectrum_nd(udata, x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None,
                           window=None, correct_signal_loss=True):
    """
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
        kxx, kyy, kzz = kxx, kyy, kzz

        # Output the SPECTRAL DENSITY
        # ... DFT outputs the integrated density (which is referred as POWER) in a pixel(delta_kx * delta_ky)
        # ... But energy spectrum is indeed plotting the SPECTRAL DENSITY!
        deltakx, deltaky, deltakz = kx[1] - kx[0], ky[1] - ky[0], kz[1] - kz[0]
        ek = ek / (deltakx * deltaky * deltakz)

        return ek, np.asarray([kxx, kyy, kzz])


def get_energy_spectrum(udata, x0=0, x1=None, y0=0, y1=None,
                        z0=0, z1=None, dx=None, dy=None, dz=None, nkout=None,
                        window=None, correct_signal_loss=True, remove_undersampled_region=True,
                        cc=1.75, notebook=True):
    """
    Returns 1D energy spectrum from velocity field data
    ... The algorithm implemented in this function is VERY QUICK because it does not use the two-point vel. autorcorrelation tensor.
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

        Returns
        -------

        """
        dim = ks.shape[0]
        duration = e_ks.shape[-1]
        if dim == 2:
            deltakx, deltaky = ks[0, 0, 1] - ks[0, 0, 0], ks[1, 1, 0] - ks[1, 0, 0]
            e_ks *= deltakx * deltaky  # use the raw DFT outputs (power=integrated density over a px)
            deltakr = np.sqrt(deltakx ** 2 + deltaky ** 2)  # radial k spacing of the velocity field
        if dim == 3:
            deltakx, deltaky, deltakz = ks[0, 0, 1, 0] - ks[0, 0, 0, 0], ks[1, 1, 0, 0] - ks[1, 0, 0, 0], ks[
                2, 0, 0, 1] - ks[2, 0, 0, 0]
            e_ks *= deltakx * deltaky * deltakz  # use the raw DFT outputs (power=integrated density over a px)
            deltakr = np.sqrt(deltakx ** 2 + deltaky ** 2 + deltakz ** 2)  # radial k spacing of the velocity field
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
                # print len(kk_flatten)
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
            # Old stuff 2: scaling that works
            # e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltak) * (deltak / deltakr) ** dim / deltak
            # e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltak) * (deltak / deltakr) ** dim / deltak
            # print deltak,  deltakr, (deltak / deltakr)
            e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltak) * (deltak / deltakr) ** dim / deltak * cc
            e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltak) * (deltak / deltakr) ** dim / deltak * cc

            # print deltak / deltakr

        return e_k1ds, e_k1d_errs, k1ds

    dim, duration = len(udata), udata.shape[-1]

    e_ks, ks = get_energy_spectrum_nd(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz,
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


def get_1d_energy_spectrum(udata, k='kx', x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None,
                           window=None, correct_signal_loss=True):
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

    n_samples = 1
    for i in range(dim):
        n_samples *= ux.shape[i]
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
        ax_ind_for_avg = (0, 1)  # axis number(s) to take statistics  (along x and y)

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
    ux_k = np.fft.fft(ux, axis=ax_ind) / np.sqrt(n_samples)
    e11_nd = np.abs(ux_k * np.conj(ux_k))
    e11 = np.nanmean(e11_nd, axis=ax_ind_for_avg)[:n // 2, :]
    e11_err = np.nanstd(e11_nd, axis=ax_ind_for_avg)[:n // 2, :]
    # E22
    uy_k = np.fft.fft(uy, axis=ax_ind) / np.sqrt(n_samples)
    e22_nd = np.abs(uy_k * np.conj(uy_k))
    e22 = np.nanmean(e22_nd, axis=ax_ind_for_avg)[:n // 2, :]
    e22_err = np.nanstd(e22_nd, axis=ax_ind_for_avg)[:n // 2, :]

    # Get an array for wavenumber
    k = np.fft.fftfreq(n, d=d)[:n // 2] * 2 * np.pi  # shape=(n, duration)
    deltak = k[1] - k[0]
    if dim == 3:
        # E33
        uz_k = np.fft.fft(uz, axis=ax_ind) / np.sqrt(n_samples)
        e33_nd = np.abs(uz_k * np.conj(uz_k))
        e33 = np.nanmean(e33_nd, axis=ax_ind_for_avg)[:n // 2, :]
        e33_err = np.nanstd(e33_nd, axis=ax_ind_for_avg)[:n // 2, :]

        eiis, eii_errs = np.array([e11, e22, e33]), np.array([e11_err, e22_err, e33_err])
    elif dim == 2:
        eiis, eii_errs = np.array([e11, e22]), np.array([e11_err, e22_err])
    else:
        raise ValueError('... 1d spectrum: Check the dimension of udata! It must be 2 or 3!')

    # Convert power to spectral density
    # ... DFT outputs the integrated power between k and k + deltak
    # ... One must divide the integrated power by deltak to account for this.
    for i in range(dim):
        eiis[i] /= deltak
        eii_errs[i] /= deltak

    # Windowing causes the loss of the signal (energy.)
    # ... This compensates for the loss.
    if correct_signal_loss:
        for i in range(dim):
            eiis[i] /= signal_intensity_loss
            eii_errs[i] /= signal_intensity_loss

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
    except:
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
    Returns a velocity field which satisfies k = sqrt(kx^2 + ky^2) < kmax in the original vel. field (udata)

    Parameters
    ----------
    udata: nd array
    ... velocity field data with shape (# of components, physical dimensions (width x height x depth), duration)
    kmax: float
    ... value of k below which spectrum is kept. i.e. cutoff k for the low-pass filter
    x0: int
    ... index used to specify a region of a vel. field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    x1: int
    ... index used to specify a region of a vel. field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    y0: int
    ... index used to specify a region of a vel. field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    y1: int
    ... index used to specify a region of a vel. field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    z0: int
    ... index used to specify a region of a vel. field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
    z1: int
    ... index used to specify a region of a vel. field in which spectrum is computed. udata[y0:y1, x0:x1, z0:z1]
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
        epsilon_2 = np.nanmean(duidxj[..., 0, 1] * duidxj[..., 0, 1], axis=tuple(range(dim)))
        epsilon = 6. * nu * (epsilon_0 + epsilon_1 + epsilon_2)  # Hinze, 1975, eq. 3-98
    return epsilon


def get_epsilon_iso(udata, lambda_f=None, lambda_g=None, nu=1.004, x=None, y=None, **kwargs):
    """
    Return epsilon computed by isotropic formula involving Taylor microscale

    Parameters
    ----------
    udata
    lambda_f: numpy array
        long. Taylor microscale
    lambda_g: numpy array
        transverse. Taylor microscale
    nu: float
        viscosity

    Returns
    -------
    epsilon: numpy array
        dissipation rate
    """
    dim = len(udata)
    u2_irms = 2. / dim * get_spatial_avg_energy(udata)[0]

    # if both of lambda_g and lambda_f are provided, use lambdaf over lambdag
    if lambda_f is None and lambda_g is None:
        print('... Both of Taylor microscales, lambda_f, lambda_g, were not provided!')
        print('... Compute lambdas from scratch. One must provide x and y.')
        if x is None or y is None:
            raise ValueError('... x and y were not provided! Exitting...')
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
    ... must have fully resolved spectrum to yield a reasonable result

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


def get_epsilon_using_struc_func(rrs, Dxxs, epsilon_guess=100000, r0=1.0, r1=10.0, p=2, method='Nelder-Mead'):
    """
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


########## advanced analysis ##########
## Spatial autocorrelation functions
def compute_spatial_autocorr(ui, x, y, roll_axis=1, n_bins=None, x0=0, x1=None, y0=0, y1=None,
                             t0=None, t1=None, coarse=1.0, coarse2=0.2, notebook=True):
    """
    Compute spatial autocorrelation function of 2+1 velocity field
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
    ui: numpy array, 2 + 1 scalar field. i.e. shape: (height, width, duration)
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
    coarse: float (0, 1], Process coarse * possible data points. This is an option to output coarse results.
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
        t1 = ui.shape[2]
    elif t1 < 0:
        t1 = ui.shape[2] - t1

    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
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
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        # uu2_norm = np.nanmean(ui[y0:y1, x0:x1, ...] ** 2, axis=(0, 1))  # mean square velocity (avg over space)
        uu2_norm = np.nanmean(uu ** 2)  # mean square velocity (avg over space)

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
            rr = sorted(rr_)

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
            rr_means, rr_edges, binnumber = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
            corr_, _, _ = binned_statistic(rr, corr, statistic='mean', bins=n_bins)
            corr_err, _, _ = binned_statistic(rr, corr, statistic='std', bins=n_bins)

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
            corr_err /= corr[0]

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
                               coarse=1.0, coarse2=0.2, notebook=True):
    """
    Compute spatial autocorrelation function of 2+1 velocity field
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

    # Use a portion of data
    z_grid, y_grid, x_grid = z[y0:y1, x0:x1, z0:z1], y[y0:y1, x0:x1, z0:z1], x[y0:y1, x0:x1, z0:z1]

    # Initialization
    rrs, corrs, corr_errs = np.zeros((n_bins, t1 - t0)), np.ones((n_bins, t1 - t0)), np.zeros((n_bins, t1 - t0))

    for t in tqdm(list(range(t0, t1)), desc='autocorr. 3d time'):
        # Call velocity field at time t as uu
        uu = ui[y0:y1, x0:x1, z0:z1, t]

        uu2_norm = np.nanmean(ui[y0:y1, x0:x1, z0:z1, ...] ** 2, axis=(0, 1, 2))  # mean square velocity

        # Initialization
        # rr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))
        # corr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))

        roll_indices = list(range(0, limits[roll_axis], int(1. / coarse)))
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        rr = np.empty((n, m))
        corr = np.empty((n, m))

        for j, i in enumerate(tqdm(roll_indices, desc='computing correlation')):
            uu_rolled = np.roll(uu, i, axis=roll_axis)
            x_grid_rolled, y_grid_rolled, z_grid_rolled = np.roll(x_grid, i, axis=roll_axis), \
                                                          np.roll(y_grid, i, axis=roll_axis), \
                                                          np.roll(z_grid, i, axis=roll_axis)
            r_grid = np.sqrt(
                (x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2 + (z_grid_rolled - z_grid) ** 2)
            corr_uu = uu * uu_rolled / uu2_norm[t]  # correlation values
            rr[:, j] = r_grid.flatten()[:n]
            corr[:, j] = corr_uu.flatten()[:n]

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

        rrs[0:, t], corr_errs[0:, t] = 0, 1.
        corrs[0, t] = 1.
        _, corrs[1:, t] = sort2arr(rr_, corr_)
        rrs[1:, t], corr_errs[1:, t] = sort2arr(rr_, corr_err)

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, corrs, corr_errs


def get_two_point_vel_corr(udata, x, y, z=None, time=None, n_bins=None,
                           x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                           nd=10 ** 3, nr=70, nt=10, notebook=True, return_rij=False, **kwargs):
    if notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    def get_rotation_matrix_between_two_vectors(a, b):
        """
        Returns a 3D rotation matrix R that rotates a unit vector onto a unit vector of b
        """
        a, b = vec.norm(a), vec.norm(b)
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
        nt = 1

    # Initialization
    rrs, fs, f_errs = np.zeros((nr, t1 - t0)), np.ones((nr, t1 - t0)), np.zeros((nr, t1 - t0))
    rrs, gs, g_errs = np.zeros((nr, t1 - t0)), np.ones((nr, t1 - t0)), np.zeros((nr, t1 - t0))
    is_R1_reasonable = False

    if dim == 2:
        xmin, xmax, ymin, ymax = np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid)
        width, height = xmax - xmin, ymax - ymin
        rs_ = np.linspace(dx * 2, min([width, height]) * 1, nr)
    elif dim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid), np.min(
            z_grid), np.max(z_grid)
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin
        rs_ = np.linspace(dx, min([width, height, depth]), nr)
    rs = np.empty(nd)
    fs_ = np.empty(nd)
    gs_ = np.empty(nd)
    denominators_f = np.empty(nd)
    denominators_g = np.empty(nd)

    uirms = get_characteristic_velocity(udata)
    for t in tqdm(list(range(t0, t1)), desc='struc. func. time'):
        for i, r in enumerate(tqdm(rs_, desc='struc. func. r-loop')):
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
                        X1, Y1, Z1 = X0 + r * np.sin(theta) * np.cos(phi), Y0 + r * np.sin(theta) * np.sin(
                            phi), Z0 + r * np.cos(theta)
                        is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin and Z1 < zmax and Z1 > zmin
                    X1_ind, _ = find_nearest(x_grid[0, :, 0], X1)
                    Y1_ind, _ = find_nearest(y_grid[:, 0, 0], Y1)
                    Z1_ind, _ = find_nearest(z_grid[0, 0, :], Z1)
                    R1 = np.asarray([x_grid[0, X1_ind, 0], y_grid[Y1_ind, 0, 0], z_grid[0, 0, Z1_ind]])

                R01 = R1 - R0
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
                               coarse=1.0, coarse2=0.2, notebook=True, return_rij=False, **kwargs):
    """
    Returns two-point velocity autocorrelation tensor, and autocorrelation functions.
    Uses the x-component of velocity. (CAUTION required for unisotropic flows)

    Pope Eq. 6.44
    Parameters
    ----------
    udata: 5D or 4D numpy array, 5D if the no. of spatial dimensions is 3. 4D if the no. of spatial dimensions is 2.
          ... (ux, uy, uz) or (ux, uy)
          ... ui has a shape (height, width, depth, duration) or (height, width, depth) (3D)
          ... ui may have a shape (height, width, duration) or (height, width) (2D)


    Returns
    -------
    rij: nd array (r, t, i, j)
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
            u2_avg = np.nanmean(udata ** 2, axis=tuple(range(dim + 1)))  # spatial average
            if dim == 2:
                x, y = r[0], r[1]
            elif dim == 3:
                x, y, z = r[0], r[1], r[2]
            r2_norm = np.zeros_like(x)
            for k in range(dim):
                r2_norm += r[k] ** 2
            r_norm = np.sqrt(r2_norm)
            Rij_value = u2_avg[t] * (
                        g(r_norm, t) * kronecker_delta_delta(i, j) + (f(r_norm, t) - g(r_norm, t)) * r[i] * r[j] / (
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
    ... the returned objects are functions NOT arrays

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
    f, g: long./trans. autocorrelation functions
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
    Returns lists of INTERPOLATED autocorrelation function values
    ... outputs of get_two_point_vel_corr_iso() may contain np.nan and np.inf which could be troublesome
    ... this method gets rid of these values, and interpolates the missing values (third-order spline)
    ... the arguments should have a shape (# of data points, duration)
    ... this method conducts intepoltation at given time respectively
    because 2D interpolation often raises an error due to the discontinuity of the autocorrelation functions
    along the temporal axis.

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

    Returns
    -------
    fs: list with length = duration = f_long.shape[-1]
        list of interpolated longitudinal structure function
    gs: list with length = duration = f_long.shape[-1]
        list of interpolated transverse structure function
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
    Returns autocorrelation tensor with isotropy assumption
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
                    g(r_norm, t) * kronecker_delta_delta(i, j) + (f(r_norm, t) - g(r_norm, t)) * r[i] * r[j] / (
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
                           x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                           p=2, nr=None, nd=10000, mode='long',
                           notebook=True):
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
        rs_ = np.linspace(dx * 2, min([width, height]) * 1, nr)
    elif dim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(x_grid), np.max(x_grid), np.min(y_grid), np.max(y_grid), np.min(
            z_grid), np.max(z_grid)
        width, height, depth = xmax - xmin, ymax - ymin, zmax - zmin
        rs_ = np.linspace(dx, min([width, height, depth]), nr)
    rs = np.empty(nd)
    Dijks_ = np.empty(nd)
    for t in tqdm(list(range(t0, t1)), desc='struc. func. time'):
        for i, r in enumerate(tqdm(rs_, desc='struc. func. r-loop')):
            for j in range(nd):
                if dim == 2:
                    while not is_R1_reasonable:
                        # Randomly pick a point in space, call it R0
                        X0, Y0 = np.random.random() * width + xmin, np.random.random() * height + ymin
                        R0 = np.asarray([X0, Y0])
                        X0_ind, _ = find_nearest(x_grid[0, :], X0)
                        Y0_ind, _ = find_nearest(y_grid[:, 0], Y0)
                        R0 = np.asarray([x_grid[0, X0_ind], y_grid[Y0_ind, 0]])
                        # Randomly pick another point in space, call it R1
                        theta = 2 * np.pi * np.random.random()
                        X1, Y1 = X0 + r * np.cos(theta), Y0 + r * np.sin(theta)
                        R1 = np.asarray([X1, Y1])
                        is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin
                    X1_ind, _ = find_nearest(x_grid[0, :], X1)
                    Y1_ind, _ = find_nearest(y_grid[:, 0], Y1)
                    R1 = np.asarray([x_grid[0, X1_ind], y_grid[Y1_ind, 0]])
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
                        R1 = np.asarray([X1, Y1])
                        is_R1_reasonable = X1 < xmax and X1 > xmin and Y1 < ymax and Y1 > ymin and Z1 < zmax and Z1 > zmin
                    X1_ind, _ = find_nearest(x_grid[0, :, 0], X1)
                    Y1_ind, _ = find_nearest(y_grid[:, 0, 0], Y1)
                    Z1_ind, _ = find_nearest(z_grid[0, 0, :], Z1)
                    R1 = np.asarray([x_grid[0, X1_ind, 0], y_grid[Y1_ind, 0, 0], z_grid[0, 0, Z1_ind]])

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

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, Dijks, Dijk_errs, rrs_scaled, Dijks_scaled, Dijk_errs_scaled


def scale_raw_structure_funciton_long(rrs, Dxxs, Dxx_errs, epsilon, nu=1.004, p=2):
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
    Dxx_errs_s: numpy array
        Scaled DLL error
    """
    if type(epsilon) == list:
        epsilon = np.asarray(epsilon)

    eta = compute_kolmogorov_lengthscale_simple(epsilon, nu)
    if type(epsilon) == np.ndarray:
        Dxxs_s, Dxx_errs_s, rrs_s = np.empty_like(Dxxs), np.empty_like(Dxxs), np.empty_like(Dxxs)
        for t in list(range(len(epsilon))):
            Dxxs_s[:, t] = Dxxs[:, t] / (epsilon[t] * rrs[:, t]) ** (p / 3.)
            Dxx_errs_s[:, t] = Dxx_errs[:, t] / (epsilon[t] * rrs[:, t]) ** (p / 3.)
            rrs_s[:, t] = rrs[:, t] / eta[t]
    else:
        Dxxs_s = Dxxs / (epsilon * rrs) ** (p / 3.)
        Dxx_errs_s = Dxx_errs / (epsilon * rrs) ** (p / 3.)
        rrs_s = rrs / eta
    return rrs_s, Dxxs_s, Dxx_errs_s


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
    ... Use the first p*100 % of the autocorrelation values.
    ... Designed to use the results of get_two_point_vel_corr_iso().

    Parameters
    ----------
    r_long: numpy 2d array with shape (no. of elements, duration)
        ... r for longitudinal autoorrelation function
    f_long: numpy 2d array with shape (no. of elements, duration)
        ... longitudinal autoorrelation values
    r_tran: numpy 2d array with shape (no. of elements, duration)
        ... r for longitudinal autoorrelation function
    g_tran: numpy 2d array with shape (no. of elements, duration)
        ... longitudinal autoorrelation values

    Returns
    -------
    lambda_f: numpy 2d array with shape (duration, )
        Longitudinal Taylor microscale
    lambda_g: numpy 2d array with shape (duration, )
        Transverse Taylor microscale
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
    Assumes isotropy and a full 1D energy spectrum. Pope 6.260.

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
        L_iso_spec.append(np.pi / (2. * u2_irms[t]) * np.trapz(e_k[1:, t] / k[1:, t], k[1:, t]))
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
    Return integral velocity scale which is identical to u' (characteristic velocity)
    See get_characteristic_velocity()

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
    eta: numpy array
        kolmogorov length scale
    """
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    eta = (nu ** 3 / epsilon) ** 0.25
    return eta


########## ALL SCALES (LENGTH, VELOCITY, TIME)  ##########
def get_integral_scales_all(udata, dx, dy, dz=None, nu=1.004):
    """
    Returns integral scales (related to LARGE EDDIES)

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
    L_le: 1d array
    u_L: 1d array
    tau_L: 1d array
    """
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    energy_avg = get_spatial_avg_energy(udata)[0]
    L_le = get_integral_scale_large_eddy(udata, epsilon)
    u_L = energy_avg ** 0.5
    tau_L = L_le / u_L
    return L_le, u_L, tau_L


def get_taylor_microscales_all(udata, r_long, f_long, r_tran, g_tran):
    """
    Returns Taylor microscales
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
    Returns Taylor microscales using isotropic formulae
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
    Returns Kolmogorov scales

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
    Returns Taylor reynolds number (Pope 6.63) using isotropic formulae

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
    ... In turbulence, skewness is approximately -0.4 according to experiments

    Parameters
    ----------
    udata
    x0
    x1
    y0
    y1
    z0
    z1
    t0
    t1

    Returns
    -------
    skewness: 1d array
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
    ... In turbulence, skekurotsiswness is approximately 7.2 according to experiments
    Parameters
    ----------
    udata
    x0
    x1
    y0
    y1
    z0
    z1
    t0
    t1

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


########## Sample velocity field ##########
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
    udata: (ux, uy)

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


def rankine_vortex_line_3d_gen(xx, yy, zz, x0=0, y0=0, z0=0, gamma=1., a=1.,
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
    udata: (ux, uy, uz)

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


def get_sample_turb_field_3d(return_coord=True):
    """
    Returns udata=(ux, uy, uz) of a slice of isotropic, homogeneous turbulence data (DNS, JHTD)

    Parameters
    ----------
    return_coord

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
    datapath = os.path.join(pdir, 'velocity_ref/isoturb_slice2.h5')
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


########## turbulence related stuff  ##########
def get_rescaled_energy_spectrum_saddoughi():
    """
    Returns values to plot rescaled energy spectrum from Saddoughi (1992)

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
    Call get_rescaled_energy_spectra_jhtd for scaled energy spectrum.

    Returns
    -------
    datadict: dict
        data stored in jhtd_e_specs.h5 is stored: k, ek
    """
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    datapath = faqm_dir + '/velocity_ref/jhtd_e_specs.h5'

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
    Call get_energy_spectra_jhtd for raw energy spectrum.

    Returns
    -------
    datadict: dict
        data stored in jhtd_e_specs.h5 is stored: Scaled k, Scaled ek
    """
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    datapath = faqm_dir + '/velocity_ref/jhtd_e_specs.h5'

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
    r_scaled: nd array
    dll: nd array

    """
    tflow_dir = os.path.split(os.path.realpath(__file__))[0]
    if p == 2:
        datapath = tflow_dir + '/velocity_ref/sv_struc_func.h5'
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
                        x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, inc=10, notebook=True, **kwargs):
    """
    (For 3D velocity field data)
    Given a path to a hdf5 file which stores udata,
    it returns the result of functions without loading an entire udata onto RAM
    ... example:
        results = process_large_udata(udatapath, func=vel.get_spatial_avg_enstrophy,
                                              inc=inc, dx=dx, dy=dy, dz=dz)
        enst, enst_err = result


    Parameters
    ----------
    udatapath
    func: function
        ... a function to compute a quantity you desire from udata
    t0: int
    t1: int, default: None
        ... If t1 were not given, it processes from the beginning to the end.
    inc: int, default: 10
        ... temporal increment at which the function is called
    kwargs: keyword arguments passed to the function

    Returns
    -------
    datalist: list
        ... results of the outputs of the function.

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

    for i, t in enumerate(tqdm(range(t0, t1, inc))):
        udata, xx, yy, zz = get_udata_from_path(udatapath, return_xy=True, t0=t, t1=t + 1,
                                                x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, verbose=False)
        if i == 0:
            datalist = []
            result = func(udata, **kwargs)
            if type(result) is tuple:
                n_output = len(result)  # number of objects returned by a function
                for j in range(n_output):
                    shape = list(result[j].shape)
                    if shape[-1] == 1:
                        shape[-1] *= n
                    datalist.append(np.empty(shape))
            else:
                n_output = 1
                for j in range(n_output):
                    datalist.append(np.empty(result.shape))
        else:
            result = func(udata, **kwargs)
        for j in range(n_output):
            if datalist[j].shape[-1] == n:
                datalist[j][..., i] = result[j][..., 0]
            else:
                if i == 0:
                    datalist[j] = result[j]
    if notebook:
        from tqdm import tqdm as tqdm
    return datalist


def get_time_avg_energy_from_udatapath(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                                       t0=0, t1=None, slicez=None, inc=1, thd=np.inf, fill_value=np.nan,
                                       notebook=True):
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

    Returns
    -------

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
            height, width = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
            dim = 2
        if t1 is None:
            t1 = duration

        if dim == 3:
            if slicez is None:
                eavg = np.zeros((height, width, depth))
                counters = np.zeros((height, width, depth))
                for t in tqdm(range(t0, t1, inc)):
                    energy_inst = (f['ux'][y0:y1, x0:x1, z0:z1, t] ** 2 + f['uy'][y0:y1, x0:x1, z0:z1, t] ** 2 + f[
                                                                                                                     'uz'][
                                                                                                                 y0:y1,
                                                                                                                 x0:x1,
                                                                                                                 z0:z1,
                                                                                                                 t] ** 2) / 2.
                    energy_inst[energy_inst > thd] = fill_value
                    eavg = np.nansum(np.stack((eavg, energy_inst)), 0)
                    counters += ~np.isnan(energy_inst)
                eavg /= counters
            else:
                eavg = np.zeros((height, width))
                counters = np.zeros((height, width))
                for t in tqdm(range(t0, t1, inc)):
                    energy_inst = (f['ux'][y0:y1, x0:x1, slicez - z0, t] ** 2 + f['uy'][y0:y1, x0:x1, slicez - z0,
                                                                                t] ** 2 + f['uz'][y0:y1, x0:x1,
                                                                                          slicez - z0, t] ** 2) / 2.
                    energy_inst[energy_inst > thd] = fill_value
                    eavg = np.nansum(np.stack((eavg, energy_inst)), 0)
                    counters += ~np.isnan(energy_inst)
                eavg /= counters
        elif dim == 2:
            eavg = np.zeros((height, width))
            counters = np.zeros((height, width))
            for t in tqdm(range(t0, t1, inc)):
                energy_inst = (f['ux'][y0:y1, x0:x1, t] ** 2 + f['uy'][y0:y1, x0:x1, t] ** 2) / 2.
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
        ... if one wants complete statistics, use inc=1 but this is often overkill.

    Returns
    -------
    array in which a fraction of nans in udata is included
    """
    shape = get_udata_dim(udatapath)
    dim = shape[0]
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
        print('... 2D udata. Returns the ratio of the no. of nans to the column length along x')
        axis = 'x'
    else:
        print('... 3D udata. Returns the ratio of the no. of nans to the number of elements on the plane along %s' % axis)

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
                                          notebook=True):
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

    Returns
    -------

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
            height, width = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
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
                    enst_inst = get_enstrophy(udata, dx, dy, dz)[..., 0]
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
                    enst_inst = get_enstrophy(udata, dx, dy, dz)[:, :, 1, 0]
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
                enst_inst = get_enstrophy(udata, dx, dy)[:, :, 0]
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
            height, width = f['ux'][y0:y1, x0:x1, z0:z1, 0].shape
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
                    ux_inst, uy_inst, uz_inst = f['ux'][y0:y1, x0:x1, z0:z1, t], f['uy'][y0:y1, x0:x1, z0:z1, t], f[
                                                                                                                      'uz'][
                                                                                                                  y0:y1,
                                                                                                                  x0:x1,
                                                                                                                  z0:z1,
                                                                                                                  t]
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
        elif dim == 2:
            ux_m, uy_m = np.zeros((height, width)), np.zeros((height, width))
            counters_ux, counters_uy = np.zeros((height, width)), np.zeros((height, width))
            for t in tqdm(range(t0, t1, inc)):
                ux_inst, uy_inst = f['ux'][y0:y1, x0:x1, z0:z1, t], f['uy'][y0:y1, x0:x1, z0:z1, t]
                ux_inst[ux_inst > thd] = fill_value
                uy_inst[uy_inst > thd] = fill_value

                ux_m = np.nansum(np.stack((ux_m, ux_inst)), 0)
                uy_m = np.nansum(np.stack((uy_m, uy_inst)), 0)
                counters_ux += ~np.isnan(ux_inst)
                counters_uy += ~np.isnan(uy_inst)
            ux_m /= counters_ux
            uy_m /= counters_uy
    if notebook:
        from tqdm import tqdm
    u_m = np.stack((ux_m, uy_m, uz_m))
    return u_m


def export_raw_file_from_dpath(udatapath, func=get_energy, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                               t0=0, t1=None, inc=1, savedir=None, dtype='uint32', thd=None,
                               interpolate=None, **kwargs):
    """
    Intended for 4D visualization. e.g. use ORS Dragonfly for visualiztion
    Exports a raw file of the output of a given function (default: get_energuy) at each time step.

    ... Argument: path to udata
    ... Philosophy: Load 3D data at an instant of time, compute the quantity of interest. Export the 3D array. Repeat.
    ...

    Parameters
    ----------
    udatapath
    func
    x0
    x1
    y0
    y1
    z0
    z1
    t0
    t1
    inc
    savedir
    dtype
    thd
    interpolate
    kwargs

    Returns
    -------

    """
    udatadir, udaname = os.path.split(udatapath)
    pdir = os.path.split(udatadir)[0]
    if savedir is None:
        savedir = os.path.join(pdir, 'raw_files')
        savedir = os.path.join(os.path.join(savedir, udaname[:-3]), func.__name__)

    with h5py.File(udatapath) as f:
        shape = f['ux'].shape
        duration = shape[-1]
        savedir += '_%03dx%03dx%03dx%05dx_inc%d' % (shape[0], shape[1], shape[2], shape[3], inc)
        if interpolate is not None:
            savedir += '_%s' % interpolate

    if t1 is None:
        t1 = duration

    for i, t in enumerate(tqdm(range(t0, t1, inc), desc='saving raw files')):
        udata, xx, yy, zz = get_udata_from_path(udatapath, return_xy=True,
                                                x0=x0, x1=x1, y0=y0, y1=y1,
                                                z0=z0, z1=z1,
                                                t0=t, t1=t + 1, verbose=False)
        dx, dy, dz = get_grid_spacing(xx, yy, zz)
        # Perform 3D interpolation
        if interpolate is not None:
            udata = clean_udata(udata, xx, yy, zz, verbose=False, method=interpolate)

            # Save energy
        data2save = func(udata, **kwargs)[..., 0]  # save 3d array

        if thd is not None:
            keep = data2save < thd
            data2save[~keep] = 0

        data2save = data2save

        savepath = os.path.join(savedir, 't%05d.raw' % t)

        if not os.path.exists(savedir):
            os.makedirs(savedir)
        data2save.astype(dtype).tofile(savepath)


def export_raw_file(data2save, savepath, dtype='uint32', thd=np.inf, interpolate=None, fill_value=np.nan,
                    contrast=True, log10=False, **kwargs):
    """
    A helper to export a raw file from a numpy array with inteprolation feature

    Parameters
    ----------
    data2save
    savepath
    dtype
    thd: float default: np.inf
        ... data > thd will be replaced by fill_value
    interpolate: str, default None
        ... choose from 'idw' and 'localmean'
            ... if specified, it replaces np.nan in array using replace_nan(...)
    fill_value: float, default: np.nan
        ... value that is used to fill in data where data value > thd
    kwargs: passed to replaced_nan(...)

    Returns
    -------*

    """
    shape = np.asarray(data2save).shape

    if savepath[-4:] == '.raw':
        savepath = savepath[:-4]
    for i in shape:
        savepath += '_%03dx' % i

    if thd is not None:
        fill = data2save > thd
        data2save[fill] = fill_value  # this fills values. np.nan won't get filled

    if interpolate is not None:
        data2save = replace_nans(data2save, method=interpolate, **kwargs)

    if contrast:
        try:
            maxint = 2 ** int(dtype[-2:]) - 1
        except:
            maxint = 2 ** int(dtype[-1:]) - 1
        max_value, min_value = np.nanmax(data2save), np.nanmin(data2save)
        contrast_value = maxint / (max_value - min_value)
        data2save = (data2save - min_value) * contrast_value
        print('intensity was enhanced by %.2f' % contrast_value)
    else:
        contrast_value = 1

    if log10:
        data2save = np.log10(data2save)

    savedir = os.path.split(savepath)[0]
    savepath += '%s_thd%06.f_intp%s_ctr%06.f_log%r.raw' % (dtype, thd, interpolate, contrast_value, log10)

    if not os.path.exists(savedir):
        os.makedirs(savedir)
    data2save.astype(dtype).tofile(savepath)


########## FFT tools ########
def get_window_radial(xx, yy, zz=None, wtype='hamming', rmax=None, duration=None,
                      x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                      n=500):
    """
    General method to get a window with shape (xx.shape[:], duration) or (xx.shape[:]) if duration is None
    ... Window types:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
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
    if window is 'rectangle' or window is None:
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
    ... griddata performs a better interpolation but this method is much faster.
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
    U: array-like
    cutoffU: float
        if |value| > cutoff, this method considers those values unphysical.
    fill_value:


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
    U_filled: numpy array

    """
    U_masked = ma.array(U, mask=mask)
    U_filled = ma.filled(U_masked, fill_value)  # numpy array. This is NOT a masked array.

    return U_filled


def clean_udata(udata, mask=None,
                method='idw', max_iter=50, tol=0.05, kernel_radius=2, kernel_sigma=2,
                cutoff=np.inf, showtqdm=True, verbose=False, notebook=True):
    """
    ND interpolation using direct convolution (replac_nan(...))

    Parameters
    ----------
    udata
    xx
    yy
    zz
    cutoffU
    fill_value
    verbose
    method
    notebook

    Returns
    -------

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    dim = udata.shape[0]

    if dim == 2:
        ncomp, height, width, duration = udata.shape
    else:
        ncomp, height, width, depth, duration = udata.shape

    if mask is not None:
        udata[mask] = np.nan
    # manual cleaning
    udata[np.logical_or(np.abs(udata) > cutoff, np.isinf(udata))] = np.nan

    nnans = np.count_nonzero(np.isnan(udata))
    if nnans == 0:  # udata is already clean (No nan values and no values beyond cutoff)
        return udata

    # Initialization of an inpainted field
    udata_i = np.empty_like(udata)

    for t in tqdm(range(duration), disable=not showtqdm):
        for i in range(ncomp):
            udata_i[i, ..., t] = replace_nans(udata[i, ..., t], max_iter=max_iter, tol=tol, kernel_radius=kernel_radius,
                                              kernel_sigma=kernel_sigma, method=method, showtqdm=verbose)
    if notebook:
        from tqdm import tqdm as tqdm

    return udata_i


def clean_udata_old(udata, xx, yy, zz=None, cutoffU=2000, fill_value=np.nan, verbose=False, method='nearest',
                    use_griddata2d=True, usetqdm=True, notebook=True):
    """
    ND interpolation using scipy.griddata
    ... griddata is notoriously known to be slow.
    ... Use method='nearest' for most of the time just to fill the missing values in udata.

    Parameters
    ----------
    udata
    xx
    yy
    zz
    cutoffU
    fill_value
    verbose
    method
    notebook

    Returns
    -------

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    dim = udata.shape[0]

    if dim == 2:
        ncomp, height, width, duration = udata.shape
    else:
        ncomp, height, width, depth, duration = udata.shape

    udata_i = np.empty_like(udata)
    for t0 in tqdm(range(duration), disable=~usetqdm):
        for i in range(ncomp):
            if i == 0 and verbose:
                mask = get_mask_for_unphysical(udata[i, ..., t0], cutoffU=cutoffU, fill_value=fill_value, verbose=True)
            else:
                mask = get_mask_for_unphysical(udata[i, ..., t0], cutoffU=cutoffU, fill_value=fill_value, verbose=False)
            if dim == 3:
                if method != 'nearest' and use_griddata2d:
                    ui_grid_d = np.empty(udata.shape[1:-1])
                    for z0 in range(depth):
                        mask_slice = mask[:, :, z0]

                        x, y = xx[..., z0][~mask_slice].flatten(), yy[..., z0][~mask_slice].flatten()
                        pts = list(zip(x, y))
                        ui_values = udata[i, :, :, z0, t0][~mask_slice].flatten()
                        if np.sum(~mask_slice) > 3:
                            ui_grid_d[..., z0] = griddata(pts, ui_values, (xx[..., z0], yy[..., z0]), method=method)
                        else:
                            # if the slice of the data has no values (i.e. no tracable tracks in this x-y plane),
                            # just fill the plane with 0- users may crop this regions later. Also, it will prevent
                            # fft to fail.
                            ui_grid_d[..., z0][mask_slice] = 0
                            ui_grid_d[..., z0][~mask_slice] = udata[i, :, :, z0, t0][~mask_slice]
                else:
                    x, y, z = xx[~mask].flatten(), yy[~mask].flatten(), zz[~mask].flatten()

                    pts = list(zip(x, y, z))
                    ui_values = udata[i, ..., t0][~mask].flatten()

                    ui_grid_d = griddata(pts, ui_values, (xx, yy, zz), method=method)
            elif dim == 2:
                x, y = xx[~mask].flatten(), yy[~mask].flatten()

                pts = list(zip(x, y))
                ui_values = udata[i, ..., t0][~mask].flatten()

                ui_grid_d = griddata(pts, ui_values, (xx, yy), method=method)

            udata_i[i, ..., t0] = ui_grid_d[...]

    if notebook:
        from tqdm import tqdm as tqdm

    return udata_i


def clean_udata_lin_int(udata, xx, yy, zz, cutoffU=2000, fill_value=np.nan, verbose=False, notebook=True):
    """
    Depreciated. Attempts 3D linear interpolation using scipy.interpolate.LinearNDInterpolator
    ... This is unbearably slow.
    ... If you'd want to replace nans by linear interpolation, use clean_udata(method='linear')


    Parameters
    ----------
    udata
    xx
    yy
    zz
    cutoffU
    fill_value
    verbose
    method
    notebook

    Returns
    -------
    udata_i

    """
    from scipy.interpolate import LinearNDInterpolator

    print('Depreciated! Use clean_udata(..., method=\'linear\')')

    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    udata = fix_udata_shape(udata)
    dim = udata.shape[0]

    if dim == 2:
        ncomp, height, width, duration = udata.shape
    else:
        ncomp, height, width, depth, duration = udata.shape

    udata_i = np.empty_like(udata)
    for t0 in tqdm(range(duration)):
        for i in range(ncomp):
            if i == 0 and verbose:
                mask = get_mask_for_unphysical(udata[i, ..., t0], cutoffU=cutoffU, fill_value=fill_value, verbose=True)
            else:
                mask = get_mask_for_unphysical(udata[i, ..., t0], cutoffU=cutoffU, fill_value=fill_value, verbose=False)
            x, y, z = xx[~mask].flatten(), yy[~mask].flatten(), zz[~mask].flatten()

            pts = list(zip(x, y, z))
            ui_values = udata[i, ..., t0][~mask].flatten()
            f = LinearNDInterpolator(pts, ui_values)
            ui_grid_d = f(xx, yy, zz)

            udata_i[i, ..., t0] = ui_grid_d[...]

    if notebook:
        from tqdm import tqdm as tqdm

    return udata_i


def find_crop_no(udata, tol=0.2):
    """
    Returns a how many pixels should be cropped to ignore missing values

    Parameters
    ----------
    udata
    tol

    Returns
    -------

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


def replace_nans(array, max_iter=50, tol=0.05, kernel_radius=2, kernel_sigma=2, method='idw',
                 notebook=True, verbose=False, showtqdm=True):
    """Replace NaN elements in an array using an iterative image inpainting algorithm.
    The algorithm is the following:
    1) For each element in the input array, replace it by a weighted average
    of the neighbouring elements which are not NaN themselves. The weights depends
    of the method type. If ``method=localmean`` weight are equal to 1/( (2*kernel_size+1)**2 -1 )
    2) Several iterations are needed if there are adjacent NaN elements.
    If this is the case, information is "spread" from the edges of the missing
    regions iteratively, until the variation is below a certain threshold.

    - Generalized to nD array by Takumi Matsuzawa (UChicago) 2020/02/20
    ... one could use array multiplication over loops for better runtime.

    Parameters
    ----------
    array : nd np.ndarray
    an array containing NaN elements that have to be replaced
    max_iter : int
    the number of iterations
    kernel_size : int
    the size of the kernel, default is 1
    method : str
    the method used to replace invalid values. Valid options are
    `localmean`, 'idw' (Gaussian kernel).
    Returns
    -------
    filled : nd np.ndarray
    a copy of the input array, where NaN elements have been replaced.
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
    elif method == 'idw':
        kernel = makeGaussianKernel(kernel_size, kernel_sigma, dim)
    #         print(kernel.shape, 'kernel')
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

def clean_data(data, mask=None,
                method='idw', max_iter=50, tol=0.05, kernel_radius=2, kernel_sigma=2,
                cutoff=np.inf, showtqdm=True, verbose=False, notebook=True):
    """
    ND interpolation using direct convolution (replac_nan(...))

    Parameters
    ----------
    udata
    xx
    yy
    zz
    cutoffU
    fill_value
    verbose
    method
    notebook

    Returns
    -------

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        # print('Using tqdm_notebook. If this is a mistake, set notebook=False')
    else:
        from tqdm import tqdm

    if mask is not None:
        data[mask] = np.nan
    # manual cleaning
    data[np.logical_or(np.abs(data) > cutoff, np.isinf(data))] = np.nan

    nnans = np.count_nonzero(np.isnan(data))
    if nnans == 0:  # udata is already clean (No nan values and no values beyond cutoff)
        return data

    # Initialization of an inpainted field

    data = replace_nans(data, max_iter=max_iter, tol=tol, kernel_radius=kernel_radius,
                                              kernel_sigma=kernel_sigma, method=method, showtqdm=verbose)
    if notebook:
        from tqdm import tqdm as tqdm

    return data

# FUNCTIONS FOR STB DATA ANALYSIS
def get_center_of_energy(dpath, inc=10, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    Returns the center of energy (essentially the center of the blob)- (xc, yc, zc)
    - It computes the time-averaged energy from udatapath
    - Then, it computes the center of energy like the center of mass.
    (i.e. the energy density serves as a weight function)

    Parameters
    ----------
    dpath
    inc
    x0
    x1
    y0
    y1
    z0
    z1

    Returns
    -------

    """
    # Load dummy udata
    udata, xx, yy, zz = get_udata_from_path(dpath, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, t0=0, t1=1,
                                                return_xy=True, verbose=False)
    etavg = get_time_avg_energy_from_udatapath(dpath, inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
    xc = np.nansum(etavg * xx) / np.nansum(etavg)
    yc = np.nansum(etavg * yy) / np.nansum(etavg)
    zc = np.nansum(etavg * zz) / np.nansum(etavg)
    center_of_energy = np.asarray([xc, yc, zc])
    return center_of_energy

    # # plotting usual stuff # REQUIRES graph module. Ask Takumi for details or check out takumi's git on immense
    # def plot_energy_spectra(udata, dx, dy, dz=None, x0=0, x1=None, y0=0, y1=None, window='flattop', epsilon_guess=10**5, nu=1.004, label='',
    #                             plot_e22=False, plot_ek=False, fignum=1, t0=0, legend=True, loc=3):
    #     """
    #     A method to quickly plot the 1D energy spectra
    #     Parameters
    #     ----------
    #     udata
    #     dx
    #     dy
    #     dz
    #     x0
    #     x1
    #     y0
    #     y1
    #     window
    #     epsilon_guess
    #     nu
    #     plot_e22
    #     plot_ek
    #     fignum
    #     t0
    #     legend
    #
    #     Returns
    #     -------
    #     fig1, (ax1, ax2)
    #
    #     """
    #     __fontsize__ = 25
    #     __figsize__ = (16, 8)
    #     # See all available arguments in matplotlibrc
    #     params = {'figure.figsize': __figsize__,
    #               'font.size': __fontsize__,  # text
    #               'legend.fontsize': 18,  # legend
    #               'axes.labelsize': __fontsize__,  # axes
    #               'axes.titlesize': __fontsize__,
    #               'xtick.labelsize': __fontsize__,  # tick
    #               'ytick.labelsize': __fontsize__,
    #               'lines.linewidth': 8,
    #               'axes.titlepad': 10}
    #     graph.update_figure_params(params)
    #
    #     ax1_ylabel = '$E_{11}$ ($mm^3/s^2$)'
    #     ax2_ylabel = '$E_{11} / (\epsilon\\nu^5)^{1/4}$'
    #
    #
    #     eiis, err, k11 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    #     if epsilon_guess is not None:
    #         e11_s, k11_s = scale_energy_spectrum(eiis[0, ...], k11, epsilon=epsilon_guess, nu=nu)
    #         e22_s, k22_s = scale_energy_spectrum(eiis[1, ...], k11, epsilon=epsilon_guess, nu=nu)
    #         eiis_s = np.stack((e11_s, e22_s))
    #         epsilon = epsilon_guess
    #     else:
    #         eiis_s, _, k11_s = get_1d_rescaled_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    #         epsilon = get_epsilon_using_sij(udata, dx, dy, dz, nu=nu)[t0]
    #     fig1, ax1 = graph.plot(k11[1:], kolmogorov_53_uni(k11[1:], epsilon, c=0.5), label='$C_1\epsilon^{2/3}\kappa^{-5/3}$', fignum=1, subplot=121, color='k')
    #     fig1, ax2 = graph.plot_saddoughi(fignum=fignum, subplot=122, color='k', label='Scaled $E_{11}$ (SV 1994)')
    #
    #     fig1, ax1 = graph.plot(k11[1:], eiis[0, 1:, t0], label='$E_{11}$' + label, fignum=fignum, subplot=121)
    #     fig1, ax2 = graph.plot(k11_s[1:], eiis_s[0, 1:, t0], label='Scaled $E_{11}$' + label, fignum=fignum, subplot=122)
    #
    #     if plot_e22:
    #         fig1, ax1 = graph.plot(k11[1:], eiis[1, 1:, t0], label='$E_{22}$' + label, fignum=fignum, subplot=121)
    #         fig1, ax2 = graph.plot(k11_s[1:], eiis_s[1, 1:, t0], label='Scaled $E_{22}$' + label, fignum=fignum, subplot=122)
    #         ax1_ylabel = ax1_ylabel[:-13] + ', $E_{22}$ ($mm^3/s^2$)'
    #         ax2_ylabel = ax2_ylabel + ', $E_{22} / (\epsilon\\nu^5)^{1/4}$'
    #
    #     if plot_ek:
    #         ek, _, kk = get_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    #         ek_s, kk_s = scale_energy_spectrum(ek, kk, epsilon=epsilon, nu=nu)
    #         fig1, ax1 = graph.plot(kk[:, t0], ek[:, t0], label='$E$' + label, fignum=fignum, subplot=121)
    #         fig1, ax2 = graph.plot(kk_s[:, t0], ek_s[:, t0], label='Scaled $E$' + label, fignum=fignum, subplot=122)
    #
    #         ax1_ylabel = ax1_ylabel[:-13] + ', $E$ ($mm^3/s^2$)'
    #         ax2_ylabel = ax2_ylabel + ', $E / (\epsilon\\nu^5)^{1/4}$'
    #
    #     graph.tologlog(ax1)
    #     graph.tologlog(ax2)
    #     if legend:
    #         ax1.legend(loc=loc)
    #         ax2.legend(loc=loc)
    #
    #     graph.labelaxes(ax1, '$\kappa$ ($mm^{-1}$)', ax1_ylabel)
    #     graph.labelaxes(ax2, '$\kappa \eta $ ', ax2_ylabel)
    #
    #     # graph.setaxes(ax1, 10 ** -1.5, 10 ** 0.8, 10 ** 0.3, 10 ** 5.3)
    #     graph.setaxes(ax2, 10 ** -3.8, 2, 10 ** -3.5, 10 ** 6.5)
    #
    #     fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     return fig1, (ax1, ax2)
    #
    # def plot_energy_spectra_w_energy_heatmap(udata, dx, dy, dz=None, x0=0, x1=None, y0=0, y1=None, window='flattop', epsilon_guess=10**5, nu=1.004, label='',
    #                             plot_e22=False, plot_ek=False, fignum=1, t0=0, legend=True, loc=3, crop_edges=5, yoffset_box=20, sb_txtloc=(-0.1, 0.4),
    #                                          vmax=10**4.8):
    #     """
    #     A method to quickly plot the energy spectra (snapshot) and time-averaged energy
    #
    #     Parameters
    #     ----------
    #     udata
    #     dx
    #     dy
    #     dz
    #     x0
    #     x1
    #     y0
    #     y1
    #     window
    #     epsilon_guess
    #     nu
    #     label
    #     plot_e22
    #     plot_ek
    #     fignum
    #     t0
    #     legend
    #     loc
    #     crop_edges
    #     yoffset_box
    #     sb_txtloc
    #
    #     Returns
    #     -------
    #
    #     """
    #     __fontsize__ = 20
    #     __figsize__ = (24, 8)
    #     # See all available arguments in matplotlibrc
    #     params = {'figure.figsize': __figsize__,
    #               'font.size': __fontsize__,  # text
    #               'legend.fontsize': 18,  # legend
    #               'axes.labelsize': __fontsize__,  # axes
    #               'axes.titlesize': __fontsize__,
    #               'xtick.labelsize': __fontsize__,  # tick
    #               'ytick.labelsize': __fontsize__,
    #               'lines.linewidth': 5,
    #               'axes.titlepad': 10}
    #     graph.update_figure_params(params)
    #
    #     dim = udata.shape[0]
    #     n = crop_edges
    #     if x1 is None:
    #         x1 = udata.shape[2]-1
    #     if y1 is None:
    #         y1 = udata.shape[1]-1
    #
    #     ax2_ylabel = '$E_{11}$ ($mm^3/s^2$)'
    #     ax3_ylabel = '$E_{11} / (\epsilon\\nu^5)^{1/4}$'
    #
    #     # Compute energy heatmap and draw rectangle
    #     energy_avg = np.nanmean(get_energy(udata), axis=dim)
    #     xx, yy = get_equally_spaced_grid(udata, spacing=dx)
    #     # Time-averaged Energy
    #     fig1, ax1, cc1 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], energy_avg[n:-n, n:-n],
    #                                          label='$\\frac{1}{2} \langle U_i   U_i \\rangle$ ($mm^2/s^2$)',
    #                                          vmin=0, vmax=vmax, fignum=fignum, subplot=131)
    #     graph.draw_box(ax1, xx, yy, yoffset=yoffset_box, sb_txtloc=sb_txtloc)
    #     graph.draw_rectangle(ax1, xx[y0, x0], yy[y0, x0], np.abs(xx[y0, x1]-xx[y0, x0]), np.abs(yy[y1, x0]-yy[y0, x0]), edgecolor='C0', linewidth=5)
    #
    #     # Coompute energy spectra
    #     eiis, err, k11 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    #     if epsilon_guess is not None:
    #         e11_s, k11_s = scale_energy_spectrum(eiis[0, ...], k11, epsilon=epsilon_guess, nu=nu)
    #         e22_s, k22_s = scale_energy_spectrum(eiis[1, ...], k11, epsilon=epsilon_guess, nu=nu)
    #         eiis_s = np.stack((e11_s, e22_s))
    #         epsilon = epsilon_guess
    #     else:
    #         eiis_s, _, k11_s = get_1d_rescaled_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    #         epsilon = get_epsilon_using_sij(udata, dx, dy, dz, nu=nu)[t0]
    #     fig1, ax2 = graph.plot(k11[1:], kolmogorov_53_uni(k11[1:], epsilon, c=0.5), label='$C_1\epsilon^{2/3}\kappa^{-5/3}$', fignum=fignum, subplot=132, color='k')
    #     fig1, ax3 = graph.plot_saddoughi(fignum=fignum, subplot=133, color='k', label='Scaled $E_{11}$ (SV 1994)')
    #
    #     fig1, ax2 = graph.plot(k11[1:], eiis[0, 1:, t0], label='$E_{11}$' + label, fignum=fignum, subplot=132)
    #     fig1, ax3 = graph.plot(k11_s[1:], eiis_s[0, 1:, t0], label='Scaled $E_{11}$' + label, fignum=fignum, subplot=133)
    #
    #     if plot_e22:
    #         fig1, ax2 = graph.plot(k11[1:], eiis[1, 1:, t0], label='$E_{22}$' + label, fignum=fignum, subplot=132)
    #         fig1, ax3 = graph.plot(k11_s[1:], eiis_s[1, 1:, t0], label='Scaled $E_{22}$' + label, fignum=fignum, subplot=133)
    #         ax2_ylabel = ax2_ylabel[:-13] + ', $E_{22}$ ($mm^3/s^2$)'
    #         ax3_ylabel = ax3_ylabel + ', $E_{22} / (\epsilon\\nu^5)^{1/4}$'
    #
    #     if plot_ek:
    #         ek, _, kk = get_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    #         ek_s, kk_s = scale_energy_spectrum(ek, kk, epsilon=epsilon, nu=nu)
    #         fig1, ax2 = graph.plot(kk[:, t0], ek[:, t0], label='$E$' + label, fignum=fignum, subplot=132)
    #         fig1, ax3 = graph.plot(kk_s[:, t0], ek_s[:, t0], label='Scaled $E$' + label, fignum=fignum, subplot=133)
    #
    #         ax2_ylabel = ax2_ylabel[:-13] + ', $E$ ($mm^3/s^2$)'
    #         ax3_ylabel = ax3_ylabel + ', $E / (\epsilon\\nu^5)^{1/4}$'
    #
    #     graph.tologlog(ax2)
    #     graph.tologlog(ax3)
    #     if legend:
    #         ax2.legend(loc=loc)
    #         ax3.legend(loc=loc)
    #
    #     graph.labelaxes(ax2, '$\kappa$ ($mm^{-1}$)', ax2_ylabel)
    #     graph.labelaxes(ax3, '$\kappa \eta $ ', ax3_ylabel)
    #
    #     # graph.setaxes(ax1, 10 ** -1.5, 10 ** 0.8, 10 ** 0.3, 10 ** 5.3)
    #     graph.setaxes(ax3, 10 ** -3.8, 2, 10 ** -3.5, 10 ** 6.5)
    #
    #     fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     return fig1, (ax1, ax2, ax3)
    #
    #
    #
    # def plot_energy_spectra_avg_w_energy_heatmap(udata, dx, dy, dz=None, x0=0, x1=None, y0=0, y1=None, window='flattop',
    #                                          epsilon_guess=10 ** 5, nu=1.004, label='',
    #                                          plot_e11=True, plot_e22=False, plot_ek=False, plot_kol=False, plot_sv=True,
    #                                          color_ref='k', alpha_ref=0.6,
    #                                          fignum=1, legend=True, loc=3,
    #                                          crop_edges=5, yoffset_box=20, sb_txtloc=(-0.1, 0.4), errorfill=True,
    #                                          vmin=0, vmax=10**5,
    #                                          figparams=None):
    #     """
    #     A method to quickly plot the energy spectra (Time-averaged) and time-averaged energy
    #
    #     Parameters
    #     ----------
    #     udata
    #     dx
    #     dy
    #     dz
    #     x0
    #     x1
    #     y0
    #     y1
    #     window
    #     epsilon_guess
    #     nu
    #     label
    #     plot_e22
    #     plot_ek
    #     fignum
    #     t0
    #     legend
    #     loc
    #     crop_edges
    #     yoffset_box
    #     sb_txtloc
    #
    #     Returns
    #     -------
    #
    #     """
    #     if figparams is None:
    #         __fontsize__ = 20
    #         __figsize__ = (24, 8)
    #         # See all available arguments in matplotlibrc
    #         params = {'figure.figsize': __figsize__,
    #                   'font.size': __fontsize__,  # text
    #                   'legend.fontsize': 18,  # legend
    #                   'axes.labelsize': __fontsize__,  # axes
    #                   'axes.titlesize': __fontsize__,
    #                   'xtick.labelsize': __fontsize__,  # tick
    #                   'ytick.labelsize': __fontsize__,
    #                   'lines.linewidth': 5,
    #                   'axes.titlepad': 10
    #                   }
    #     else:
    #         params = figparams
    #     graph.update_figure_params(params)
    #
    #     dim = udata.shape[0]
    #     n = crop_edges
    #
    #     ax2_ylabel = '$E_{11}$ ($mm^3/s^2$)'
    #     ax3_ylabel = '$E_{11} / (\epsilon\\nu^5)^{1/4}$'
    #
    #     # Compute energy heatmap and draw rectangle
    #     energy_avg = np.nanmean(get_energy(udata), axis=dim)
    #     xx, yy = get_equally_spaced_grid(udata, spacing=dx)
    #     # Time-averaged Energy
    #     fig1, ax1, cc1 = graph.color_plot(xx[n:-n, n:-n], yy[n:-n, n:-n], energy_avg[n:-n, n:-n],
    #                                       label='$\\frac{1}{2} \langle U_i   U_i \\rangle$ ($mm^2/s^2$)',
    #                                       vmin=vmin, vmax=vmax, fignum=fignum, subplot=131)
    #     graph.draw_box(ax1, xx, yy, yoffset=yoffset_box, sb_txtloc=sb_txtloc)
    #     graph.draw_rectangle(ax1, xx[y0, x0], yy[y0, x0], np.abs(xx[y0, x1] - xx[y0, x0]), np.abs(yy[y1, x0] - yy[y0, x0]),
    #                          edgecolor='C0', linewidth=5)
    #
    #     # Coompute energy spectra
    #     eiis_raw, err, k11 = get_1d_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    #     eiis = np.nanmean(eiis_raw, axis=2)
    #     eii_errs = np.nanstd(eiis_raw, axis=2)/2.
    #     if epsilon_guess is not None:
    #         e11_s, k11_s = scale_energy_spectrum(eiis_raw[0, ...], k11, epsilon=epsilon_guess, nu=nu)
    #         e22_s, k22_s = scale_energy_spectrum(eiis_raw[1, ...], k11, epsilon=epsilon_guess, nu=nu)
    #         eiis_s = np.stack((np.nanmean(e11_s, axis=1), np.nanmean(e22_s, axis=1)))
    #         eii_errs_s = np.stack((np.nanstd(e11_s, axis=1), np.nanstd(e22_s, axis=1)))/2.
    #         epsilon = epsilon_guess
    #     else:
    #         eiis_s_raw, _, k11_s = get_1d_rescaled_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    #         eiis_s = np.nanmean(eiis_s_raw, axis=2)
    #         eii_errs_s = np.nanstd(eiis_s_raw, axis=2)
    #         epsilon = np.nanmean(get_epsilon_using_sij(udata, dx, dy, dz, nu=nu))
    #     if plot_kol:
    #         fig1, ax2 = graph.plot(k11[1:], kolmogorov_53_uni(k11[1:], epsilon, c=0.5),
    #                                label='$C_1\epsilon^{2/3}\kappa^{-5/3}$',
    #                                fignum=fignum, subplot=132,
    #                                color=color_ref, alpha=alpha_ref, lw=params['lines.linewidth']*0.6)
    #     if plot_sv:
    #         fig1, ax3 = graph.plot_saddoughi(fignum=fignum, subplot=133,
    #                                          color=color_ref, alpha=alpha_ref, lw=params['lines.linewidth']*0.6,
    #                                          label='Scaled $E_{11}$ (SV 1994)')
    #     if plot_e11:
    #         if not errorfill:
    #             fig1, ax2 = graph.plot(k11[1:], eiis[0, 1:], label='$E_{11}$' + label, fignum=fignum, subplot=132, linewidth=10)
    #             fig1, ax3 = graph.plot(k11_s[1:], eiis_s[0, 1:], label='Scaled $E_{11}$' + label, fignum=fignum, subplot=133, linewidth=10)
    #         else:
    #             fig1, ax2, _ = graph.errorfill(k11[1:], eiis[0, 1:], eii_errs[0, 1:], label='$E_{11}$' + label, fignum=fignum, subplot=132)
    #             fig1, ax3, _ = graph.errorfill(k11_s[1:], eiis_s[0, 1:], eii_errs_s[0, 1:], label='Scaled $E_{11}$' + label, fignum=fignum, subplot=133)
    #
    #
    #     if plot_e22:
    #         if not errorfill:
    #             fig1, ax2 = graph.plot(k11[1:], eiis[1, 1:], label='$E_{22}$' + label, fignum=fignum, subplot=132)
    #             fig1, ax3 = graph.plot(k11_s[1:], eiis_s[1, 1:], label='Scaled $E_{22}$' + label, fignum=fignum, subplot=133)
    #         else:
    #             fig1, ax2, _ = graph.errorfill(k11[1:], eiis[1, 1:], eii_errs[1, 1:],label='$E_{22}$' + label, fignum=fignum, subplot=132)
    #             fig1, ax3, _ = graph.errorfill(k11_s[1:], eiis_s[1, 1:], eii_errs_s[1, 1:],label='Scaled $E_{22}$' + label, fignum=fignum, subplot=133)
    #         ax2_ylabel = ax2_ylabel[:-13] + ', $E_{22}$ ($mm^3/s^2$)'
    #         ax3_ylabel = ax3_ylabel + ', $E_{22} / (\epsilon\\nu^5)^{1/4}$'
    #
    #     if plot_ek:
    #         ek_raw, _, kk = get_energy_spectrum(udata, dx=dx, dy=dy, x0=x0, y0=y0, x1=x1, y1=y1, window=window)
    #         ek_s_raw, kk_s = scale_energy_spectrum(ek_raw, kk, epsilon=epsilon, nu=nu)
    #
    #         ek, ek_s = np.nanmean(ek_raw, axis=1), np.nanmean(ek_s_raw, axis=1)
    #         ek_err, ek_s_err = np.nanstd(ek_raw, axis=1), np.nanstd(ek_s_raw, axis=1)
    #         if not errorfill:
    #             fig1, ax2 = graph.plot(kk[:, 0], ek, label='$E$' + label, fignum=fignum, subplot=132, color='b', linestyle='--', linewidth=10)
    #             fig1, ax3 = graph.plot(kk_s[:, 0], ek_s, label='Scaled $E$' + label, fignum=fignum, subplot=133, color='b', linestyle='--', linewidth=10)
    #         else:
    #             fig1, ax2, _ = graph.errorfill(kk[:, 0], ek, ek_err, label='$E$' + label, fignum=fignum, subplot=132)
    #             fig1, ax3, _ = graph.errorfill(kk_s[:, 0], ek_s, ek_s_err, label='Scaled $E$' + label, fignum=fignum, subplot=133)
    #
    #
    #         ax2_ylabel = ax2_ylabel[:-13] + ', $E$ ($mm^3/s^2$)'
    #         ax3_ylabel = ax3_ylabel + ', $E / (\epsilon\\nu^5)^{1/4}$'
    #
    #     graph.tologlog(ax2)
    #     graph.tologlog(ax3)
    #     if legend:
    #         ax2.legend(loc=loc)
    #         ax3.legend(loc=loc)
    #
    #     graph.labelaxes(ax2, '$\kappa$ ($mm^{-1}$)', ax2_ylabel)
    #     graph.labelaxes(ax3, '$\kappa \eta $ ', ax3_ylabel)
    #
    #     # graph.setaxes(ax1, 10 ** -1.5, 10 ** 0.8, 10 ** 0.3, 10 ** 5.3)
    #     graph.setaxes(ax3, 10 ** -3.8, 2, 10 ** -3.5, 10 ** 6.5)
    #
    #     fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    #
    #     return fig1, (ax1, ax2, ax3)
    #
    #
    # def plot_mean_flow(udata, xx, yy, f_p=5., crop_edges=4, fps=1000., data_spacing=1, umin=-200, umax=200, tau0=0, tau1=10, yoffset_box=20.):
    #     """
    #     A method to quickly plot the mean flow
    #     ... dependencies: takumi.library.graph
    #     ... graph is Takumi's plotting module which mainly utilizes matplotlib.
    #     ... graph can be found under library/display her: https://github.com/tmatsuzawa/library
    #
    #
    #     Parameters
    #     ----------
    #     udata
    #     xx
    #     yy
    #     f_p
    #     crop_edges
    #     fps
    #     data_spacing
    #     umin
    #     umax
    #     tau0
    #     tau1
    #     yoffset_box
    #
    #     Returns
    #     -------
    #
    #     """
    #     __figsize__, __fontsize__ = (24, 20), 16
    #     params = {'figure.figsize': __figsize__,
    #               'font.size': __fontsize__,  # text
    #               'legend.fontsize': 16,  # legend
    #               'axes.labelsize': __fontsize__,  # axes
    #               'axes.titlesize': __fontsize__,
    #               'xtick.labelsize': __fontsize__,  # tick
    #               'ytick.labelsize': __fontsize__,
    #               'axes.edgecolor': 'black',
    #               'axes.linewidth': 0.8,
    #               'lines.linewidth': 5.}
    #     graph.update_figure_params(params)
    #     # Forcing Period
    #     tau_p = int(1. / f_p * fps / data_spacing)  # 1/f *(fps/data_spacing) in frames
    #     # no. of pixels to ignore at the edge
    #     n = crop_edges
    #
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
    enstrophy = get_enstrophy(udata[..., mask], dx=dx, dy=dy)

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
    x1
    y0
    y1
    t0
    t1

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

    Parameters
    ----------
    udata
    xx
    yy
    x0
    x1
    y0
    y1
    t0
    t1

    Returns
    -------

    """
    enstrophy = get_enstrophy(udata[:, y0:y1, x0:x1, t0:t1])
    en = np.nanmean(enstrophy, axis=-1)
    fig, ax, cc = graph.color_plot(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], en, label=label, vmin=vmin, vmax=vmax, **kwargs)
    graph.labelaxes(ax, xlabel, ylabel)

    return fig, ax, cc


def plot_spatial_avg_energy(udata, time, x0=0, x1=None, y0=0, y1=None, t0=0, t1=None,
                            ylabel='$\\frac{1}{2} \langle U_i U_i\\rangle~(mm^2/s^2)$',
                            xlabel='$t~(s)$', xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
    """

    Parameters
    ----------
    udata
    xx
    yy
    x0
    x1
    y0
    y1
    t0
    t1

    Returns
    -------

    """
    energy = get_energy(udata[:, y0:y1, x0:x1, t0:t1])
    e = np.nanmean(energy, axis=(0, 1))
    fig, ax = graph.plot(time, e)
    graph.labelaxes(ax, xlabel, ylabel)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    return fig, ax


# movie
def make_movie(imgname=None, imgdir=None, movname=None, indexsz='05', framerate=10, rm_images=False,
               save_into_subdir=False, start_number=0, framestep=1, ext='png', option='normal', overwrite=False,
               invert=False, add_commands=[], ffmpeg_path=moddirpath + '/ffmpeg'):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    ... ffmpeg is not smart enough to recognize a pattern like 0, 50, 100, 150... etc.
        It tries up to an interval of 4. So 0, 3, 6, 9 would work, but this hinders practicality.
        Use the glob feature in that case. i.e. option='glob'

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
        command = [ffmpeg_path,
                   '-framerate', str(int(framerate)),
                   '-start_number', str(start_number),
                   '-i', imgname + '%' + indexsz + 'd.' + ext,
                   '-pix_fmt', 'yuv420p',
                   '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100']
    else:
        # If images are not numbered or not labeled in a sequence, you can use the glob feature.
        # On command line,
        # ffmpeg -r 1
        # -pattern_type glob
        # -i '/Users/stephane/Documents/git/takumi/library/image_processing/images2/*.png'  ## It is CRITICAL to include '' on the command line!!!!!
        # -vcodec libx264 -crf 25  -pix_fmt yuv420p /Users/stephane/Documents/git/takumi/library/image_processing/images2/sample.mp4
        command = [ffmpeg_path,
                   '-pattern_type', 'glob',  # Use glob feature
                   '-framerate', str(int(framerate)),  # framerate
                   '-i', imgname + '/*.' + ext,  # images
                   '-vcodec', 'libx264',  # codec
                   '-crf', '12',  # quality
                   '-pix_fmt', 'yuv420p']
    if overwrite:
        command.append('-y')
    if invert:
        command.append('-vf')
        command.append('negate')
    # check if image has dimensions divisibly by 2 (if not ffmpeg raises an error... why ffmpeg...)
    # ffmpeg raises an error if image has dimension indivisible by 2. Always make sure that this is not the case.
    # image_paths = glob.glob(imgname + '/*.' + ext)
    # img = mpimg.imread(image_paths[0])
    # height, width = img.shape
    # if not (height % 2 == 0 and width % 2 == 0):
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


# def make_time_evo_movie_from_udata(qty, xx, yy, time, t=1, inc=100, label='$\\frac{1}{2} U_i U_i$ ($mm^2/s^2$)',
#                                    x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, vmin=0, vmax=None, option='scientific',
#                                    draw_box=True, xlabel='$x$ ($mm$)', ylabel='$y$ ($mm$)',
#                                    savedir='./', qtyname='qty', framerate=10,
#                                    ffmpeg_path='ffmpeg', overwrite=True, only_movie=False,
#                                    notebook=True, verbose=False):
#     """
#     Make a movie about the running average (number of frames to average is specified by "t"
#
#     Parameters
#     ----------
#     qty: 3D array (height, width, time)
#         ... quantity to show as a movie (energy, enstrophy, vorticity component, etc)
#     xx
#     yy
#     time
#     t
#     inc
#     label
#     x0
#     x1
#     y0
#     y1
#     z0
#     z1
#     vmin
#     vmax
#     draw_box
#     xlabel
#     ylabel
#     savedir
#     qtyname
#     framerate
#     ffmpeg_path
#     overwrite
#     only_movie
#     notebook
#
#     Returns
#     -------
#
#     """
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
#             fig1, ax1, cc1 = graph.color_plot(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], qty_ravg[y0:y1, x0:x1, t_ind*inc],
#                                               fignum=1, vmin=vmin, vmax=vmax, label=label, option=option)
#             if draw_box:
#                 graph.draw_box(ax1, xx, yy)
#             graph.labelaxes(ax1, xlabel, ylabel)
#             graph.title(ax1, '$t=%02.3f$ s' % t)
#             fig1.tight_layout()
#             graph.save(savedir + '/' +  qtyname + '/img%07d' % t_ind, ext='png', transparent=False, close=True,
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
#
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
def get_binned_stats(arg, var, n_bins=100, mode='linear', bin_center=True):
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

    Returns
    -------
    arg_bins: 1d array, bin centers
    var_mean: 1d array, mean values of data in each bin
    var_err: 1d array, std of data in each bin

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
        arg_means, arg_edges, binnumber = binned_statistic(arg[mask], arg[mask], statistic='mean', bins=bins)
    var_mean, bin_edges, binnumber = binned_statistic(arg[mask], var[mask], statistic='mean', bins=bins)
    var_err, _, _ = binned_statistic(arg[mask], var[mask], statistic='std', bins=bins)
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
        arg_bins, var_err = sort2arr(bin_centers, var_err)
    else:
        arg_bins, var_mean = sort2arr(arg_means, var_mean)
        arg_bins, var_err = sort2arr(arg_means, var_err)

    return arg_bins, var_mean, var_err / np.sqrt(counts)


def get_udata_from_path(udatapath, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                        t0=0, t1=None, inc=1, frame=None, return_xy=False, verbose=True,
                        slicez=None, crop=None, mode='r',
                        reverse_x=False, reverse_y=False, reverse_z=False):
    """
    Returns udata from a path to udata
    If return_xy is True, it returns udata, xx(2d grid), yy(2d grid)
    Parameters
    ----------
    udatapath
    x0: int
    x1: int
    y0: int
    y1: int
    t0: int
    t1: int
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
    udata, xx, yy

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
    with h5py.File(udatapath, mode) as f:
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
                    xx, yy, zz = f['x'][y0:y1, x0:x1, z0:z1], f['y'][y0:y1, x0:x1, z0:z1], f['z'][y0:y1, x0:x1, z0:z1]
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


# Spatial Pofile
def get_spatial_profile(xx, yy, qty, xc=None, yc=None, x0=0, x1=None, y0=0, y1=None, n=50,
                        return_center=False):
    """
    Returns a spatial profile (radial histogram) of 3D object with shape (height, width, duration)
    ... Computes a histogram of a given quantity as a function of distance from the center (xc, yc)
    ... If (xc, yc) are not given, it uses the center of the mass of the quantity at the first frame.

    Parameters
    ----------
    xx: 2d array
    yy: 2d array
    qty: 3D numpy array
    ... energy, enstrophy, etc. with shape (height, width, duration)
    xc: float
    ... x-coordinate of the origin of the polar coordinate
    yc: float
    ... y-coordinate of the origin of the polar coordinate
    x0: int, default: 0
    ... used to specify a portion of data for computing the histograms. xx[y0:y1, x0:x1]
    x1: int, default: None
    y0: int, default: 0
    y1: int, default: None
    n: int
    ... number of bins for the computed histograms

    Returns
    -------
    rs: 2d numpy array
    ... radial distance with shape (n, duration)
    qty_ms: 2d numpy array
    ... mean values of the quantity between r and r+dr with shape (n, duration)
    qty_errs: 2d numpy array
    ... std of the quantity between r and r+dr with shape (n, duration)
    """
    duration = qty.shape[-1]
    x, y = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
    qty_local = qty[y0:y1, x0:x1, ...]
    if xc is None or yc is None:
        # find a center of the mass from the initial image
        xc_i, yc_i = ndimage.measurements.center_of_mass(qty_local[y0:y1, x0:x1, 0])
        xc, yc = x[int(np.round(xc_i)), int(np.round(yc_i))], y[int(np.round(xc_i)), int(np.round(yc_i))]
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
                                                                  n_bins=n)
    if not return_center:
        return rs, qty_ms, qty_errs
    else:
        return rs, qty_ms, qty_errs, np.asarray([xc, yc])


########## Helpers ###########

def fix_udata_shape(udata):
    """
    It is better to always have udata with shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)
    This method fixes the shape of udata whose shape is (height, width, depth) or (height, width)

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
            # print ux.shape
            nrows, ncols, nstacks, duration = ux.shape
            return udata
        except:
            nrows, ncols, nstacks = ux.shape
            duration = 1
            ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
            uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
            uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
            return np.stack((ux, uy, uz))

def count_nans(arr):
    """Returns the number of nans in the given array"""
    nnans = np.count_nonzero(np.isnan(arr))
    return nnans

def get_equally_spaced_grid(udata, spacing=1):
    """
    Returns a equally spaced grid to plot udata

    Parameters
    ----------
    udata
    spacing: spacing of the grid

    Returns
    -------
    xx, yy, (zz): 2D or 3D numpy arrays
    """
    dim = len(udata)
    if dim == 2:
        height, width, duration = udata[0].shape
        x, y = list(range(width)), list(range(height))
        xx, yy = np.meshgrid(x, y)
        return xx * spacing, yy * spacing
    elif dim == 3:
        height, width, depth, duration = udata[0].shape
        x, y, z = list(range(width)), list(range(height)), list(range(depth))
        xx, yy, zz = np.meshgrid(y, x, z)
        return xx * spacing, yy * spacing, zz * spacing


def get_equally_spaced_kgrid(udata, dx=1):
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
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dx)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi  # Convert inverse length into wavenumber
        return kxx, kyy
    elif dim == 3:
        ncomp, height, width, depth, duration = udata.shape
        # k space grid
        kx = np.fft.fftfreq(width, d=dx)
        ky = np.fft.fftfreq(height, d=dx)
        kz = np.fft.fftfreq(depth, d=dx)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
        return kxx, kyy, kzz


def get_grid_spacing(xx, yy, zz=None):
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


def compute_kolmogorov_lengthscale_simple(epsilon, nu):
    """
    Return Kolmogorov length scale for a given set of dissipation rate and viscosity
    Parameters
    ----------
    epsilon: float, dissipation rate
    nu: float, viscosity

    Returns
    -------
    float, Kolmogorov length scale
    """
    return (nu ** 3 / epsilon) ** (0.25)


def get_characteristic_velocity(udata):
    """
    Return 1-component RMS velocity, u'
    energy = dim / 2 *  u'^2

    Parameters
    ----------
    udata

    Returns
    -------
    u_irms
    """
    dim = len(udata)
    u_irms = np.sqrt(2. / dim * get_spatial_avg_energy(udata)[0])
    return u_irms


def kronecker_delta_delta(i, j):
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
    x: numpy array
    y: numpy array

    Returns
    -------
    r: numpy array
    phi: numpy array
    """
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return r, phi


def cart2sph(x, y, z):
    """
    Transformation: cartesian to spherical
    z = r cos theta
    y = r sin theta sin phi
    x = r sin theta cos phi

    Parameters
    ----------
    x
    y
    z

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
    Returns velocity in the spherical basis when a velocity vector in cartesian coordinates is given

    z = r cos theta
    y = r sin theta sin phi
    x = r sin theta cos phi

    Spherical coorrdinates:
    (r, theta, phi) = (radial distance, polar angle, azimuthal angle)
    r: radius
    theta: polar angle [-pi/2, pi/2] (angle from the z-axis)
    phi: azimuthal angle [-pi, pi] (angle on the x-y plane)

    http://www.astrosurf.com/jephem/library/li110spherCart_en.html
    """
    xx_tmp, yy_tmp, zz_tmp = xx - xc, yy - yc, zz - zc
    R = np.sqrt(xx_tmp**2 + yy_tmp**2 + zz_tmp**2)
    Rxy = np.sqrt(xx_tmp**2 + yy_tmp**2)
    ur = (xx_tmp * ux + yy_tmp * uy + zz_tmp * uz ) / R
    utheta = - (zz_tmp * (xx_tmp * ux + yy_tmp * uy) - Rxy ** 2 * uz) / (R **2 * Rxy)
    uphi = (yy_tmp * ux - xx_tmp * uy) / (Rxy ** 2)
    return ur, utheta, uphi

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
    x, y, z


    """
    x = r * np.sin(theta) * np.cos(phi) + xc
    y = r * np.sin(theta) * np.sin(phi) + yc
    z = r * np.cos(theta) + zc
    return x, y, z

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


def find_nearest(array, value, option='normal'):
    """
    Find an element and its index closest to 'value' in 'array'
    Parameters
    ----------
    array: nd array
    value: float/int

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


def get_running_avg_1d(x, t, notebook=True):
    """
    Returns a running average of 1D array x. The number of elements to average over is specified by t.
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
    Returns a running average of nD array x. The number of elements to average over is specified by t.
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
                      axis=-1):
    """
    Returns phase average of a ND array (generalization of get_average_data_from_periodic_data)
    ... Assume x is a periodic data, and you are interested in averaging data by locking the phase.
       This function returns time (a cycle), phase-averaged data, std of the data
    ... Two methods to do this
        1. This is easy IF data is spaced evenly in time
        ... average [x[0], x[period], x[period*2], ...],
           then average [x[1], x[1+period], x[1+period*2], ...], ...
        2. Provide a time array as well as data.
            ... For example, one can give unevenly spaced data (ND array) in time
               one can take phase average by taking a histogram appropriately
    ... Arguments for each method:
        1. x, period_ind
        2. x, time, freq

    Parameters
    ----------
    x: ND array, data
        ... one of the array shape must must match len(time)
    period_ind: int
        ... period in index space
    time: 1d array, default: None
        ... time of data
    freq: float
        ... frequency of the periodicity of the data
    nbins: int
        ... number of points to probe data in the period
    axis: int, default:-1
        ... axis number to specify the temporal axis of the data

    Returns
    -------
    t_p: time (a single cycle)
        ... For the method 1, it returns np.arange(nbins)
    x_pavg: phase-averaged data (N-1)D array
    x_perr: std of the data by phase averaging (N-1)D array
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
            x_pavg[..., i] = np.nanmean(x.take(indices=range(0, x.shape[axis], period_ind), axis=axis))
            x_perr[..., i] = np.nanstd(x.take(indices=range(0, x.shape[axis], period_ind), axis=axis))
        x_pavg = np.swapaxes(x_pavg, axis, -1)
        x_perr = np.swapaxes(x_perr, axis, -1)

    if freq is not None and period_ind is None:
        time, x = np.asarray(time), np.asarray(x)
        time_mod = time % (1./freq)

        period = 1. / freq
        dt = period / nbins
        t_p = np.arange(nbins) * dt + dt / 2.

        shape_pavg = list(x.shape)
        del shape_pavg[axis]
        shape_pavg += [nbins]
        x_pavg = np.empty(shape_pavg)
        x_perr = np.empty(shape_pavg)

        for i in range(nbins):
            tmin, tmax = t_p[i] - dt / 2., t_p[i] + dt / 2,
            keep1, keep2 = time_mod >= tmin, time_mod < tmax
            keep = keep1 * keep2

            indices = np.arange(x.shape[axis])[keep]
            x_pavg[..., i] = np.nanmean(x.take(indices=indices, axis=axis), axis=0)
            x_perr[..., i] = np.nanstd(x.take(indices=indices, axis=axis), axis=0)

        x_pavg = np.swapaxes(x_pavg, axis, -1)
        x_perr = np.swapaxes(x_perr, axis, -1)
    return t_p, x_pavg, x_perr


def write_hdf5_dict(filepath, data_dict, overwrite=False):
    """
    Stores data_dict = {'varname0': var0, 'varname1': var1, ...} in hdf5
    - A quick function to store multiple data into a single hdf5 file

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
    hf = h5py.File(filename, 'a')
    for key in data_dict:
        if key in hf.keys() and overwrite:
            del hf[key]
        try:
            hf.create_dataset(key, data=data_dict[key])
        except RuntimeError:
            if overwrite:
                del hf[key]
                hf.create_dataset(key, data=data_dict[key])
            else:
                print(key, ' already exists in the h5 file! It will NOT be overwriting the existing data')

    hf.close()
    print('Data was successfully saved as ' + filename)


def convert_dat2h5files(dpath, savedir=None, verbose=False, overwrite=True, start=0):
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

def suggest_udata_dim2load(dpath, p=1., n=5, show=True, return_tuple=False):
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

    """
    height, width, depth, duration = get_udata_dim(dpath)
    inc = int(duration / n)
    # fractional number of nans
    nx = count_nans_along_axis(dpath, axis='x', inc=inc)
    ny = count_nans_along_axis(dpath, axis='y', inc=inc)
    nz = count_nans_along_axis(dpath, axis='z', inc=inc)

    lx, ly, lz = len(nx), len(ny), len(nz)

    x0, _ = find_nearest(nx[:int(lx/2)], (np.nanmin(nx) + np.nanmax(nx)) / 2. * p )
    x1, _ = find_nearest(nx[int(lx/2):], (np.nanmin(nx) + np.nanmax(nx)) / 2. * p )
    y0, _ = find_nearest(ny[:int(ly/2)], (np.nanmin(ny) + np.nanmax(ny)) / 2. * p )
    y1, _ = find_nearest(ny[int(ly/2):], (np.nanmin(ny) + np.nanmax(ny)) / 2. * p )
    z0, _ = find_nearest(nz[:int(lz/2)], (np.nanmin(nz) + np.nanmax(nz)) / 2. * p )
    z1, _ = find_nearest(nz[int(lz/2):], (np.nanmin(nz) + np.nanmax(nz)) / 2. * p )

    x1 += int(lx/2)
    y1 += int(ly/2)
    z1 += int(lz/2)

    if show:
        import tflow.graph as graph
        fig, ax = graph.plot(nx, label='x', subplot=121)
        fig, ax = graph.plot(ny, label='y', subplot=121)
        fig, ax = graph.plot(nz, label='z', subplot=121)

        graph.axvline(ax, x=x0, color='C0')
        graph.axvline(ax, x=x1, color='C0')

        graph.axvline(ax, x=y0, color='C1')
        graph.axvline(ax, x=y1, color='C1')

        graph.axvline(ax, x=z0, color='C2')
        graph.axvline(ax, x=z1, color='C2')
        ax.legend()

        nx_new = count_nans_along_axis(dpath, axis='x', inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        ny_new = count_nans_along_axis(dpath, axis='y', inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
        nz_new = count_nans_along_axis(dpath, axis='z', inc=inc, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)

        fig, ax2 = graph.plot(nx_new, label='x', subplot=122)
        fig, ax2 = graph.plot(ny_new, label='y', subplot=122)
        fig, ax2 = graph.plot(nz_new, label='z', subplot=122, figsize=(17, 8))
        ax.legend()
    print('... Suggested volume (x0, x1, y0, y1, z0, z1) = (%d, %d, %d, %d, %d, %d)' % (x0, x1, y0, y1, z0, z1))
    if not return_tuple:
        # Return x0,... in a dictionary- one can pass this to get_udata_from_path(..., **ind_dict)
        ind_dict = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}
        return ind_dict
    else:
        return x0, x1, y0, y1, z0, z1

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
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]

# functions to derive major quantities of turbulence from udata and save it into a hdf5 format
def derive_all(udata, dx, dy, savepath, udatapath='none', **kwargs):
    """
    A shortcut function to derive major quantities of turbulence

    Parameters
    ----------
    udata
    dx
    dy
    savepath
    kwargs

    Returns
    -------

    """
    print('... Derive quantities which require small computational power')
    derive_easy(udata, dx, dy, savepath, udatapath=udatapath, **kwargs)
    print('... Derive quantities which require immense computational power')
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
    udata
    dx
    dy
    savepath
    time
    inc
    dz
    nu
    x0
    x1
    y0
    y1
    z0
    z1
    t0
    t1
    reynolds_decomp
    notebook
    overwrite

    Returns
    -------

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
     taylor microscale as a curvature of the long. vel. autocorr. func. etc.

    Parameters
    ----------
    udata
    dx
    dy
    savepath
    time
    inc
    dz
    nu
    x0
    x1
    y0
    y1
    z0
    z1
    t0
    t1
    reynolds_decomp
    coarse
    coarse2
    notebook
    overwrite

    Returns
    -------

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

        # dissipation rate using the isotropic formula and Taylor microscale from the vel. autocorrelation function
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
    savepath
    datanames
    verbose
    mode

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
