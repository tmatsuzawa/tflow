from scipy.stats import binned_statistic
from tqdm import tqdm
import numpy as np
import sys
import re
import h5py
import os
import numpy.ma as ma
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy.ma as ma
from scipy import signal
import os
import copy

"""
Philosophy:
udata = (ux, uy, uz) or (ux, uy)
each ui has a shape (height, width, (depth), duration)

If ui's are individually given, make udata like 
udata = np.stack((ux, uy))
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
    shape = udata.shape #shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
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
        print 'Not implemented yet.'
        return None
    return sij

def decompose_duidxj(sij):
    """
    Decompose a duidxj tensor into a symmetric and an antisymmetric parts
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
    dim = len(sij.shape) - 3 # spatial dim
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
                    eij[..., t, i, j] = 1./2. * (sij[..., t, i, j] + sij[..., t, j, i])
                    # gij[..., i, j] += 1./2. * (sij[..., i, j] - sij[..., j, i]) #anti-symmetric part
                else:
                    eij[..., t, i, j] = eij[..., t, j, i]
                    # gij[..., i, j] = -gij[..., j, i] #anti-symmetric part

    gij = sij - eij
    return eij, gij

def reynolds_decomposition(udata):
    """
    Apply the Reynolds decomposition to a velocity field

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
        u_mean[i] = np.nanmean(udata[i], axis=dim) # axis=dim is always the time axis in this convention
        for t in range(udata.shape[-1]):
            u_turb[i, ..., t] = udata[i,...,t] - u_mean[i]
    return u_mean, u_turb

def fft_velocity(udata, x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, dx=None, dy=None, dz=None):
    """
    Return Fourier transform of velocity
    Parameters
    ----------
    udata

    Returns
    -------
    ukdata
    """
    if dx is None or dy is None:
        print 'ERROR: dx or dy is not provided! dx is grid spacing in real space.'
        print '... k grid will be computed based on this spacing! Please provide.'
        raise ValueError
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = udata[:, y0:y1, x0:x1, :]
    dim = udata.shape[0]
    if dim == 2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        if dz is None:
            print 'ERROR: dz is not provided! dx is grid spacing in real space.'
            print '... k grid will be computed based on this spacing! Please provide.'
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    udata = fix_udata_shape(udata)
    ukdata = np.zeros_like(udata)

    for i in range(dim):
        ukdata[i, ...] = np.fft.fftn(udata[i, ...], axes=range(dim))
        ukdata[i, ...] = np.fft.fftshift(ukdata[i, ...], axes=range(dim))

    if dim == 2:
        ncomp, height, width, duration = ukdata.shape
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dy)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber

        return ukdata, np.asarray([kxx, kyy])

    elif dim == 3:
        ncomp, height, width, depth, duration = ukdata.shape
        kx = np.fft.fftfreq(width, d=dx)
        ky = np.fft.fftfreq(height, d=dy)
        kz = np.fft.fftfreq(depth, d=dz)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi
        return ukdata, np.asarray([kxx, kyy, kzz])


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
    sij = get_duidxj_tensor(udata) #shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    dim = len(sij.shape) - 3  # spatial dim
    div_u = np.zeros(sij.shape[:-2])
    for d in range(dim):
        div_u += sij[..., d, d]
    return div_u

def curl(udata, dx=1., dy=1., dz=1.):
    """
    Computes curl of a velocity field using a rate of strain tensor
    ... For dim=3, the sign might need to be flipped... not tested
    ... if you already have velocity data as ux = array with shape (m, n) and uy = array with shape (m, n),
        udata = np.stack((ugrid1, vgrid1))
        omega = vec.curl(udata)
    Parameters
    ----------
    udata: (ux, uy, uz) or (ux, uy)

    Returns
    -------
    omega: numpy array with shape (height, width, duration) (2D) or (height, width, duration) (2D)

    """
    sij = get_duidxj_tensor(udata, dx=dx, dy=dy, dz=dz)
    dim = len(sij.shape) - 3  # spatial dim
    eij, gij = decompose_duidxj(sij)
    if dim == 2:
        omega = 2 * gij[..., 1, 0]
    elif dim == 3:
        # sign might be wrong
        omega1, omega2, omega3 = 2.* gij[..., 2, 1], 2.* gij[..., 0, 2], 2.* gij[..., 1, 0]
        omega = np.stack((omega1, omega2, omega3))
    else:
        print 'Not implemented yet!'
        return None
    return omega

def curl_2d(ux, uy):
    """
    Calculate curl of 2D field
    Parameters
    ----------
    var: 2d array
        element of var must be 2d array

    Returns
    -------

    """

    #ux, uy = var[0], var[1]
    xx, yy = ux.shape[0], uy.shape[1]

    omega = np.zeros((xx, yy))
    # duxdx = np.gradient(ux, axis=1)
    duxdy = np.gradient(ux, axis=0)
    duydx = np.gradient(uy, axis=1)
    # duydy = np.gradient(uy, axis=0)

    omega = duydx - duxdy

    return omega


########## Elementary analysis ##########
def get_energy(udata):
    shape = udata.shape  # shape=(dim, nrows, ncols, nstacks) if nstacks=0, shape=(dim, nrows, ncols)
    dim = udata.shape[0]
    energy = np.zeros(shape[1:])
    for d in range(dim):
        energy += udata[d, ...] ** 2
    energy /= 2.
    return energy

def get_enstrophy(udata, dx=1., dy=1., dz=1.):
    dim = udata.shape[0]
    omega = curl(udata, dx=dx, dy=dy, dz=dz)
    shape = omega.shape # shape=(dim, nrows, ncols, nstacks, duration) if nstacks=0, shape=(dim, nrows, ncols, duration)
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
    Returns time-averaged-energy
    Parameters
    ----------
    udata

    Returns
    -------

    """
    dim = udata.shape[0]
    energy = get_energy(udata)
    energy_avg = np.nanmean(energy, axis=dim)
    return energy_avg

def get_time_avg_enstrophy(udata, dx=1., dy=1., dz=1.):
    """
    Returns time-averaged-enstrophy
    Parameters
    ----------
    udata

    Returns
    -------

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
    udata

    Returns
    -------
    energy_vs_t

    """
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim==2:
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
    udata

    Returns
    -------
    enstrophy_vs_t

    """
    if x1 is None:
        x1 = udata[0].shape[1]
    if y1 is None:
        y1 = udata[0].shape[0]

    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim==2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    enstrophy = get_enstrophy(udata, dx=dx, dy=dy, dz=dz)
    enstrophy_vs_t = np.nanmean(enstrophy, axis=tuple(range(dim)))
    enstrophy_vs_t_err = np.nanstd(enstrophy, axis=tuple(range(dim)))
    return enstrophy_vs_t, enstrophy_vs_t_err


########## Energy spectrum, Dissipation spectrum ##########

def fft_nd(field, dx=1, dy=1, dz=1):
    """
    Parameters
    ----------
    field: np array, (height, width, depth, duration) or (height, width, duration)
    dx: spacing along x-axis
    dy: spacing along x-axis
    dz: spacing along x-axis

    Returns
    -------
    field_fft
    np.asarray([kx, ky, kz])

    """
    dim = len(field.shape) - 1
    n_samples = 1
    for d in range(dim):
        n_samples *= field.shape[d]

    field_fft = np.fft.fftn(field, axes=range(dim))
    field_fft = np.fft.fftshift(field_fft, axes=range(dim))
    field_fft /= n_samples# Divide the result by the number of samples (this is because of discreteness of FFT)

    if dim == 2:
        height, width, duration = field.shape
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dy)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber

        return ek, np.asarray([kxx, kyy])

    elif dim == 3:
        height, width, depth, duration = field.shape
        kx = np.fft.fftfreq(width, d=dx)
        ky = np.fft.fftfreq(height, d=dy)
        kz = np.fft.fftfreq(depth, d=dz)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kz = np.fft.fftshift(kz)
        kxx, kyy, kzz = np.meshgrid(ky, kx, kz)
        kxx, kyy, kzz = kxx * 2 * np.pi, kyy * 2 * np.pi, kzz * 2 * np.pi


def get_energy_spectrum_nd(udata, x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None,
                           window=None, correct_signal_loss=True):

    """
    Returns nd energy spectrum from velocity data
    Parameters
    ----------
    udata
    dx: data spacing in x (units: mm/px)
    dy: data spacing in y (units: mm/px)
    dz: data spacing in z (units: mm/px)

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
    ek, ks = vel.get_energy_spectrum_nd(udata_test, dx=dx, dy=dy)
    graph.color_plot(xx, yy, (ux**2 + uy**2 ) /2., fignum=2, subplot=121)
    fig22, ax22, cc22 = graph.color_plot(ks[0], ks[1], ek.real[..., 0], fignum=2, subplot=122, figsize=(16, 8))
    graph.setaxes(ax22, -10, 10, -10, 10)

    """
    if dx is None or dy is None:
        print 'ERROR: dx or dy is not provided! dx is grid spacing in real space.'
        print '... k grid will be computed based on this spacing! Please provide.'
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
            print 'ERROR: dz is not provided! dx is grid spacing in real space.'
            print '... k grid will be computed based on this spacing! Please provide.'
            raise ValueError
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

    n_samples = 1
    for d in range(dim):
        n_samples *= udata.shape[d+1]

    # Apply a window to get lean FFT spectrum for aperiodic signals
    duration = udata.shape[-1]
    if window is not None:
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows =get_window_radial(xx, yy, wtype=window, duration=duration)
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
        ukdata = np.fft.fftn(udata_tapered, axes=range(1, dim + 1))
        ukdata = np.fft.fftshift(ukdata, axes=range(1, dim + 1))

        energy, energy_tapered = get_energy(udata), get_energy(udata_tapered)
        signal_intensity_losses = np.nanmean(energy_tapered, axis=tuple(range(dim))) / np.nanmean(energy, axis=tuple(range(dim)))

    else:
        ukdata = np.fft.fftn(udata, axes=range(1, dim+1))
        ukdata = np.fft.fftshift(ukdata, axes=range(1, dim+1))
        signal_intensity_losses = np.ones(duration)

    # compute E(k)
    ek = np.zeros(ukdata[0].shape)

    for i in range(dim):
        ek[...] += np.abs(ukdata[i, ...]) ** 2 / n_samples
    ek /= 2.



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
                        window=None, remove_undersampled_region=True, cc=1.75, notebook=True):
    """
    Returns 1D energy spectrum from velocity field data

    Parameters
    ----------
    udata
    x0
    x1
    y0
    y1
    z0
    z1
    z
    dx
    dy
    dz
    nkout: int, number of bins to take histograms from nd fft output
    notebook: bool, if True, use tqdm_notebook instead of tqdm.
        Set it False if you are NOT calling this function from notebook.

    Returns
    -------

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print 'Using tqdm_notebook. If this is a mistake, set notebook=False'
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
        Importantly, this will output the SPECTRAL DENSITY,
        not power which is integrated spectrated density in a pixel/voxel (delta_kx, delta_ky.)
        ... Ask Takumi for derivation. The derivation goes like this.
        ...... 1. Start with the Parseval's theorem.
        ...... 2. Write the discretized equation about the TKE: Average TKE = sum deltak * E(k)
        ...... 3. Using 1, write down the avg TKE
        ...... 4. Equate 2 and 3. You get e_k1d * jacobian / (n_samples * deltak)
        ......   IF deltak = deltakr where deltakr = np.sqrt(deltakx**2 + deltaky**2) for 2D
        ......   where e_k1d is just a histogram value obtained from the DFT result (i.e. POWER- spectral density integrated over a px)
        ...... 5. Finally, convert this into the SPECTRAL DENSITY. This is two-fold.
        ...... 5.1.
        ......   e_k1d * jacobian / (n_samples * deltak) is not necessarily the correct density if deltak is not equal to deltakr.
        ......   This is because e_k1d comes from the histogram of the input velocity field.
        ......   One can show that the correction is just (deltak / deltakr) ** dim
        ...... 5.2
        ......   After 5.1, this finally the integrated power between k and k + deltak
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
            e_ks *= deltakx * deltaky # use the raw DFT outputs (power=integrated density over a px)
            deltakr = np.sqrt(deltakx**2 + deltaky**2) # radial k spacing of the velocity field
        if dim == 3:
            deltakx, deltaky, deltakz = ks[0, 0, 1, 0] - ks[0, 0, 0, 0], ks[1, 1, 0, 0] - ks[1, 0, 0, 0], ks[2, 0, 0, 1] - ks[2, 0, 0, 0]
            e_ks *= deltakx * deltaky * deltakz # use the raw DFT outputs (power=integrated density over a px)
            deltakr = np.sqrt(deltakx ** 2 + deltaky ** 2 + deltakz ** 2)  # radial k spacing of the velocity field
        kk = np.zeros((ks.shape[1:]))
        for i in range(dim):
            kk += ks[i, ...] ** 2
        kk = np.sqrt(kk) # radial k

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
            e_k1ds[..., t] = e_k1d * jacobian / (n_samples * deltak) * (deltak / deltakr) ** 2 / deltak * cc
            e_k1d_errs[..., t] = e_k1d_err * jacobian / (n_samples * deltak) * (deltak / deltakr) ** 2 / deltak * cc

            # print deltak / deltakr

        return e_k1ds, e_k1d_errs, k1ds

    dim, duration = len(udata), udata.shape[-1]

    e_ks, ks = get_energy_spectrum_nd(udata, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, dx=dx, dy=dy, dz=dz, window=window)
    # OLD: only capable to convert 2d spectra to one
    # if dim == 3:
    #     kx, ky, kz = ks[0], ks[1], ks[2]
    #     ######################################################################
    #     # Currently, I haven't cleaned up a code to 3d power spectra into 1d.
    #     # The currently implemented solution is just use a 2D slice of the 3D data. <- boo
    #     ######################################################################
    #     kx, ky = kx[..., 0], ky[..., 0]
    # elif dim == 2:
    #     kx, ky = ks[0], ks[1]
    #
    # if dim == 3:
    #     e_ks = e_ks[:, :, z, :]
    # e_k, kk = convert_2d_spec_to_1d(e_ks, kx, ky, nkout=nkout)
    e_k, e_k_err, kk = convert_nd_spec_to_1d(e_ks, ks, nkout=nkout, cc=cc)

    # #### NORMALIZATION IS NO LONGER NEEDED ####
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

    return e_k[0:], e_k_err[0:], kk[0:]

def get_1d_energy_spectrum(udata, k='kx', x0=0, x1=None, y0=0, y1=None,
                           z0=0, z1=None, dx=None, dy=None, dz=None,
                           window=None, correct_signal_loss=True, notebook=True):
    """
    Returns 1D energy spectrum from velocity field data

    Parameters
    ----------
    udata
    x0
    x1
    y0
    y1
    z0
    z1
    dx
    dy
    dz
    nkout: int, number of bins to take histograms from nd fft output
    notebook: bool, if True, use tqdm_notebook instead of tqdm.
        Set it False if you are NOT calling this function from notebook.

    Returns
    -------

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
        ux, uy = udata[0, y0:y1, x0:x1, :],  udata[1, y0:y1, x0:x1, :]
        height, width, duration = ux.shape
        udata_tmp = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        ux, uy, uz = udata[0, y0:y1, x0:x1, z0:z1, :], udata[1, y0:y1, x0:x1, z0:z1, :], udata[2, y0:y1, x0:x1, z0:z1, :]
        height, width, depth, duration = ux.shape
        udata_tmp = udata[:, y0:y1, x0:x1, z0:z1, :]
    else:
        raise ValueError('... Error: Invalid dimension is given. Use 2 or 3 for the number of spatial dimensions. ')

    n_samples = 1
    for i in range(dim):
        n_samples *= ux.shape[i]
    if k == 'kx':
        ax_ind = 1 # axis number to take 1D DFT
        n = width
        d = dx
        if dim == 2:
            ax_ind_for_avg = 0  # axis number(s) to take statistics (along y)
        elif dim == 3:
            ax_ind_for_avg = (0, 2)  # axis number(s) to take statistics  (along y and z)
    elif k == 'ky':
        ax_ind = 0 # axis number to take 1D DFT
        n = height
        d = dy
        if dim == 2:
            ax_ind_for_avg = 1 # axis number(s) to take statistics  (along x)
        elif dim == 3:
            ax_ind_for_avg = (1, 2) # axis number(s) to take statistics  (along x and z)
    elif k == 'kz':
        ax_ind = 2 # axis number to take 1D DFT
        n = depth
        d = dz
        ax_ind_for_avg = (0, 1)  # axis number(s) to take statistics  (along x and y)

    # Apply hamming window to get lean FFT spectrum for aperiodic signals
    if window is not None:
        duration = udata.shape[-1]
        if dim == 2:
            xx, yy = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
            udata_tapered = np.empty_like(udata_tmp)
            for i in range(dim):
                udata_tapered[i, ...] = udata_tmp[i, ...] * windows
            ux, uy = udata_tapered
        elif dim == 3:
            xx, yy, zz = get_equally_spaced_grid(udata, spacing=dx)
            windows = get_window_radial(xx, yy, zz, wtype=window, duration=duration, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1)
            # windows = np.repeat(window[np.newaxis, ...], dim, axis=0)
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
    e11 = np.nanmean(e11_nd, axis=ax_ind_for_avg)[:n/2, :]
    e11_err = np.nanstd(e11_nd, axis=ax_ind_for_avg)[:n/2, :]

    # E22
    uy_k = np.fft.fft(uy, axis=ax_ind) / np.sqrt(n_samples)
    e22_nd = np.abs(uy_k * np.conj(uy_k))
    e22 = np.nanmean(e22_nd, axis=ax_ind_for_avg)[:n/2, :]
    e22_err = np.nanstd(e22_nd, axis=ax_ind_for_avg)[:n/2, :]

    # Get an array for wavenumber
    k = np.fft.fftfreq(n, d=d)[:n/2] * 2 * np.pi # shape=(n, duration)
    deltak = k[1] - k[0]
    if dim == 3:
        # E33
        uz_k = np.fft.fft(uz, axis=ax_ind) / np.sqrt(n_samples)
        e33_nd = np.abs(uz_k * np.conj(uz_k))
        e33 = np.nanmean(e33_nd, axis=ax_ind_for_avg)[:n/2, :]
        e33_err = np.nanstd(e33_nd, axis=ax_ind_for_avg)[:n/2, :]

        eiis, eii_errs = np.array([e11, e22, e33]), np.array([e11_err, e22_err, e33_err])
    elif dim == 2:
        eiis, eii_errs = np.array([e11, e22]), np.array([e11_err, e22_err])
    else:
        raise ValueError('... 1d spectrum: Check the dimension of udata! It must be 2 or 3!')

    # Get the correct power
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
                        notebook=True):
    """
    Returns dissipation spectrum D(k) = 2 nu k^2 E(k) where E(k) is the 1d energy spectrum
    Parameters
    ----------
    udata
    nu
    x0
    x1
    y0
    y1
    z0
    z1
    dx
    dy
    dz
    nkout
    notebook

    Returns
    -------
    D_k: nd arraay
    D_k_err: nd arraay
    k1d: nd arraay
    """
    e_k, e_k_err, k1d = get_energy_spectrum(udata, dx=dx, dy=dy, dz=dz, nkout=nkout, x0=x0, x1=x1, y0=y0, y1=y1,
                                                z0=z0, z1=z1, notebook=notebook)
    # Plot dissipation spectrum
    D_k, D_k_err = 2 * nu * e_k * (k1d ** 2), 2 * nu * e_k_err * (k1d ** 2)
    return D_k, D_k_err, k1d

def get_rescaled_energy_spectrum(udata, epsilon=10**5, nu=1.0034,x0=0, x1=None,
                                 y0=0, y1=None, z0=0, z1=None,
                                 dx=1, dy=1, dz=1, nkout=None,
                                 window=None, notebook=True):
    # get energy spectrum
    e_k, e_k_err, kk = get_energy_spectrum(udata, x0=x0, x1=x1,
                                 y0=y0, y1=y1, z0=z0, z1=z1,
                                 dx=dx, dy=dy, dz=dz, nkout=nkout, window=window,
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
                                 dx=1, dy=1, dz=1, notebook=True, window=None, correct_signal_loss=True):
    dim = len(udata)
    duration = udata.shape[-1]
    # get energy spectrum
    eii_arr, eii_err_arr, k1d = get_1d_energy_spectrum(udata, x0=x0, x1=x1,
                                 y0=y0, y1=y1, z0=z0, z1=z1,
                                 dx=dx, dy=dy, dz=dz,
                                 window=window, correct_signal_loss=correct_signal_loss, notebook=notebook)
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


def get_rescaled_dissipation_spectrum(udata, epsilon=10**5, nu=1.0034,x0=0, x1=None,
                                 y0=0, y1=None, z0=0, z1=None, z=0,
                                 dx=1, dy=1, dz=1, nkout=None, notebook=True):
    """
    Return rescaled dissipation spectra
    D(k)/(u_eta^3) vs k * eta
    ... convention: k =  2pi/ L

    Parameters
    ----------
    udata
    epsilon
    nu
    x0
    x1
    y0
    y1
    z0
    z1
    z
    dx
    dy
    dz
    nkout
    notebook

    Returns
    -------
    D_k_norm:
    D_k_err_norm:
    k_norm:
    """
    # get dissipation spectrum
    D_k, D_k_err, k1d = get_dissipation_spectrum(udata, nu=nu, x0=x0, x1=x1,
                                 y0=y0, y1=y1, z0=z0, z1=z1,
                                 dx=dx, dy=dy, dz=dz, nkout=nkout, notebook=notebook)

    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** 0.25  # mm
    u_eta = (nu * epsilon) ** 0.25
    print 'dissipation rate, Kolmogorov scale: ', epsilon, eta

    k_norm = k1d * eta
    D_k_norm = D_k[...] / (u_eta ** 3)
    D_k_err_norm = D_k_err[...] / (u_eta ** 3)
    return D_k_norm, D_k_err_norm, k_norm

def scale_energy_spectrum(e_k, kk, epsilon=10**5, nu=1.0034, e_k_err=None):
    # Kolmogorov length scale
    eta = (nu ** 3 / epsilon) ** (0.25)  # mm
    # print 'dissipation rate, Kolmogorov scale: ', epsilon, eta

    k_norm = kk * eta
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
        print 'ERROR: dx or dy is not provided! dx is grid spacing in real space.'
        print '... k grid will be computed based on this spacing! Please provide.'
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
            print 'ERROR: dz is not provided! dx is grid spacing in real space.'
            print '... k grid will be computed based on this spacing! Please provide.'
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
        ukdata = np.fft.fftn(udata_tapered, axes=range(1, dim + 1))
        ukdata = np.fft.fftshift(ukdata, axes=range(1, dim + 1))
    else:
        ukdata = np.fft.fftn(udata, axes=range(1, dim + 1))
        ukdata = np.fft.fftshift(ukdata, axes=range(1, dim + 1))

    if dim == 2:
        ncomp, height, width, duration = udata.shape
        # k space grid
        kx = np.fft.fftfreq(width, d=dx)  # this returns FREQUENCY (JUST INVERSE LENGTH) not ANGULAR FREQUENCY
        ky = np.fft.fftfreq(height, d=dy)
        kx = np.fft.fftshift(kx)
        ky = np.fft.fftshift(ky)
        kxx, kyy = np.meshgrid(kx, ky)
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber
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
    print '# of Masked Elements / total: %d / %d = %.4f' % (np.sum(mask), n_samples, np.sum(mask) / float(n_samples))
    # Let uk array into a masked array
    ukdata = ma.asarray(ukdata)

    mask = np.repeat(mask[..., np.newaxis], duration, axis=dim)
    mask = np.repeat(mask[np.newaxis, ...], dim, axis=0)
    ukdata.mask = mask
    # Make all elements where kr > kmax = 0 (Low pass filter)
    ukdata = ukdata.filled(fill_value=0)

    # Inverse FT
    ukdata = np.fft.ifftshift(ukdata, axes=range(1, dim + 1))
    udata_ifft = np.fft.ifftn(ukdata, axes=range(1, dim + 1)).real# Use only a real part

    return udata_ifft, coords




########## DISSIPATION RATE ##########

def get_epsilon_using_sij(udata, dx=None, dy=None, dz=None, nu=1.004,
                          x0=0, x1=None, y0=0, y1=None, z0=0, z1=None):
    """
    sij: numpy array with shape (nrows, ncols, duration, 2, 2) (dim=2) or (nrows, ncols, nstacks, duration, 3, 3) (dim=3)
    ... idea is... sij[spacial coordinates, time, tensor indices]
        e.g.-  sij(x, y, t) can be accessed by sij[y, x, t, i, j]
    ... sij = d ui / dxj
    Parameters
    ----------
    udata

    Returns
    -------

    """
    udata = fix_udata_shape(udata)
    dim = len(udata)

    if dim==2:
        udata = udata[:, y0:y1, x0:x1, :]
    elif dim == 3:
        if z1 is None:
            z1 = udata[0].shape[2]
        udata = udata[:, y0:y1, x0:x1, z0:z1, :]

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
        # Estimate epsilon from 2D data
        epsilon_0 = np.nanmean(duidxj[..., 0, 0] ** 2, axis=tuple(range(dim)))
        epsilon_1 = np.nanmean(duidxj[..., 1, 1] ** 2, axis=tuple(range(dim)))
        epsilon_2 = np.nanmean(duidxj[..., 0, 1] * duidxj[..., 0, 1], axis=tuple(range(dim)))
        epsilon = 6. * nu * (epsilon_0 + epsilon_1 + epsilon_2)
    return epsilon

def get_epsilon_iso(udata, lambda_f=None, lambda_g=None, nu=1.004, x=None, y=None, **kwargs):
    """
    Return epsilon computed by isotropic formula involving Taylor microscale
    Parameters
    ----------
    udata
    lambda_f
    lambda_g
    nu

    Returns
    -------

    """
    dim = len(udata)
    u2_irms = 2. / dim * get_spatial_avg_energy(udata)[0]

    # if both of lambda_g and lambda_f are provided, use lambdaf over lambdag
    if lambda_f is None and lambda_g is None:
        print '... Both of Taylor microscales, lambda_f, lambda_g, were not provided!'
        print '... Compute lambdas from scratch. One must provide x and y.'
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

def get_epsilon_using_diss_spectrum(udata, nu=1.0034,x0=0, x1=None,
                                 y0=0, y1=None, z0=0, z1=None,
                                 dx=1, dy=1, dz=1, nkout=None, notebook=True):
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
        print 'Using tqdm_notebook. If this is a mistake, set notebook=False'
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
        arr1, arr2 = zip(*sorted(zip(arr1, arr2)))
        return arr1, arr2


    if x0 is None:  # if None, use the whole space
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:  # if None, use the whole space
        x1 = ui.shape[1]
    if y1 is None:
        y1 = ui.shape[0]
    if t0 is None:
        t0 = 0
    if t1 is None:
        t1 = ui.shape[2]

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

    for t in tqdm(range(t0, t1), desc='autocorr. time'):
        # Call velocity field at time t as uu
        uu = ui[y0:y1, x0:x1, t]

        roll_indices = range(0, limits[roll_axis], int(1. / coarse))
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        uu2_norm = np.nanmean(ui[y0:y1, x0:x1, ...] ** 2, axis=(0, 1))  # mean square velocity (avg over space)

        rr = np.empty((n, m))
        corr = np.empty((n, m))

        # for i in tqdm(range(int(coarse * limits[roll_axis])), desc='computing correlation'):
        for i in range(int(coarse * limits[roll_axis])):
            uu_rolled = np.roll(uu, i, axis=roll_axis)
            x_grid_rolled, y_grid_rolled = np.roll(x_grid, i, axis=roll_axis), np.roll(y_grid, i, axis=roll_axis)
            r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)
            corr_uu = uu * uu_rolled / uu2_norm[t]  # correlation values
            rr[:, i] = r_grid.flatten()[:n]
            corr[:, i] = corr_uu.flatten()[:n]

        # flatten arrays to feed to binned_statistic
        rr, corr = rr.flatten(), corr.flatten()

        # get a histogram
        rr_means, rr_edges, binnumber = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
        corr_, _, _ = binned_statistic(rr, corr, statistic='mean', bins=n_bins)
        corr_err, _, _ = binned_statistic(rr, corr, statistic='std', bins=n_bins)

        # One may use rr_means or the middle point of each bin for plotting
        # Default is the middle point
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
        rrs[:, t] = rr
        corrs[:, t] = corr
        corr_errs[:, t] = corr_err

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, corrs, corr_errs

def compute_spatial_autocorr3d(ui, x, y, z, roll_axis=1, n_bins=None, x0=None, x1=None, y0=None, y1=None,
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
        print 'Using tqdm_notebook. If this is a mistake, set notebook=False'
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
        arr1, arr2 = zip(*sorted(zip(arr1, arr2)))
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

    for t in tqdm(range(t0, t1), desc='autocorr. 3d time'):
        # Call velocity field at time t as uu
        uu = ui[y0:y1, x0:x1, z0:z1, t]

        uu2_norm = np.nanmean(ui[y0:y1, x0:x1, z0:z1, ...] ** 2, axis=(0, 1, 2))  # mean square velocity

        # Initialization
        # rr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))
        # corr = np.empty((x_grid.size, int(coarse * limits[roll_axis]) * 2 - 1))

        roll_indices = range(0, limits[roll_axis], int(1./coarse))
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        rr = np.empty((n, m))
        corr = np.empty((n, m))

        for j, i in enumerate(tqdm(roll_indices, desc='computing correlation')):
            uu_rolled = np.roll(uu, i, axis=roll_axis)
            x_grid_rolled, y_grid_rolled, z_grid_rolled = np.roll(x_grid, i, axis=roll_axis), \
                                                          np.roll(y_grid, i, axis=roll_axis), \
                                                          np.roll(z_grid, i, axis=roll_axis)
            r_grid = np.sqrt((x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2 + (z_grid_rolled - z_grid) ** 2)
            corr_uu = uu * uu_rolled / uu2_norm[t]  # correlation values
            rr[:, j] = r_grid.flatten()[:n]
            corr[:, j] = corr_uu.flatten()[:n]

        # flatten arrays to feed to binned_statistic
        rr, corr = rr.flatten(), corr.flatten()

        # get a histogram
        # rr_, _, _ = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
        rr_means, rr_edges, binnumber = binned_statistic(rr, rr, statistic='mean', bins=n_bins-1)
        rr_binwidth = (rr_edges[1] - rr_edges[0])
        rr_ = rr_edges[1:] - rr_binwidth / 2
        corr_, _, _ = binned_statistic(rr, corr, statistic='mean', bins=n_bins-1)
        corr_err, _, _ = binned_statistic(rr, corr, statistic='std', bins=n_bins-1)

        # # Sort arrays
        # rr, corr = sort2arr(rr_, corr_)
        # rr, corr_err = sort2arr(rr_, corr_err)
        #
        # # Insert to a big array
        # rrs[:, t] = rr
        # corrs[:, t] = corr
        # corr_errs[:, t] = corr_err

        # This is faster?
        rrs[0:, t], corr_errs[0:, t] = 0, 1.
        corrs[0, t] = 1.
        _, corrs[1:, t] = sort2arr(rr_, corr_)
        rrs[1:, t], corr_errs[1:, t] = sort2arr(rr_, corr_err)

    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, corrs, corr_errs


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

    udata = fix_udata_shape(udata)
    dim = len(udata)
    if dim == 2:
        height, width, duration = udata[0].shape
        ux, uy = udata[0], udata[1]
    elif dim == 3:
        height, width, depth, duration = udata[0].shape
        ux, uy, uz = udata[0], udata[1], udata[2]

    print 'Compute two-point velocity autocorrelation'
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

    if return_rij:
        print 'Compute two-point velocity autocorrelation tensor Rij.'
        # Make long./trans. autocorrelation functions
        ## make an additional list to use 2d interpolation: i.e. supply autocorrelation values as a function of (r, t)
        time_list = []
        for t in time:
            time_list += [t] * len(r_long[:, 0])
        # 2d interpolation of long./trans. autocorrelation functions
        print '... 2D interpolation to define long./trans. autocorrelation function (this may take a while)'
        f = interp2d(r_long.flatten(), time_list, f_long.flatten())
        g = interp2d(r_tran.flatten(), time_list, g_tran.flatten())

        # Define Rij(r, t) as a function.
        def two_pt_velocity_autocorrelation_tensor(i, j, r, t, udata):
            dim = len(r)
            u2_avg = np.nanmean(udata ** 2, axis=tuple(range(dim + 1))) # spatial average
            if dim == 2:
                x, y = r[0], r[1]
            elif dim == 3:
                x, y, z = r[0], r[1], r[2]
            r2_norm = np.zeros_like(x)
            for k in range(dim):
                r2_norm += r[k] ** 2
            r_norm = np.sqrt(r2_norm)
            Rij_value = u2_avg[t] * (g(r_norm, t) * klonecker_delta(i, j) + (f(r_norm , t)-g(r_norm , t)) * r[i] * r[j] / (r_norm ** 2))
            return Rij_value
        print '... Returning two-point velocity autocorrelation tensor Rij(r, t). Arguments: i, j, r, t. Pope Eq. 6.44.'
        return two_pt_velocity_autocorrelation_tensor, autocorrs
    else:
        return autocorrs

def get_autocorr_functions(r_long, f_long, r_tran, g_tran, time):
    """
    Return interpolated functions using the outputs of get_two_point_vel_corr_iso()

    Parameters
    ----------
    r_long
    f_long
    r_tran
    g_tran
    time

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

        # Make autocorrelation function an even function for curvature calculation
        r_long_tmp = np.concatenate((-np.flip(r_long_tmp, axis=0)[:-1], r_long_tmp))
        f_long_tmp = np.concatenate((np.flip(f_long_tmp, axis=0)[:-1], f_long_tmp))
        r_tran_tmp = np.concatenate((-np.flip(r_tran_tmp, axis=0)[:-1], r_tran_tmp))
        g_tran_tmp = np.concatenate((np.flip(g_tran_tmp, axis=0)[:-1], g_tran_tmp))

        f_spl = UnivariateSpline(r_long_tmp, f_long_tmp, s=0, k=3)  # longitudinal autocorrelation func.
        g_spl = UnivariateSpline(r_tran_tmp, g_tran_tmp, s=0, k=3)  # transverse autocorrelation func.

        fs.append(f_spl)
        gs.append((g_spl))
    return fs, gs

def get_autocorrelation_tensor_iso(r_long, f_long, r_tran, g_tran, time):
    """
    Returns autocorrelation tensor with isotropy assumption

    Parameters
    ----------
    r_long
    f_long
    r_tran
    g_tran
    time

    Returns
    -------

    """
    f, g = get_autocorr_functions(r_long, f_long, r_tran, g_tran, time)

    # Define Rij(r, t) as a function.
    def two_pt_velocity_autocorrelation_tensor(i, j, r, t, udata):
        dim = len(r)
        u2_avg = np.nanmean(udata ** 2, axis=tuple(range(dim + 1))) # spatial average
        if dim == 2:
            x, y = r[0], r[1]
        elif dim == 3:
            x, y, z = r[0], r[1], r[2]
        r2_norm = np.zeros_like(x)
        for k in range(dim):
            r2_norm += r[k] ** 2
        r_norm = np.sqrt(r2_norm)
        rij_value = u2_avg[t] * (g(r_norm, t) * klonecker_delta(i, j) + (f(r_norm, t)-g(r_norm , t)) * r[i] * r[j] / (r_norm ** 2))
        return rij_value

    rij = two_pt_velocity_autocorrelation_tensor
    return rij

def get_structure_function_long(udata, x, y, z=None, p=2, roll_axis=1, n_bins=None, nu=1.004, u='ux',
                                x0=0, x1=None, y0=0, y1=None, z0=0, z1=None, t0=0, t1=None,
                                coarse=1.0, coarse2=0.2, notebook=True):
    """
    Structure tensor Dij is essentially the covariance of the two-point velocity difference
    There is one-to-one correspondence between Dij and Rij. (Pope 6.36)
    This method returns the LONGITUDINAL STRUCTURE FUNCTION.
    If p=2, this returns D_LL.

    Parameters
    ----------
    udata
    x
    y
    z
    p
    roll_axis
    n_bins
    nu
    u
    x0
    x1
    y0
    y1
    z0
    z1
    t0
    t1
    coarse
    coarse2
    notebook

    Returns
    -------

    """
    if notebook:
        from tqdm import tqdm_notebook as tqdm
        print 'Using tqdm_notebook. If this is a mistake, set notebook=False'
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
        arr1, arr2 = zip(*sorted(zip(arr1, arr2)))
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
    # Some useful numbers for processing
    nrows, ncolumns = y1 - y0, x1 - x0
    limits = [ncolumns, nrows]
    if dim == 3:
        if z1 is None:
            z1 = ui.shape[2]
        nsteps =  z1 - z0
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

    for t in tqdm(range(t0, t1), desc='struc. func. time'):
        # Call velocity field at time t as uu
        uu = ui[..., t]

        # Initialization
        ## m: number of rolls it tries. coarse is a parameter to sample different rs evenly
        #### coarse=1: Compute DLL(r,t) for all possible r. if coarse=0.5, it samples only a half of possible rs.
        ## n: number of data points from which DLL statistics is computed.
        #### coarse2=1: use all data points. (e.g. for 1024*1024 grid, use 1024*1024*coarse2 data points)
        roll_indices = range(0, limits[roll_axis], int(1./coarse))
        m = len(roll_indices)
        n = int(x_grid.size * coarse2)

        rr = np.empty((n, m))
        Dxx = np.empty((n, m))

        for j, i in enumerate(tqdm(roll_indices, desc='computing DL^(%d)' % p)):
            # for i in range(int(coarse * limits[roll_axis])):
            uu_rolled = np.roll(uu, i, axis=roll_axis)
            if dim == 3:
                x_grid_rolled, y_grid_rolled, z_grid_rolled = np.roll(x_grid, i, axis=roll_axis), \
                                                          np.roll(y_grid, i, axis=roll_axis), \
                                                          np.roll(z_grid, i, axis=roll_axis)
                r_grid = np.sqrt( (x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2 + (z_grid_rolled - z_grid) ** 2)
            elif dim == 2:
                x_grid_rolled, y_grid_rolled = np.roll(x_grid, i, axis=roll_axis), np.roll(y_grid, i, axis=roll_axis)
                r_grid = np.sqrt( (x_grid_rolled - x_grid) ** 2 + (y_grid_rolled - y_grid) ** 2)


            uu_rolled = np.roll(uu, i, axis=roll_axis)
            Dxx_raw = (uu - uu_rolled) ** p
            rr[:, j] = r_grid.flatten()[:n]
            Dxx[:, j] = Dxx_raw.flatten()[:n]

        # flatten arrays to feed to binned_statistic
        rr_flatten, Dxx_flatten = rr.flatten(), Dxx.flatten()

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
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    eta = (nu **  3 / epsilon) ** 0.25
    rrs_scaled = rrs / eta
    Dxxs_scaled = Dxxs / ((epsilon * rrs) ** (float(p) / 3.))
    Dxx_errs_scaled = Dxx_errs / ((epsilon * rrs) ** (float(p) / 3.))
    if notebook:
        from tqdm import tqdm as tqdm
    return rrs, Dxxs, Dxx_errs, rrs_scaled, Dxxs_scaled, Dxx_errs_scaled


    # udata = fix_udata_shape(udata)
    #
    # if nn is None:
    #     n_bins = xx.shape[1]
    # else:
    #     n_bins = nn
    # rr, DLL = np.empty((xx.shape[0], xx.shape[1], n_bins)), np.empty((xx.shape[0], xx.shape[1], n_bins))
    # for i in range(n_bins):  # in x direction
    #     if i % 100 == 0:
    #         print '%d / %d' % (i, n_bins)
    #     ux_rolled = np.roll(ux0, i, axis=0)
    #     xx_rolled = np.roll(xx, i, axis=0)
    #     yy_rolled = np.roll(yy, i, axis=0)
    #
    #     DLL_raw = (ux0 - ux_rolled) ** p
    #     rr_raw = np.sqrt((xx - xx_rolled) ** 2 + (yy - yy_rolled) ** 2)
    #     rr[..., i], DLL[..., i] = rr_raw, DLL_raw
    # # rr, DLL = np.concatenate((rr, rr_raw.flatten())), np.concatenate((DLL, DLL_raw.flatten()))
    # rr, DLL = np.ravel(rr), np.ravel(DLL)
    # #     return rr, DLL
    #
    # # Binning
    # bin_centers, _, _ = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
    # bin_averages, _, _ = binned_statistic(rr, DLL, statistic='mean', bins=n_bins)
    # bin_stdevs, _, _ = binned_statistic(rr, DLL, statistic='std', bins=n_bins)
    #
    # # Scale
    # rr = bin_centers
    # Dx_p = bin_averages
    # rr_scaled = bin_centers / eta
    # Dx_p_scaled = Dx_p / ((epsilon * rr) ** (float(p) / 3.))

def scale_raw_structure_funciton_long(rrs, Dxxs, Dxx_errs, epsilon, nu=1.004, p=2):
    """
    Returns the scaled structure functions when raw structure function data are given
    ... This allows users to scale the structure functions with epsilon and nu input by the users.

    Parameters
    ----------
    rrs
    Dxxs
    Dxx_errs
    epsilon
    nu
    p

    Returns
    -------
    rrs_s, Dxxs_s, Dxx_errs_s: Scaled r, DLL, DLL error
    """
    Dxxs_s = Dxxs / (epsilon * rrs) ** (p / 3.)
    Dxx_errs_s = Dxx_errs / (epsilon * rrs) ** (p / 3.)
    eta = compute_kolmogorov_lengthscale_simple(epsilon, nu)
    rrs_s = rrs / eta
    return rrs_s, Dxxs_s, Dxx_errs_s

########## Length scales ##########
## TAYLOR MICROSCALES ##
# Taylor microscales 1: using autocorrelation functions
### DEFAULT ###
def remove_nans_for_array_pair(arr1, arr2):
    """
    remove nans or infs in arr1 and arrs, and returns the compressed arrays with the same length

    Parameters
    ----------
    arr1
    arr2

    Returns
    -------
    compressed_arr1, compressed_arr2
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

def get_taylor_microscales(r_long, f_long, r_tran, g_tran):
    """
    Returns Taylor microscales as the curvature of the autocorrelation functions at r=0
    ... First, it spline-interpolates the autocorr. functions, and compute the curvature at r=0
    ... Designed to use the results of get_two_point_vel_corr_iso().

    Parameters
    ----------
    r_long: numpy 2d array with shape (no. of elements, duration)
        ... r for longitudinal autoorrelation functon
    f_long: numpy 2d array with shape (no. of elements, duration)
        ... longitudinal autoorrelation values
    r_tran: numpy 2d array with shape (no. of elements, duration)
        ... r for longitudinal autoorrelation functon
    g_tran: numpy 2d array with shape (no. of elements, duration)
        ... longitudinal autoorrelation values

    Returns
    -------
    lambda_f: numpy 2d array with shape (duration, )
        Longitudinal Taylor microscale
    lambda_g: numpy 2d array with shape (duration, )
        Transverse Taylor microscale

    """
    n, duration = r_long.shape
    data = [r_long, f_long, r_tran, g_tran]
    # Remove nans if necessary
    for i, datum in enumerate(data):
        if ~np.isnan(data[i]).any():
            data[i] = data[i][~np.isnan(data[i])]
    # inter
    # polate data (3rd order spline)
    lambda_f, lambda_g = [], []
    for t in range(duration):
        # if r_long contains nans, UnivariateSpline fails. so clean it up.
        r_long_tmp, f_long_tmp = remove_nans_for_array_pair(r_long[:, t], f_long[:, t])
        r_tran_tmp, g_tran_tmp = remove_nans_for_array_pair(r_tran[:, t], g_tran[:, t])


        # Make sure that f(r=0, t)=g(r=0,t)=1
        f_long_tmp /= f_long_tmp[0]
        g_tran_tmp /= g_tran_tmp[0]


        # Make autocorrelation function an even function for curvature calculation
        r_long_tmp = np.concatenate((-np.flip(r_long_tmp, axis=0)[:-1], r_long_tmp))
        f_long_tmp = np.concatenate((np.flip(f_long_tmp, axis=0)[:-1], f_long_tmp))
        r_tran_tmp = np.concatenate((-np.flip(r_tran_tmp, axis=0)[:-1], r_tran_tmp))
        g_tran_tmp = np.concatenate((np.flip(g_tran_tmp, axis=0)[:-1], g_tran_tmp))


        f_spl = UnivariateSpline(r_long_tmp, f_long_tmp, s=0, k=3) # longitudinal autocorrelation func.
        g_spl = UnivariateSpline(r_tran_tmp, g_tran_tmp, s=0, k=3) # transverse autocorrelation func.


        # take the second derivate of the spline function
        f_spl_2d = f_spl.derivative(n=2)
        g_spl_2d = g_spl.derivative(n=2)

        lambda_f_ = (- f_spl_2d(0) / 2.) ** (-0.5)  # Compute long. Taylor microscale
        lambda_g_ = (- g_spl_2d(0) / 2.) ** (-0.5)  # Compute trans. Taylor microscale

        # # Show Taylor microscale (debugging)
        # fig, ax = plt.subplots(num=5)
        # ax.plot(r_tran_tmp, g_tran_tmp)
        # ax.plot(r_tran_tmp, g_spl_2d(0) / 2. * r_tran_tmp ** 2 + 1)
        # ax.set_ylim(-0.2, 1.1)
        # plt.show()

        lambda_f.append(lambda_f_)
        lambda_g.append(lambda_g_)
    return np.asarray(lambda_f), np.asarray(lambda_g)

# Taylor microscales 2: isotropic formula. Must know epsilon beforehand
def get_taylor_microscales_iso(udata, epsilon, nu=1.004):
    """
    Return Taylor microscales computed by isotropic formulae: lambda_g_iso = (15 nu * u_irms^2 / epsilon) ^ 0.5
    Parameters
    ----------
    udata: nd array
    epsilon: float or array with the same length as udata
    nu: float, viscoty

    Returns
    -------
    lambda_f_iso, lambda_g_iso
    """
    u_irms = get_characteristic_velocity(udata)
    lambda_g_iso = np.sqrt(15. * nu * u_irms ** 2 / epsilon)
    lambda_f_iso = np.sqrt(30. * nu * u_irms ** 2 / epsilon)
    return lambda_f_iso, lambda_g_iso


## INTEGRAL SCALES ##
# Integral scales 1: using autocorrelation functions
### DEFAULT ###
def get_integral_scales(r_long, f_long, r_tran, g_tran, method='trapz'):
    n, duration = r_long.shape
    data = [r_long, f_long, r_tran, g_tran]
    # Remove nans if necessary
    for i, datum in enumerate(data):
        if ~np.isnan(data[i]).any():
            data[i] = data[i][~np.isnan(data[i])]
    # interpolate data (3rd order spline)
    L11, L22 = [], []
    for t in range(duration):
        if method=='quad':
            rmin, rmax = np.nanmin(r_long), np.nanmax(r_long)
            f_spl = UnivariateSpline(r_long[:, t], f_long[:, t] / f_long[0, t], s=0,
                                     k=3)  # longitudinal autocorrelation func.
            g_spl = UnivariateSpline(r_tran[:, t], g_tran[:, t] / g_tran[0, t], s=0,
                                     k=3)  # transverse autocorrelation func.
            L11.append(integrate.quad(lambda r: f_spl(r), rmin, rmax)[0])
            L22.append(integrate.quad(lambda r: g_spl(r), rmin, rmax)[0])
        elif method=='trapz':
            L11.append(np.trapz(f_long[:, t], r_long[:, t]))
            L22.append(np.trapz(g_tran[:, t], r_tran[:, t]))
    return L11, L22

# Integral scales 2: using autocorrelation tensor. Should be equivalent to get_integral_scales()
def get_integral_scales_using_rij(udata, Rij, rmax, n=100):
    """
    Use autocorrelation tensor, Rij to calculate integral length scale
    Parameters
    ----------
    udata
    Rij
    rmax
    n

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
    Integral scale. Assumes isotropy and a full 1D energy spectrum. Pope 6.260.
    Parameters
    ----------
    udata
    e_k: an output of get_energy_spectrum()
    k: an output of get_energy_spectrum()

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
    dissipation rate, epsilon is proportional to u'^3 / L.

    Some just define an integral scale L as u'^3 / epsilon.
    This method just returns this integral scale. It is often interpreted as the characteristic scale of large eddies.

    Parameters
    ----------
    udata
    epsilon

    Returns
    -------

    """
    u_irms = get_characteristic_velocity(udata)
    L = u_irms ** 3 / epsilon
    return L

def get_integral_velocity_scale(udata):
    """
    Return integral velocity scale which is identical to u' (characteristic velocity)
    Parameters
    ----------
    udata

    Returns
    -------
    u_irms: See get_characteristic_velocity()
    """
    return get_characteristic_velocity(udata)

## KOLMOGOROV SCALES ##
### DEFAULT ###
def get_kolmogorov_scale(udata, dx, dy, dz, nu=1.004):
    """
    Returns kolmogorov LENGTh scale
    Parameters
    ----------
    udata
    dx
    dy
    dz
    nu

    Returns
    -------
    eta
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
    udata
    dx
    dy
    dz
    nu

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
    udata
    r_long
    f_long
    r_tran
    g_tran

    Returns
    -------
    lambda_f: 1d array
    u_lambda: 1d array
    tau_lambda: 1d array
    """
    lambda_f, lambda_g = get_taylor_microscales(r_long, f_long, r_tran, g_tran)
    u_lambda = get_characteristic_velocity(udata) # u_irms = u_lambda
    tau_lambda = lambda_g / u_lambda # other way to define the time scale is through temporal autocorrelation
    return lambda_g, u_lambda, tau_lambda

def get_kolmogorov_scales_all(udata, dx, dy, dz, nu=1.004):
    """
    Returns Kolmogorov scales
    Parameters
    ----------
    udata
    dx
    dy
    dz
    nu

    Returns
    -------
    eta: 1d array
    u_eta: 1d array
    tau_eta: 1d array
    """
    epsilon = get_epsilon_using_sij(udata, dx=dx, dy=dy, dz=dz, nu=nu)
    eta = (nu ** 3 / epsilon) ** 0.25
    u_eta = (nu * epsilon) ** 0.25
    tau_eta =  (nu / epsilon) ** 0.5
    return eta, u_eta, tau_eta


########## REYNOLDS NUMBERS ##########
def get_turbulence_re(udata, dx, dy, dz=None,  nu=1.004):
    """
    Returns turbulence reynolds number (Pope 6.59)
    Parameters
    ----------
    udata
    dx
    dy
    dz
    nu

    Returns
    -------
    Re_L
    """
    L, u_L, tau_L = get_integral_scales_all(udata, dx, dy, dz,  nu=nu)
    Re_L = u_L * L / nu
    return Re_L

def get_taylor_re(udata, r_long, f_long, r_tran, g_tran, nu=1.004):
    """
    Returns Taylor reynolds number (Pope 6.63)
    Parameters
    ----------
    udata
    dx
    dy
    dz
    nu

    Returns
    -------
    Re_L
    """
    lambda_g, u_irms, tau_lambda = get_taylor_microscales_all(udata, r_long, f_long, r_tran, g_tran)
    Re_lambda = u_irms * lambda_g / nu
    return Re_lambda


########## Sample velocity field ##########
def rankine_vortex_2d(xx, yy, x0=0, y0=0, gamma=1., a=1.):
    """
    Reutrns a 2D velocity field with a single Rankine vortex at (x0, y0)

    Parameters
    ----------
    xx
    yy
    x0
    y0
    gamma
    a

    Returns
    -------
    udata: (ux, uy)

    """
    rr, phi = cart2pol(xx - x0, yy - y0)

    cond = rr < a
    ux, uy = np.empty_like(rr), np.empty_like(rr)

    ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
    ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
    uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

    udata = np.stack((ux, uy))

    return udata


def rankine_vortex_line_3d(xx, yy, zz, x0=0, y0=0, gamma=1., a=1., uz0=0):
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
    rr, theta, phi = cart2sph(xx - x0, yy - y0, zz)

    cond = rr < a
    ux, uy, uz = np.empty_like(rr), np.empty_like(rr), np.empty_like(rr)

    ux[cond] = -gamma * rr[cond] / (2 * np.pi * a ** 2) * np.sin(phi[cond])
    uy[cond] = gamma * rr[cond] / (2 * np.pi * a ** 2) * np.cos(phi[cond])
    ux[~cond] = -gamma / (2 * np.pi * rr[~cond]) * np.sin(phi[~cond])
    uy[~cond] = gamma / (2 * np.pi * rr[~cond]) * np.cos(phi[~cond])

    uz = np.ones_like(ux) * uz0

    udata = np.stack((ux, uy, uz))

    return udata


def get_sample_turb_field_3d(return_coord=True):
    # get module location
    mod_loc = os.path.abspath(__file__)
    pdir, filename = os.path.split(mod_loc)
    datapath = os.path.join(pdir, 'reference_data/isoturb_slice2.h5')
    data = h5py.File(datapath, 'r')

    keys = data.keys()
    keys_u = [key for key in keys if 'u' in key]
    keys_u = natural_sort(keys_u)
    duration = len(keys_u)
    depth, height, width, ncomp = data[keys_u[0]].shape
    udata = np.empty((ncomp, height, width, depth, duration))

    Lx, Ly, Lz = 2 * np.pi, 2 * np.pi, 2 * np.pi
    dx = dy = dz = Lx / 1023


    for t in range(duration):
        udata_tmp = data[keys_u[t]]
        udata_tmp = np.swapaxes(udata_tmp, 0, 3)
        udata[..., t] = udata_tmp
    data.close()

    if return_coord:
        x, y, z = range(width), range(height), range(depth)
        xx, yy, zz = np.meshgrid(y, x, z)
        return udata, xx * dx, yy * dy, zz  * dz
    else:
        return udata


########## turbulence related stuff  ##########
def get_rescaled_energy_spectrum_saddoughi():
    """
    Returns values to plot rescaled energy spectrum from Saddoughi (1992)
    Returns
    -------

    """
    k = np.asarray([1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    e = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])
    return e, k

def get_energy_spectra_jhtd():
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    datapath = faqm_dir + '/reference_data/jhtd_e_specs.h5'

    datadict = {}
    with h5py.File(datapath, 'r') as data:
        keys = data.keys()
        for key in keys:
            if not '_s' in key:
                datadict[key] = data[key][...]

    return datadict

def get_rescaled_energy_spectra_jhtd():
    """
    Returns values to plot rescaled energy spectrum from Saddoughi (1992)
    Returns
    -------

    """
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    datapath = faqm_dir + '/reference_data/jhtd_e_specs.h5'

    datadict = {}
    with h5py.File(datapath, 'r') as data:
        keys = data.keys()
        for key in keys:
            if '_s' in key:
                datadict[key] = data[key][...]
    return datadict

def get_rescaled_structure_function_saddoughi(p=2):
    """
    Returns the values of rescaled structure function reported in Saddoughi and Veeravalli 1994 paper
    ... this is a curve about a specific Reynolds number! i.e. there is no universal structure function
    ----------
    p: int
    ... order of the structure function

    Returns
    -------
    r_scaled: nd array
    dll: nd array
    """
    faqm_dir = os.path.split(os.path.realpath(__file__))[0]
    if p==2:
        datapath = faqm_dir + '/reference_data/sv_struc_func.txt'
        data = np.loadtxt(datapath, skiprows=1, delimiter=',')
        r_scaled, dll = data[:, 0], data[:, 1]
        return r_scaled, dll
    else:
        print '... Only the rescaled, second-order structure function is available at the moment!'
        return None, None

########## FFT tools ########
def get_window_radial(xx, yy, zz=None, wtype='hamming', rmax=None, duration=None,
                    x0=0, x1=None, y0=0, y1=None, z0=0, z1=None,
                      n=500):
    """
    General function to get a window
    ...
    ... Window types:
        boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann,
        kaiser (needs beta), gaussian (needs standard deviation), general_gaussian (needs power, width),
        slepian (needs width), chebwin (needs attenuation), exponential (needs decay scale),
        tukey (needs taper fraction)

    Parameters
    ----------
    xx
    yy
    zz
    wtype
    rmax
    duration
    x0
    x1
    y0
    y1
    z0
    z1
    n

    Returns
    -------

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

        origin = ((xx[0, x1 - 1] + xx[0, x0]) / 2., (yy[y1 - 1, 0] + yy[y0, 0]) / 2., (zz[0, 0, z1-1] + zz[0, 0, z0]) / 2.)
        xx, yy, zz = xx[y0:y1, x0:x1, z0:z1], yy[y0:y1, x0:x1, z0:z1], zz[y0:y1, x0:x1, z0:z1]
        rr = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 + (zz - origin[2]) ** 2)

    if rmax is None:
        xmax, ymax = np.nanmax(xx[0, :]), np.nanmax(yy[:, 0])
        rmax = min(xmax, ymax)

    # x = rr + rmax
    # window = 0.54 - 0.46 * np.cos(2 * np.pi * (2 * rmax - x) / rmax / 2.)


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
        signal_intensity_loss =  np.nanmean(window_arr)
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
    window/windows: nd array,
    ... hamming window with shape

    """
    # Let the center of the grid be the origin
    if zz is None:
        dim = 2
        if x1 is None:
            x1 = xx.shape[1]
        if y1 is None:
            y1 = xx.shape[0]
        origin = ((xx[0, x1-1] - xx[0, x0]) / 2., (yy[y1-1, 0] - yy[y0, 0]) / 2.)
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

        origin = ((xx[0, x1-1] - xx[0, x0]) / 2., (yy[y1-1, 0] - yy[y0, 0]) / 2.)
        xx, yy, zz = xx[y0:y1, x0:x1, z0:z1], yy[y0:y1, x0:x1, z0:z1], zz[y0:y1, x0:x1, z0:z1]
        rr = np.sqrt((xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 + (zz - origin[2]) ** 2)

    if rmax is None:
        rmax = np.nanmax(rr)

    x = rr + rmax
    window = 0.54 - 0.46 * np.cos(2*np.pi * (2*rmax-x)/rmax/2.)
    window[rr > rmax] = 0
    if duration is not None:
        windows = np.repeat(window[..., np.newaxis], duration, axis=dim)
        return windows
    else:
        return window

# cleaning velocity field data
def clean_vdata(udata, cutoffU=2000, fill_value=np.nan, verbose=True):
    """
    Clean M class objects.
    Parameters
    ----------
    M
    cutoffU
    fill_value
    verbose

    Returns
    -------

    """
    udata_cleaned = np.empty_like(udata)
    print 'Cleaning M.Ux...'
    mask = get_mask_for_unphysical(udata[0, ...], cutoffU=cutoffU, fill_value=fill_value, verbose=verbose)
    Ux_filled_with_nans = fill_unphysical_with_sth(udata[0, ...], mask, fill_value=fill_value)
    Ux_interpolated = interpolate_using_mask(Ux_filled_with_nans, mask)
    udata_cleaned[0, ...]= Ux_interpolated[:]
    print 'Cleaning M.Uy...'
    mask = get_mask_for_unphysical(udata[1, ...], cutoffU=cutoffU, fill_value=fill_value, verbose=verbose)
    Uy_filled_with_nans = fill_unphysical_with_sth(udata[1, ...], mask, fill_value=fill_value)
    Uy_interpolated = interpolate_using_mask(Uy_filled_with_nans, mask)
    udata_cleaned[1, ...]= Uy_interpolated[:]
    print '...Cleaning Done.'
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
    mask: multidimensional boolean array

    """
    U = np.array(U)
    if verbose:
        print '...Note that nan/inf values in U are replaced by ' + str(fill_value)
        print '...number of invalid values (nan and inf) in the array: ' + str(np.isnan(U).sum() + np.isinf(U).sum())
        print '...number of nan values in U: ' + str(np.isnan(U).sum())
        print '...number of inf values in U: ' + str(np.isinf(U).sum()) + '\n'

    # Replace all nan and inf values with fill_value.
    # fix_invalid still enforces a mask on elements with originally invalid values
    U_fixed = ma.fix_invalid(U, fill_value=fill_value)
    n_invalid = ma.count_masked(U_fixed)
    if verbose:
        print '...number of masked elements by masked_invalid: ' + str(n_invalid)
    # Update the mask to False (no masking)
    U_fixed.mask = False



    # Mask unreasonable values of U_fixed
    b = ma.masked_greater(U_fixed, cutoffU)
    c = ma.masked_less(U_fixed, -cutoffU)
    n_greater = ma.count_masked(b) - n_invalid
    n_less = ma.count_masked(c)
    if verbose:
        print '...number of masked elements greater than cutoff: ' + str(n_greater)
        print '...number of masked elements less than -cutoff: ' + str(n_less)

    # Generate a mask for all nonsense values in the array U
    mask = ~(~b.mask * ~c.mask)

    d = ma.array(U_fixed, mask=mask)
    n_total = ma.count_masked(d)
    # U_filled = ma.filled(d, fill_value)

    #Total number of elements in U
    N = 1
    for i in range(len(U.shape)):
        N *= U.shape[i]
    print '...total number of unphysical values: ' + str(ma.count_masked(d)) + '  (' + str((float(n_total)/N*100)) + '%)\n'
    return mask

def fill_unphysical_with_sth(U, mask, fill_value=np.nan):
    """
    Returns an array whose elements are replaced by fill_value if its mask value is True
    Parameters
    ----------
    U   array-like
    mask   multidimensional boolean array
    fill_value   value that replaces masked values

    Returns
    -------
    U_filled  numpy array

    """
    U_masked = ma.array(U, mask=mask)
    U_filled = ma.filled(U_masked, fill_value)  # numpy array. This is NOT a masked array.

    return U_filled

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

########## misc ###########
def fix_udata_shape(udata):
    """
    It is better to always have udata with shape (height, width, depth, duration) (3D) or  (height, width, duration) (2D)
    This method fixes the shape of udata such that if the original shape is  (height, width, depth) or (height, width)
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

def get_equally_spaced_grid(udata, spacing=1):
    """
    Returns a grid to plot udata
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
        x, y = range(width), range(height)
        xx, yy = np.meshgrid(x, y)
        return xx * spacing, yy * spacing
    elif dim == 3:
        height, width, depth, duration = udata[0].shape
        x, y, z = range(width), range(height), range(depth)
        xx, yy, zz = np.meshgrid(y, x, z)
        return xx * spacing, yy * spacing, zz * spacing

def get_equally_spaced_kgrid(udata, dx=1):
    """
    Returns a grid to plot udata
    Parameters
    ----------
    udata
    spacing: spacing of the grid in the real space

    Returns
    -------
    xx, yy, (zz): 2D or 3D numpy arrays
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
        kxx, kyy = kxx * 2 * np.pi, kyy * 2 * np.pi # Convert inverse length into wavenumber
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

def kolmogorov_53(k, k0=50):
    """
    Customizable Kolmogorov Energy spectrum
    Parameters
    ----------
    k: array-like, wavenumber: convention is k= 1/L NOT 2pi/L
    k0: float, coefficient

    Returns
    -------
    e_k: power-law spectrum with exponent -5/3 for a given k and k0
    """
    e_k = k0*k**(-5./3)
    return e_k

def kolmogorov_53_uni(k, epsilon, c=1.5):
    """
    Universal Kolmogorov Energy spectrum
    Parameters
    ----------
    k: array-like, wavenumber
    epsilon: float, dissipation rate
    c: float, Kolmogorov constant c=1.5 (default)
    ... E(k) = c epsilon^(2/3) k^(-5/3)
    ... E11(k) = c1 epsilon^(2/3) k^(-5/3)
    ... E22(k) = c2 epsilon^(2/3) k^(-5/3)
    ... c1:c2:c = 1: 4/3: 55/18
    ... If c = 1.5, c1 = 0.491, c2 = 1.125
    ... Exp. values: c = 1.5, c1 = 0.5, c2 =

    Returns
    -------
    e_k: array-like, Kolmogorov energy spectrum for a given range of k
    """
    e_k = c*epsilon**(2./3)*k**(-5./3)
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

def klonecker_delta(i, j):
    """
    Klonecker Delta function
    Parameters
    ----------
    i
    j

    Returns
    -------
    0 or 1

    """
    if i == j:
        return 1
    else:
        return 0

def cart2pol(x, y):
    """
    Cartesian coord to polar coord
    Parameters
    ----------
    x
    y

    Returns
    -------
    r
    phi
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

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