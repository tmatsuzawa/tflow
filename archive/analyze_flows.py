import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import h5py
import argparse
import os
from decimal import Decimal
import sys
from tqdm import tqdm
from . import velocity as vel
from . import movie

import fapm.graph as graph
import fapm.box_param as box_param
import itertools

global color_cycle, box_param, Rij, ff, gg
colors = (['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*10)  #matplotliv v2.0
# Load box parameters for experiment.
box_params = box_param.get_box_params()


def compute_basics(hdf5datapath, overwrite=False, verbose=False, nu=1.004, use_fluc_vel=True):

    print('-------------BASIC QUANTITIES (E, Omega, etc.)------------------')
    def get_basics(udata, dx=None, dy=None):
        # REYNOLDS DECOMPOSITION
        udata_mean, udata_turb = vel.reynolds_decomposition(udata)

        # COMPUTE BASIC QUANTITIES
        omega = vel.curl(udata)
        omega_t = vel.curl(udata_turb)
        energy = vel.get_energy(udata) # energy density heat map
        enstrophy = vel.get_energy(udata)  # (omegaz squared)
        energy_t = vel.get_energy(udata_turb)  # turblent energy
        enstrophy_t = vel.get_energy(udata_turb)  # turblent enstrophy

        # spatially-averaged quantites
        energy_space_avg, energy_space_avg_err = vel.get_spatial_avg_energy(udata)
        enstrophy_space_avg, enstrophy_space_avg_err = vel.get_spatial_avg_enstrophy(udata)
        energy_t_space_avg, energy_t_space_avg_err = vel.get_spatial_avg_energy(udata_turb)
        enstrophy_t_space_avg, enstrophy_t_space_avg_err = vel.get_spatial_avg_enstrophy(udata_turb)

        ## time-averaged quantities
        energy_time_avg = vel.get_time_avg_energy(udata)
        enstrophy_time_avg = vel.get_time_avg_enstrophy(udata)
        ## energy_spectra
        # dx = dy = fyle['exp'].attrs['scale'] * fyle['exp'].attrs['W'] * 0.5 # 0.5 is overwrapping fraction of interrogation area in piv algorithm
        #
        print('----------------')
        print(dx, dy)
        print('----------------')
        e_k2d, (kx, ky) = vel.get_energy_spectrum_nd(udata_turb, dx=dx, dy=dy)
        e_k1d, e_k1d_err, k = vel.get_energy_spectrum(udata_turb, dx=dx, dy=dy, notebook=False)

        # may not be physically meaningful but use the original energy field to compute the spectrum
        e_k2d_org, (kx_org, ky_org) = vel.get_energy_spectrum_nd(udata, dx=dx, dy=dy)
        e_k1d_org, e_k1d_err_org, k_org = vel.get_energy_spectrum(udata, dx=dx, dy=dy, notebook=False)

        epsilon_test = [10**3, 10**5, 10**7]
        for i, epsilon in enumerate(epsilon_test):
            e_k1d_s, e_k1d_err_s, keta = vel.get_rescaled_energy_spectrum(udata_turb, dx=dx, dy=dy, epsilon=epsilon, notebook=False)
            e_k1d_s_org, e_k1d_err_s_org, keta_org = vel.get_rescaled_energy_spectrum(udata, dx=dx, dy=dy, epsilon=epsilon, notebook=False)

            if i == 0:
                shape1 = e_k1d_s.shape
                shape_e_ks = (len(epsilon_test), shape1[0], shape1[1])
                shape_ks = (len(epsilon_test), len(keta))
                e_k1d_s_test = np.empty(shape_e_ks)
                keta_test = np.empty(shape_ks)

                e_k1d_s_test_org = np.empty(shape_e_ks)
                keta_test_org = np.empty(shape_ks)
            e_k1d_s_test[i, ...] = e_k1d_s
            keta_test[i, :] = keta[:, 0]
            e_k1d_s_test_org[i, ...] = e_k1d_s_org
            keta_test_org[i, :] = keta_org[:, 0]
        epsilon_sij = vel.get_epsilon_using_sij(udata_turb, dx=dx, dy=dy, nu=nu)

        epsilon_sij_org = vel.get_epsilon_using_sij(udata, dx=dx, dy=dy, nu=nu)


        new_datasets = [udata_mean[0], udata_mean[1],
                        udata_turb[0], udata_turb[1],
                        omega, omega_t,
                        energy, enstrophy,
                        energy_t, enstrophy_t,
                        energy_space_avg, energy_space_avg_err,
                        enstrophy_space_avg,enstrophy_space_avg_err,
                        energy_t_space_avg, energy_t_space_avg_err,
                        enstrophy_t_space_avg, enstrophy_t_space_avg_err,
                        energy_time_avg, enstrophy_time_avg,
                        e_k2d, kx, ky,
                        e_k1d, e_k1d_err_s, k,
                        e_k1d_s_test, keta_test,
                        epsilon_sij,
                        e_k2d_org, kx_org, ky_org,
                        e_k1d_org, e_k1d_err_s_org, k_org,
                        e_k1d_s_test_org, keta_test_org,
                        epsilon_sij_org
                        ]
        return new_datasets


    print('Processing : ', os.path.split(hdf5datapath)[1])
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv']) # number of piv data for a given cine
        for ind in range(n_piv):
            key = 'piv%03d' % ind
            print('...', key)
            pivdata = fyle['piv'][key]



            new_datasetnames = ['ux_mean', 'uy_mean',
                                'ux_turb', 'uy_turb',
                                'omega', 'omega_t',
                                'energy', 'enstrophy',
                                'energy_t', 'enstrophy_t',
                                'energy_space_avg', 'energy_space_avg_err',
                                'enstrophy_space_avg', 'enstrophy_space_avg_err',
                                'energy_t_space_avg', 'energy_t_space_avg_err',
                                'enstrophy_t_space_avg', 'enstrophy_t_space_avg_err',
                                'energy_time_avg', 'enstrophy_time_avg',
                                'e_k2d', 'kx', 'ky',
                                'e_k1d', 'e_k1d_err', 'k',
                                'e_k1d_s_test', 'keta_test',
                                'epsilon_sij',
                                'e_k2d_org', 'kx_org', 'ky_org', # energy spectrum using raw velocity field (mm/s)
                                'e_k1d_org', 'e_k1d_err_org', 'k_org', # energy spectrum using raw velocity field (mm/s)
                                'e_k1d_s_test_org', 'keta_test_org', # energy spectrum using raw velocity field (mm/s)
                                'epsilon_sij_org' # dissipation computed using raw v-field
                                ]
            basics_is_computed = False
            for i, datasetname in enumerate(new_datasetnames):
                if datasetname in list(pivdata.keys()) and not overwrite:
                    if verbose:
                        print(datasetname, ' already exists under ', pivdata.name)
                        print('... skipping')
                else:
                    if not basics_is_computed:
                        print(datasetname , ' does not exist under ', pivdata.name)
                        print('... compute basic quantities')
                        # LOAD DATA
                        deltax, deltay = pivdata['deltax'][...], pivdata['deltay'][...]
                        ux, uy = pivdata['ux'], pivdata['uy']
                        udata = np.stack((ux, uy))

                        new_datasets = get_basics(udata, dx=deltax, dy=deltay)
                        basics_is_computed = True
                        if overwrite:
                            pivdata[datasetname][...] = new_datasets[i]
                        else:
                            pivdata.create_dataset(datasetname, data=new_datasets[i])
                    else:
                        if overwrite:
                            pivdata[datasetname][...] = new_datasets[i]
                        else:
                            pivdata.create_dataset(datasetname, data=new_datasets[i])

    print('... successfully computed basic quantities: energy, enstrophy, turb. energy spectrum etc.')


def draw_rec(ax, rect_x, rect_y, width, height, color='r', linewidth=10, alpha=0.7):
    rect = mpl.patches.Rectangle((rect_x, rect_y), width, height, linewidth=linewidth, edgecolor=color, facecolor='none', alpha=alpha)
    # Add the patch to the Axes
    # ax = plt.gca()
    ax.add_patch(rect)
    return rect

def ask_what_to_plot(hdf5datapath):
    """
    Asks users to type which piv data sets to show.

    Parameters
    ----------
    hdf5datapath: str, location of hdf5 data for a cine of intereest

    Returns
    -------
    indices: list, name of piv datasets to use. e.g. ['piv000', 'piv003']
    """
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        print('There are %d piv data files for this cine' % n_piv)
        for n in range(n_piv):
            key = 'piv%03d' % n
            m = fyle['piv'][key].attrs['Nr._of_passes']
            W, Dt, step = fyle['piv'][key].attrs['Int._area_%d' % m], fyle['piv'][key].attrs['Dt'], fyle['piv'][key].attrs['step']
            umin, umax = float(fyle['piv'][key].attrs['umin']), float(fyle['piv'][key].attrs['umax'])
            print('... piv%03d: (W, Dt, step, umin, umax) = (%d, %d, %d, %f, %f)' % (n, W, Dt, step, umin, umax))

        while True:
            try:
                n_piv_to_use = eval(input('Enter number of pivdata to use: '))
                if n_piv_to_use > n_piv:
                    print('... There are only %d piv data for this file. Try again.' % n_piv)
                else:
                    break
            except:
                print('... Invalid input. Type integers.')
        if n_piv_to_use == n_piv:
            indices = ['piv%03d' % ind for ind in range(n_piv_to_use)]
        else:
            indices = []
            while len(indices) < n_piv_to_use:
                try:
                    index = eval(input('Choose which piv_data to use. Type integers. piv___ : '))
                    if isinstance(index, int) and index < n_piv:
                        indices.append('piv%03d' % index)
                    elif index >= n_piv:
                        print('... Type integers between 0 and %d' % (n_piv - 1))

                    else:
                        print('... Non-integer input was received. Type integers.')

                except:
                    print('... Invalid input. Type integers.')
    return indices

def get_energy_spectrum_1d_int(hdf5datapath, indices, t0=0, save=True, show_org_spectrum=False):
    """
    Plots the 1D tuburlent energy spectrum
    Parameters
    ----------
    hdf5datapath
    indices
    t0
    save
    show_org_spectrum: bool
        If True, it shows the energy spectrum of the original velocity field.
        Turn this on if you suspect your mean velocity field is not accurate.
        For example, if you used only a few data points to extract mean flow, you probably UNDERSAMPLED.
        Thus, the mean flow is not accurate. Often, this results inaccurate energy spectrum.

    Returns
    -------

    """
    print('-------------Interactive 1D Energy Spectrum------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        n_piv_to_use = len(indices)
        for n in range(n_piv):
            key = 'piv%03d' % n
            m = fyle['piv'][key].attrs['Nr._of_passes']
            W, Dt, step = fyle['piv'][key].attrs['Int._area_%d' % m], fyle['piv'][key].attrs['Dt'], fyle['piv'][key].attrs['step']
            umin, umax = float(fyle['piv'][key].attrs['umin']), float(fyle['piv'][key].attrs['umax'])
            # print '... piv%03d: (W, Dt, step, umin, umax) = (%d, %d, %d, %f, %f)' % (n, W, Dt, step, umin, umax)

        pivdata = fyle['piv']
        scale = fyle['exp'].attrs['scale']

        # See all available arguments in matplotlibrc
        params = {'figure.figsize': (25, 10),
                  'font.size': 25,  # text
                  'legend.fontsize': 12,  # legend
                  'axes.labelsize': 12,  # axes
                  'axes.titlesize': 10,
                  'xtick.labelsize': 12,  # tick
                  'ytick.labelsize': 12}
        pylab.rcParams.update(params)

        # PLOT
        ## FIG1. ENERGY_SPECTRUM RELATED DATA
        fig1 = plt.figure(1)
        ax_e_spec_s = plt.subplot2grid((n_piv_to_use, 8), (0, 6), rowspan=n_piv_to_use, colspan=2)  # scaled spectrum
        axes_e_time_avg, axes_energy, axes_e_spec = [], [], []

        for i in range(n_piv_to_use):
            ax_e_time_avg = plt.subplot2grid((n_piv_to_use, 8), (i, 0), colspan=2)
            ax_energy = plt.subplot2grid((n_piv_to_use, 8), (i, 2), colspan=2)
            ax_e_spec = plt.subplot2grid((n_piv_to_use, 8), (i, 4), colspan=2)
            e_2d = pivdata[indices[i]]['energy']
            e_time_avg = pivdata[indices[i]]['energy_time_avg']
            x, y = pivdata[indices[i]]['x'], pivdata[indices[i]]['y']
            # (Turbulent) energy spectrum
            e_k, k = pivdata[indices[i]]['e_k1d'], np.asarray(pivdata[indices[i]]['k'])
            e_k_test, keta_test = pivdata[indices[i]]['e_k1d_s_test'], pivdata[indices[i]]['keta_test']
            epsilon_sij = pivdata[indices[i]]['epsilon_sij'][:]

            if show_org_spectrum:
                # (Original) energy spectrum
                e_k_org, k_org = pivdata[indices[i]]['e_k1d_org'], np.asarray(pivdata[indices[i]]['k_org'])
                e_k_test_org, keta_test_org = pivdata[indices[i]]['e_k1d_s_test_org'], pivdata[indices[i]]['keta_test_org']
                epsilon_sij_org = pivdata[indices[i]]['epsilon_sij_org'][:]


            # TIME-AVG ENERGY
            fig1, ax_e_time_avg, cc_e_avg = graph.color_plot(x, y, e_time_avg, ax=ax_e_time_avg, fignum=1)
            # ENERGY (SNAPSHOT) AT T=T0
            fig1, ax_energy, cc_e = graph.color_plot(x, y, e_2d[..., t0], ax=ax_energy, fignum=1)
            # RAW 1D ENERGY SPECTRA
            ax_e_spec.plot(k[:, t0], e_k[:, t0], color=colors[i])
            if show_org_spectrum:
                ax_e_spec.plot(k_org[:, t0], e_k_org[:, t0], color=colors[i])
                ax_e_spec.scatter(k_org[:, t0], e_k_org[:, t0], color=colors[i], s=10, marker='x')
            ax_e_spec.plot(k[:, t0], vel.kolmogorov_53_uni(k, epsilon_sij[t0]), color='k', label='$1.6\epsilon^{2/3}k^{-5/3}$')


            # RESCALED ENERGY SPECTRA(SNAPSHOT)
            for j in range(e_k_test.shape[0]):
                label = indices[i] + '   $\epsilon=10^%d$ $mm^2/s^2$' % (j+3)
                linestyles = ['-', '-.', ':']
                ax_e_spec_s.plot(keta_test[j, :], e_k_test[j, :, t0], color=colors[i], linestyle=linestyles[j])
                if show_org_spectrum:
                    ax_e_spec_s.plot(keta_test_org[j, :], e_k_test_org[j, :, t0], color=colors[i], linestyle=linestyles[j])
                    ax_e_spec_s.scatter(keta_test_org[j, :], e_k_test_org[j, :, t0], color=colors[i], s=10, marker='x')

            graph.tologlog(ax_e_spec)
            graph.add_colorbar(cc_e_avg, ax=ax_e_time_avg, option='scientific', label='$\\bar{E}$ ($mm^2/s^2$)')
            graph.add_colorbar(cc_e_avg, ax=ax_energy, option='scientific', label='$E$ ($mm^2/s^2$)')
            graph.title(ax_e_time_avg, indices[i] + '(time avg)')
            graph.title(ax_energy, indices[i] + '   $t = %.4f$ ($s$)' % pivdata[indices[i]]['t'][t0])
            graph.title(ax_e_spec_s, 'Color corresponds to each piv dataset.')
            graph.title(ax_e_spec, 'Estimated $\epsilon = {0:.2E}$ ($mm^2/s^3$)'.format(Decimal(str(epsilon_sij[t0]))))
            graph.labelaxes(ax_e_time_avg, '$x$ ($mm$)', '$y$ ($mm$)')
            graph.labelaxes(ax_energy, '$x$ ($mm$)', '$y$ ($mm$)')
            graph.labelaxes(ax_e_spec, '$k$ ($mm^{-1}$)', '$E(k)$ ($mm^3/s^2$)')

            graph.axvline(ax_e_spec, x=2*np.pi / box_params['L'], color='b', label='Box Size: $%d mm$' % int(box_params['L']))
            graph.axvline(ax_e_spec, x=2*np.pi / box_params['blob_size'], color='darkorange', label = 'Typical Blob Diameter: $100mm$')
            graph.axvline(ax_e_spec, x=2*np.pi / box_params['l_limit']  , color='r', label='Resolution Limit')
            graph.axvline(ax_e_spec, x=2*np.pi / box_params['eta_estimated'], color='g', label='Estimated $\eta=50 \mu m$')

            axes_e_time_avg.append(ax_e_time_avg)
            axes_energy.append(ax_energy)
            axes_e_spec.append(ax_e_spec)
            ax_e_spec.legend(loc=1, fontsize=8)

        e_saddoughi, k_saddoughi = vel.get_rescaled_energy_spectrum_saddoughi()
        ax_e_spec_s.plot(k_saddoughi, e_saddoughi, color='k', label='Saddoughi and Veeravalli')
        graph.tologlog(ax_e_spec_s)
        graph.labelaxes(ax_e_spec_s, '$k\eta$', '$E(k)/(\epsilon \\nu^5)^{1/4}$')
        ax_e_spec_s.legend(['Rescaled with $\epsilon=10^3$ ($mm^2/s^3$)',
                            'Rescaled with $\epsilon=10^5$ ($mm^2/s^3$)',
                            'Rescaled with $\epsilon=10^7$ ($mm^2/s^3$)'])


    # Add interactive features
    xcoords, ycoords, patches = [], [], []

    annots = []
    for ax in axes_e_time_avg:
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"), fontsize=10)
        annots.append(annot)
    # number of lines in the energy spec plot
    n_lines_spec_list = []
    for ax in axes_e_spec:
        n_lines_spec_list.append(len(ax.lines))
    def onclick(event):
        fig1.tight_layout()
        # MIRACULOUSLY, h5py closes the hdf5 file when matplotlib.mpl_connect is active. so open again.
        with h5py.File(hdf5datapath, 'a') as fyle:
            pivdata = fyle['piv']
            button = ['', 'left', 'middle', 'right']
            if event.inaxes is not None:
                ax = event.inaxes
                # DO FFT IF any ax in axes_energy is clicked
                if ax in axes_energy:
                    ax_ind = axes_energy.index(ax)

                    xcoords.append(event.xdata)
                    ycoords.append(event.ydata)
                    print(list(zip(xcoords, ycoords)), button[event.button])

                    if len(xcoords) % 2 == 0:
                        width, height = xcoords[-1] - xcoords[-2], ycoords[-1] - ycoords[-2]
                        rect = draw_rec(ax, xcoords[-2], ycoords[-2], width, height, color=colors[len(xcoords) % 20])
                        patches.append(rect)

                        deltax, deltay = pivdata[indices[ax_ind]]['deltax'][...], pivdata[indices[ax_ind]]['deltay'][...]
                        ux_t, uy_t = pivdata[indices[ax_ind]]['ux_turb'], pivdata[indices[ax_ind]]['uy_turb']
                        # data_spacing = pivdata[indices[ax_ind]].attrs['W'] / 2
                        udata = np.stack((ux_t, uy_t))
                        x0_tmp, x1_tmp, y0_tmp, y1_tmp = int(xcoords[-2] /deltax), int(
                            xcoords[-1] / deltax), \
                                                         int(ycoords[-2] / deltax), int(
                            ycoords[-1] / deltax)
                        x0, x1, y0, y1 = min(x0_tmp, x1_tmp), max(x0_tmp, x1_tmp), min(y0_tmp, y1_tmp), max(y0_tmp, y1_tmp)
                        e_k, e_k_err, k = vel.get_energy_spectrum(udata, x0, x1, y0, y1, dx=deltax, dy=deltax, notebook=False)
                        axes_e_spec[ax_ind].plot(k, e_k[:, t0], color=colors[len(xcoords) % 20])
                    if button[event.button] == 'right':
                        xcoords[:] = []
                        ycoords[:] = []
                        for patch in patches:
                            try:
                                patch.remove()
                            except:
                                pass
                        patches[:] = []

                        n_new_lines = len(axes_e_spec[ax_ind].lines)
                        # axes_e_spec[ax_ind].lines[0].remove()
                        for i in range(n_lines_spec_list[ax_ind], n_new_lines):
                            try:
                                axes_e_spec[ax_ind].lines[n_lines_spec_list[ax_ind]].remove()
                            except:
                                pass
                    fig1.canvas.draw()

                # Show piv params IF any ax in axes_e_time_avg is clicked
                elif ax in axes_e_time_avg:
                    ax_ind = axes_e_time_avg.index(ax)
                    annots[ax_ind].set_visible(False)
                    x, y = event.xdata, event.ydata
                    annots[ax_ind].xy = [x, y]
                    if button[event.button] == 'left':
                        pivparamnames = list(pivdata[indices[ax_ind]].attrs.keys())
                        pivparamvalues = list(pivdata[indices[ax_ind]].attrs.values())
                        text0 = str(dict(list(zip(pivparamnames, pivparamvalues))))
                        words = text0.split(', ')
                        text_list = []
                        for i, word in enumerate(words):
                            text_list.append(word.replace('u\'', '') + ', ')
                            if i % 3 == 0 and i !=0:
                                text_list.append(', \n')
                        text = ''.join(str(x) for x in text_list)
                        # text = 'test'
                        annots[ax_ind].set_text(text)
                        annots[ax_ind].set_visible(True)
                        # print ax.texts.pop()
                        fig1.texts.append(ax.texts[-1])
                        fig1.canvas.draw()
                    elif button[event.button] == 'right':
                        annots[ax_ind].set_visible(False)
                        fig1.canvas.draw()

    cid1 = fig1.canvas.mpl_connect('button_press_event', onclick)
    # cid2 = fig1.canvas.mpl_connect("motion_notify_event", hover)
    fig1.tight_layout()

    if save:
        savedir = os.path.split(os.path.split(hdf5datapath)[0])[0] + '/results/' + os.path.split(hdf5datapath)[1][:-3]
        graph.save(savedir + '/e_spec', ext='png')


def get_time_evolution(hdf5datapath, indices, t0=0, save=True):
    """

    Parameters
    ----------
    hdf5datapath
    t: int
        time index to plot velocity histogram at a certain time

    Returns
    -------

    """
    print('-------------Time-Evolution------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        n_piv_to_use = len(indices)
        pivdata = fyle['piv']
        scale = fyle['exp'].attrs['scale']
        # See all available arguments in matplotlibrc
        params = {'figure.figsize': (25, 10),
                  'font.size': 25,  # text
                  'legend.fontsize': 12,  # legend
                  'axes.labelsize': 12,  # axes
                  'axes.titlesize': 10,
                  'xtick.labelsize': 12,  # tick
                  'ytick.labelsize': 12}
        pylab.rcParams.update(params)



        # FIG2. Time-evolution of data (E vs t, k vs t, etc.)

        fig2 = plt.figure(num=2)
        axes_evst, axes_u_pdf = [], []
        for i in range(n_piv_to_use):
            time = pivdata[indices[i]]['t']
            ux = pivdata[indices[i]]['ux'][:]
            uy = pivdata[indices[i]]['uy'][:]
            omega = pivdata[indices[i]]['omega']
            omega_t = pivdata[indices[i]]['omega_t']

            energy_space_avg = pivdata[indices[i]]['energy_space_avg'] # spatially averaged energy
            # energy_space_avg_err = pivdata[indices[i]]['energy_space_avg_err'] # spatially averaged energy
            energy_t_space_avg = pivdata[indices[i]]['energy_t_space_avg'] # spatially averaged turbulent energy
            # energy_t_space_avg_err = pivdata[indices[i]]['energy_t_space_avg_err'] # spatially averaged turbulent energy

            enstrophy_space_avg = pivdata[indices[i]]['enstrophy_space_avg']
            # enstrophy_space_avg_err = pivdata[indices[i]]['enstrophy_space_avg_err']
            # enstrophy_t_space_avg = pivdata[indices[i]]['enstrophy_t_space_avg']
            # enstrophy_t_space_avg_err = pivdata[indices[i]]['enstrophy_t_space_avg_err']

            ax_evst = plt.subplot2grid((n_piv_to_use, 4), (i, 0))
            ax_evst.plot(time, energy_space_avg,label='$\\bar{E}$')
            ax_evst.plot(time, energy_t_space_avg, label='$\\bar{k}$')

            graph.tosemilogy(ax_evst)
            graph.labelaxes(ax_evst, 'time ($s$)', '$\\bar{E}, \\bar{k}$ ($mm^2/s^2$)')


            ax_evst.set_ylim(bottom=1, top=10**8)
            ax_evst.legend(loc=1, fontsize=10)
            ax_evst2 = ax_evst.twinx()
            emin, emax = ax_evst.get_ylim()
            ax_evst2.set_ylim(np.sqrt(emin), np.sqrt(emax))
            ax_evst2.set_ylabel('$u\'$ ($mm/s$)')
            ax_evst2.set_yscale('log')

            ## ENSTROPHY PDF
            ax_enstvst = plt.subplot2grid((n_piv_to_use, 4), (i, 1))
            ax_enstvst.plot(time, enstrophy_space_avg,label='$\\bar{\omega}_z^2$')
            # ax_enstvst.plot(time, enstrophy_t_space_avg, label='$\\bar{\omega}_{z, turb}^2$')
            graph.tosemilogy(ax_enstvst)
            graph.labelaxes(ax_enstvst, 'time ($s$)', '$\\bar{\omega}_z^2$ ($1/s^2$)')
            ax_enstvst.set_ylim(bottom=1, top=10**8)
            ax_enstvst.legend(loc=1, fontsize=10)

            ## VELOCITY PDF
            ax_u_pdf = plt.subplot2grid((n_piv_to_use, 4), (i, 2))
            graph.pdf(ux[..., t0], ax=ax_u_pdf, nbins=100, label='$u_x$')
            graph.pdf(uy[..., t0], ax=ax_u_pdf, nbins=100, label='$u_y$')
            graph.labelaxes(ax_u_pdf, '$u_i$ ($mm/s$)', 'Prob. density')
            graph.tosemilogy(ax_u_pdf)
            ax_u_pdf.legend(loc=1, fontsize=10)

            ## VORTICITY PDF
            ax_omega_pdf = plt.subplot2grid((n_piv_to_use, 4), (i, 3))
            graph.pdf(omega[..., t0], ax=ax_omega_pdf, nbins=100, label='$\omega_z$')
            graph.labelaxes(ax_omega_pdf, '$\omega_z$ ($1/s$)', 'Prob. density')
            graph.tosemilogy(ax_omega_pdf)
            ax_omega_pdf.legend(loc=1, fontsize=10)

        fig2.tight_layout()

        if save:
            savedir = os.path.split(os.path.split(hdf5datapath)[0])[0] + '/results/' + os.path.split(hdf5datapath)[1][:-3]
            graph.save(savedir + '/time_evo', ext='png')

def compute_autocorrelations(hdf5datapath, coarse=1.0, coarse2=1.0, overwrite=False, verbose=True):
    """

    Parameters
    ----------
    hdf5datapath
    coarse
    coarse2
    overwrite
    verbose

    Returns
    -------

    """
    def get_autocorrelations(udata_t, xx, yy, time, coarse=1.0, coarse2=0.2):
        # COMPUTE AUTOCORRELATION FUNCTIONS/TENSOR
        autocorrs = vel.get_two_point_vel_corr_iso(udata_t, xx, yy, time=time, n_bins=None, coarse=coarse, coarse2=coarse2, notebook=False, return_rij=False)
        r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran = autocorrs

        new_datasets = [r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran]
        return new_datasets

    print('-------------COMPUTE CORRELATIONS------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv']) # number of piv data for a given cine
        for ind in range(n_piv):
            key = 'piv%03d' % ind
            pivdata = fyle['piv'][key]

            ux_t, uy_t = pivdata['ux_turb'], pivdata['uy_turb']
            udata_t = np.stack((ux_t, uy_t))
            xx, yy = pivdata['x'], pivdata['y']
            time = np.asarray(pivdata['t'])

            new_datasetnames = ['r_long', 'f_long', 'f_err_long', 'r_tran', 'g_tran', 'g_err_tran']

            autocorr_is_computed = False

            for i, datasetname in enumerate(new_datasetnames):
                if datasetname in list(pivdata.keys()):
                    if verbose:
                        print(datasetname , ' already exists under ', pivdata.name)
                        print('... skipping')
                    if i == len(new_datasetnames)-1:
                        r_long = np.asarray(pivdata['r_long'][:])
                        f_long = np.asarray(pivdata['f_long'][:])
                        f_err_long = np.asarray(pivdata['f_err_long'][:])
                        r_tran = np.asarray(pivdata['r_tran'][:])
                        g_tran = np.asarray(pivdata['g_tran'][:])
                        g_err_tran = np.asarray(pivdata['g_err_tran'][:])
                        new_datasets = [r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran]
                else:
                    if not autocorr_is_computed:
                        new_datasets = get_autocorrelations(udata_t, xx, yy, time=time, coarse=coarse, coarse2=coarse2)
                        autocorr_is_computed = True

                    if overwrite:
                        pivdata[datasetname][...] = new_datasets[i]
                    else:
                        pivdata.create_dataset(datasetname, data=new_datasets[i])

            #
            # for i, datasetname in enumerate(new_datasetnames):
            #     if not datasetname in pivdata.keys():
            #         new_datasets = get_autocorrelations(udata_t, xx, yy, time=time, coarse=coarse, coarse2=coarse2)
            #         pivdata.create_dataset(datasetname, data=new_datasets[i])
            #         break
            #     elif overwrite:
            #         new_datasets = get_autocorrelations(udata_t, xx, yy, time=time, coarse=coarse, coarse2=coarse2)
            #         pivdata[datasetname][...] = new_datasets[i]
            #         break
            #     else:
            #         r_long = np.asarray(pivdata['r_long'][:])
            #         f_long = np.asarray(pivdata['f_long'][:])
            #         f_err_long = np.asarray(pivdata['f_err_long'][:])
            #         r_tran = np.asarray(pivdata['r_tran'][:])
            #         g_tran = np.asarray(pivdata['g_tran'][:])
            #         g_err_tran = np.asarray(pivdata['g_err_tran'][:])
            #         new_datasets = [r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran]
            #         if verbose:
            #             print datasetname , ' already exists under ', pivdata.name
            #             print '... skipping'

    # Function cannot be stored in HDF5. So always make one here
    r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran = new_datasets
    # ff, gg = vel.get_autocorr_functions(r_long, f_long, r_tran, g_tran, time)
    # rij = vel.get_autocorrelation_tensor_iso(r_long, f_long, r_tran, g_tran, time)
    # return rij, ff, gg
    return r_long, f_long, f_err_long, r_tran, g_tran, g_err_tran

def compute_lengthscales(hdf5datapath, indices, overwrite=False, verbose=False):

    def get_lengthscales_and_Re(udata_t, r_long, f_long, r_tran, g_tran, dx, dy, dz=None, nu=1.004):
        # COMPUTE LENGTHSCALES
        L_le, u_L, tau_L = vel.get_integral_scales_all(udata_t, dx, dy, dz, nu=nu)
        lambda_g, u_lambda, tau_lambda = vel.get_taylor_microscales_all(udata_t, r_long, f_long, r_tran, g_tran)
        eta, u_eta, tau_eta = vel.get_kolmogorov_scales_all(udata_t, dx, dy, dz, nu=nu)
        # COMPUTE REYNOLDS NUMBERES
        Re_L = vel.get_turbulence_re(udata_t, dx, dy, dz, nu=nu)
        Re_lambda = vel.get_taylor_re(udata_t, r_long, f_long, r_tran, g_tran, nu)

        new_datasets = [L_le, u_L, tau_L,
                        lambda_g, u_lambda, tau_lambda,
                        eta, u_eta, tau_eta,
                        Re_L, Re_lambda]

        return new_datasets


    print('-------------COMPUTE Length Scales------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        n_piv_to_use = len(indices)
        pivdata = fyle['piv']
        scale = fyle['exp'].attrs['scale']
        nu = float(fyle['exp'].attrs['nu'])

        # See all available arguments in matplotlibrc
        params = {'figure.figsize': (25, 10),
                  'font.size': 25,  # text
                  'legend.fontsize': 12,  # legend
                  'axes.labelsize': 12,  # axes
                  'axes.titlesize': 10,
                  'xtick.labelsize': 12,  # tick
                  'ytick.labelsize': 12}
        pylab.rcParams.update(params)

        for i, key in enumerate(indices):
            pivdata = fyle['piv'][key]

            time = pivdata['t']
            ux_t = np.asarray(pivdata['ux_turb'][:])
            uy_t = np.asarray(pivdata['uy_turb'][:])
            dx = pivdata['deltax'][...]
            dy = pivdata['deltay'][...]
            r_long = pivdata['r_long'][:]
            f_long =pivdata['f_long'][:]
            r_tran =pivdata['r_tran'][:]
            g_tran =pivdata['g_tran'][:]


            udata_t = np.stack((ux_t, uy_t))




            new_datasetnames = ['L_le', 'u_L', 'tau_L',
                                'lambda_g', 'u_lambda', 'tau_lambda',
                                'eta', 'u_eta', 'tau_eta',
                                're_l', 're_lambda']

            for i, datasetname in enumerate(new_datasetnames):
                if not datasetname in list(pivdata.keys()):
                    new_datasets = get_lengthscales_and_Re(udata_t, r_long, f_long, r_tran, g_tran, dx, dy, dz=None, nu=nu)
                    pivdata.create_dataset(datasetname, data=new_datasets[i])
                elif overwrite:
                    new_datasets = get_lengthscales_and_Re(udata_t, r_long, f_long, r_tran, g_tran, dx, dy, dz=None, nu=nu)
                    pivdata[datasetname][...] = new_datasets[i]
                else:
                    L_le = np.asarray(pivdata['L_le'][:])
                    u_L = np.asarray(pivdata['u_L'][:])
                    tau_L = np.asarray(pivdata['tau_L'][:])
                    lambda_g = np.asarray(pivdata['lambda_g'][:])
                    u_lambda = np.asarray(pivdata['u_lambda'][:])
                    tau_lambda = np.asarray(pivdata['tau_lambda'][:])
                    eta = np.asarray(pivdata['eta'][:])
                    u_eta = np.asarray(pivdata['u_eta'][:])
                    tau_eta = np.asarray(pivdata['tau_eta'][:])
                    Re_L = np.asarray(pivdata['re_l'][:])
                    Re_lambda = np.asarray(pivdata['re_lambda'][:])

                    new_datasets = [L_le, u_L, tau_L,
                                    lambda_g, u_lambda, tau_lambda,
                                    eta, u_eta, tau_eta,
                                    Re_L, Re_lambda]
                    if verbose:
                        print(datasetname, ' already exists under ', pivdata.name)
                        print('... skipping')

def get_lengthscales(hdf5datapath, indices=[], t0=0, save=True):
    print('-------------Length Scales------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        n_piv_to_use = len(indices)
        pivdata = fyle['piv']
        scale = fyle['exp'].attrs['scale']
        # See all available arguments in matplotlibrc
        params = {'figure.figsize': (25, 10),
                  'font.size': 25,  # text
                  'legend.fontsize': 12,  # legend
                  'axes.labelsize': 12,  # axes
                  'axes.titlesize': 10,
                  'xtick.labelsize': 12,  # tick
                  'ytick.labelsize': 12}
        pylab.rcParams.update(params)

        # FIG3. Length Scales
        fig3 = plt.figure(num=3)
        axes_ls, ax_relambda = [], []
        for i in range(n_piv_to_use):
            time = pivdata[indices[i]]['t']
            ux = pivdata[indices[i]]['ux'][:]
            uy = pivdata[indices[i]]['uy'][:]

            r_long = np.asarray(pivdata[indices[i]]['r_long'][:])
            f_long = np.asarray(pivdata[indices[i]]['f_long'][:])
            r_tran = np.asarray(pivdata[indices[i]]['r_tran'][:])
            g_tran = np.asarray(pivdata[indices[i]]['g_tran'][:])

            L_le = np.asarray(pivdata[indices[i]]['L_le'][:])
            u_L = np.asarray(pivdata[indices[i]]['u_L'][:])
            tau_L = np.asarray(pivdata[indices[i]]['tau_L'][:])
            lambda_g = np.asarray(pivdata[indices[i]]['lambda_g'][:])
            u_lambda = np.asarray(pivdata[indices[i]]['u_lambda'][:])
            tau_lambda = np.asarray(pivdata[indices[i]]['tau_lambda'][:])
            eta = np.asarray(pivdata[indices[i]]['eta'][:])
            u_eta = np.asarray(pivdata[indices[i]]['u_eta'][:])
            tau_eta = np.asarray(pivdata[indices[i]]['tau_eta'][:])
            Re_L = np.asarray(pivdata[indices[i]]['re_l'][:])
            Re_lambda = np.asarray(pivdata[indices[i]]['re_lambda'][:])


            ## PLOTTING
            ### autocorrelation functions
            ax_autocorr = plt.subplot2grid((n_piv_to_use, 3), (i, 0))
            ax_autocorr.plot(r_long[:, t0], f_long[:, t0], label='$f$: Longitudinal')
            ax_autocorr.plot(r_tran[:, t0], g_tran[:, t0], label='$g$: Transverse')
            graph.title(ax_autocorr, 'Autocorrelation functions')
            graph.labelaxes(ax_autocorr, '$r$ ($mm$)', '$f$, $g$, $t=%.3f$' % time[t0])
            ax_autocorr.legend(loc=1, fontsize=10)

            ### TIME-EVOLUTION OF LENGTH SCALES ###
            ax_ls = plt.subplot2grid((n_piv_to_use, 3), (i, 1))
            ax_ls.plot(time, L_le, label='$L_{large eddy}$')
            ax_ls.plot(time, lambda_g, label='$\lambda_g$')
            ax_ls.plot(time, eta, label='$\eta$')
            graph.tosemilogy(ax_ls)
            graph.labelaxes(ax_ls, 'time ($s$)', '$L_{large eddy}$, $\lambda_g$, $\eta$ (mm)')
            ax_ls.legend(loc=1, fontsize=10)

            ### TIME-EVOLUTION OF Reynolds numbers ###
            ax_Re = plt.subplot2grid((n_piv_to_use, 3), (i, 2))
            ax_Re.plot(time, Re_L, label='$Re_L=\\frac{k^{1/2} L_{large eddy}}{\\nu}$')
            ax_Re.plot(time, Re_lambda, label= '$Re_\lambda=\\frac{u\' \lambda}{\\nu}$')
            graph.tosemilogy(ax_Re)
            graph.labelaxes(ax_Re, 'time ($s$)', '$Re_L$, $Re_\lambda$')
            ax_Re.legend(loc=1, fontsize=10)
        fig3.tight_layout()

        if save:
            savedir = os.path.split(os.path.split(hdf5datapath)[0])[0] + '/results/' + os.path.split(hdf5datapath)[1][:-3]
            graph.save(savedir + '/lengthscales', ext='png')

def inspect_piv_outputs(hdf5datapath, indices=[], t0=0, save=True):
    print('-------------Inspect PIV outputs------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        n_piv_to_use = len(indices)
        pivdata = fyle['piv']
        scale = px2mm = fyle['exp'].attrs['scale']
        mm2px = 1./ px2mm
        # See all available arguments in matplotlibrc
        params = {'figure.figsize': (25, 10),
                  'font.size': 25,  # text
                  'legend.fontsize': 12,  # legend
                  'axes.labelsize': 12,  # axes
                  'axes.titlesize': 10,
                  'xtick.labelsize': 12,  # tick
                  'ytick.labelsize': 12}
        pylab.rcParams.update(params)


        # FIG4. PIV outputs
        fig4 = plt.figure(num=4)
        axes_u, axes_u_pdf = [], []
        for i in range(n_piv_to_use):
            time = np.asarray(pivdata[indices[i]]['t'])
            frame2sec = np.asarray(pivdata[indices[i]]['deltat'])
            sec2frame = 1. / frame2sec
            ux_raw = pivdata[indices[i]]['ux'][:] * mm2px / sec2frame
            uy_raw = pivdata[indices[i]]['uy'][:] * mm2px / sec2frame

            ux_rms = np.sqrt(np.nanmean(ux_raw ** 2, axis=(0, 1)))
            uy_rms = np.sqrt(np.nanmean(uy_raw ** 2, axis=(0, 1)))

            # RMS velocity in px/frame
            ax_u = plt.subplot2grid((n_piv_to_use, 2), (i, 0))  # scaled spectrum
            ax_u.plot(time, ux_rms, label='$u_{x,rms}$')
            ax_u.plot(time, uy_rms, label='$u_{y,rms}$')
            graph.labelaxes(ax_u, 'time ($s$)', '$u_i$ ($px/frame$)')
            ax_u.legend(loc=1, fontsize=10)

            ## VELOCITY PDF
            ax_u_pdf = plt.subplot2grid((n_piv_to_use, 2), (i, 1))
            graph.pdf(ux_raw[..., t0], ax=ax_u_pdf, nbins=100, label='$u_x$')
            graph.pdf(uy_raw[..., t0], ax=ax_u_pdf, nbins=100, label='$u_y$')
            graph.labelaxes(ax_u_pdf, '$u_i$ ($px/frame$)', 'Prob. density')
            graph.tosemilogy(ax_u_pdf)
            graph.title(ax_u_pdf, '$t=%.4f$ $s$' % time[t0])
            ax_u_pdf.legend(loc=1, fontsize=10)

            axes_u.append(ax_u)
            axes_u_pdf.append(ax_u_pdf)

        fig4.tight_layout()

        if save:
            savedir = os.path.split(os.path.split(hdf5datapath)[0])[0] + '/results/' + os.path.split(hdf5datapath)[1][:-3]
            graph.save(savedir + '/inspect_piv_outputs', ext='png')



def make_movie_espec(hdf5datapath, indices=[], framerate=10):
    print('-------------1D Energy Spectrum Movie------------------')
    with h5py.File(hdf5datapath, 'a') as fyle:
        n_piv = len(fyle['piv'])  # number of piv data for a given cine
        n_piv_to_use = len(indices)
        for n in range(n_piv):
            key = 'piv%03d' % n
            m = fyle['piv'][key].attrs['Nr._of_passes']
            W, Dt, step = fyle['piv'][key].attrs['Int._area_%d' % m], fyle['piv'][key].attrs['Dt'], \
                          fyle['piv'][key].attrs['step']
            umin, umax = float(fyle['piv'][key].attrs['umin']), float(fyle['piv'][key].attrs['umax'])
            # print '... piv%03d: (W, Dt, step, umin, umax) = (%d, %d, %d, %f, %f)' % (n, W, Dt, step, umin, umax)

        pivdata = fyle['piv']
        scale = fyle['exp'].attrs['scale']

        # See all available arguments in matplotlibrc
        params = {'figure.figsize': (25, 10),
                  'font.size': 25,  # text
                  'legend.fontsize': 12,  # legend
                  'axes.labelsize': 12,  # axes
                  'axes.titlesize': 10,
                  'xtick.labelsize': 12,  # tick
                  'ytick.labelsize': 12,
                  'lines.linewidth': 5} # lines
        pylab.rcParams.update(params)

        # PLOT
        ## FIG10. ENERGY_SPECTRUM RELATED DATA

        axes_e_time_avg, axes_energy, axes_e_spec = [], [], []

        for i in range(n_piv_to_use):
            e_2d = pivdata[indices[i]]['energy']
            e_time_avg = pivdata[indices[i]]['energy_time_avg']
            x, y = pivdata[indices[i]]['x'], pivdata[indices[i]]['y']
            e_k, k = pivdata[indices[i]]['e_k1d'], np.asarray(pivdata[indices[i]]['k'])
            e_k_test, keta_test = pivdata[indices[i]]['e_k1d_s_test'], pivdata[indices[i]]['keta_test']
            epsilon_sij = pivdata[indices[i]]['epsilon_sij'][:]
            time = np.asarray(pivdata[indices[i]]['t'])

            e_saddoughi, k_saddoughi = vel.get_rescaled_energy_spectrum_saddoughi()


            e_2d_max = np.nanmax(e_2d)
            e_k_min, e_k_max = np.nanmin(e_k), np.nanmax(e_k)
            e_k_s_min, e_k_s_max = np.nanmin(e_saddoughi), np.nanmax(e_saddoughi)


            for t0 in range(len(time)):
                fig10 = plt.figure(10)
                ax_e_spec_s = plt.subplot2grid((1, 8), (0, 6), rowspan=1, colspan=2)  # scaled spectrum
                ax_e_time_avg = plt.subplot2grid((1, 8), (i, 0), colspan=2)
                ax_energy = plt.subplot2grid((1, 8), (i, 2), colspan=2)
                ax_e_spec = plt.subplot2grid((1, 8), (i, 4), colspan=2)

                # TIME-AVG ENERGY
                fig10, ax_e_time_avg, cc_e_avg = graph.color_plot(x, y, e_time_avg, ax=ax_e_time_avg, fignum=10)
                # ENERGY (SNAPSHOT) AT T=T0
                fig10, ax_energy, cc_e = graph.color_plot(x, y, e_2d[..., t0], ax=ax_energy, fignum=10, vmin=0, vmax=e_2d_max)
                # RAW 1D ENERGY SPECTRA
                ax_e_spec.plot(k[:, t0], e_k[:, t0], color=colors[i])
                ax_e_spec.plot(k[:, t0], vel.kolmogorov_53_uni(k, epsilon_sij[t0]), color='k',
                               label='$1.6\epsilon^{2/3}k^{-5/3}$')

                # RESCALED ENERGY SPECTRA(SNAPSHOT)
                for j in range(e_k_test.shape[0]):
                    label = indices[i] + '   $\epsilon=10^%d$ $mm^2/s^2$' % (j + 3)
                    linestyles = ['-', '-.', ':']
                    ax_e_spec_s.plot(keta_test[j, :], e_k_test[j, :, t0], color=colors[i], linestyle=linestyles[j])

                graph.tologlog(ax_e_spec)
                graph.add_colorbar(cc_e_avg, ax=ax_e_time_avg, option='scientific', label='$\\bar{E}$ ($mm^2/s^2$)')
                graph.add_colorbar(cc_e_avg, ax=ax_energy, option='scientific', label='$E$ ($mm^2/s^2$)')
                graph.title(ax_e_time_avg, indices[i] + '(time avg)')
                graph.title(ax_energy, indices[i] + '   $t = %.4f$ ($s$)' % pivdata[indices[i]]['t'][t0])
                graph.title(ax_e_spec_s, 'Color corresponds to each piv dataset.')
                graph.title(ax_e_spec, 'Estimated $\epsilon = {0:.2E}$ ($mm^2/s^3$)'.format(Decimal(str(epsilon_sij[t0]))))
                graph.labelaxes(ax_e_time_avg, '$x$ ($mm$)', '$y$ ($mm$)')
                graph.labelaxes(ax_energy, '$x$ ($mm$)', '$y$ ($mm$)')
                graph.labelaxes(ax_e_spec, '$k/(2\pi)$ ($mm^{-1}$)', '$E(k)$ ($mm^3/s^2$)')

                graph.axvline(ax_e_spec, x=2*np.pi / box_params['L'], color='b',
                              label='Box Size: $%d mm$' % int(box_params['L']))
                graph.axvline(ax_e_spec, x=2*np.pi / box_params['blob_size'], color='darkorange',
                              label='Typical Blob Diameter: $100mm$')
                graph.axvline(ax_e_spec, x=2*np.pi / (box_params['l_limit']*8.), color='purple', label='Resolution Limit $\\times 8$')
                graph.axvline(ax_e_spec, x=2*np.pi / box_params['l_limit'], color='r', label='Resolution Limit')
                graph.axvline(ax_e_spec, x=2*np.pi / box_params['eta_estimated'], color='g',
                              label='Estimated $\eta=50 \mu m$')






                axes_e_time_avg.append(ax_e_time_avg)
                axes_energy.append(ax_energy)
                axes_e_spec.append(ax_e_spec)
                ax_e_spec.legend(loc=1, fontsize=8)
                ax_e_spec_s.plot(k_saddoughi, e_saddoughi, color='k', label='Saddoughi and Veeravalli')
                graph.tologlog(ax_e_spec_s)
                graph.labelaxes(ax_e_spec_s, '$k\eta / (2 \pi)$', '$E(k)/(\epsilon \\nu^5)^{1/4}$')
                ax_e_spec_s.legend(['Rescaled with $\epsilon=10^3$ ($mm^2/s^3$)',
                                    'Rescaled with $\epsilon=10^5$ ($mm^2/s^3$)',
                                    'Rescaled with $\epsilon=10^7$ ($mm^2/s^3$)'])

                # formatting
                ax_e_spec.set_ylim(e_k_min, e_k_max)
                ax_e_spec_s.set_ylim(e_k_s_min, e_k_s_max)

                fig10.tight_layout()


                savedir = os.path.split(os.path.split(hdf5datapath)[0])[0] + '/results/' + os.path.split(hdf5datapath)[1][
                                                                                       :-3]
                graph.save(savedir + '/' + indices[i] + '/e_spec_movie/im%05d' % t0, ext='png')
                plt.close(fig10)

            imgdir = savedir + '/' + indices[i] + '/e_spec_movie/'
            movname = savedir + '/' + indices[i] + '/e_spec_movie'
            movie.make_movie(imgname=imgdir + 'im', movname=movname, indexsz='05', framerate=framerate)

def main(hdf5datapath, t=0, overwrite=False, verbose=False, save=True):
    compute_basics(hdf5datapath, overwrite=overwrite, verbose=verbose)

    print('-------------------------------------------------------------')
    indices = ask_what_to_plot(hdf5datapath)


    # ENERGY SPECTRA + TIME-AVERAGED ENERGY
    get_energy_spectrum_1d_int(hdf5datapath, indices, t0=t, save=save, show_org_spectrum=False)

    # # make a movie
    # make_movie_espec(hdf5datapath, indices)



    # # TIME EVOLUTION
    # get_time_evolution(hdf5datapath, indices=indices, save=save)
    #
    #
    # # Computationally expensive task 1: Autocorrelation functions
    # compute_autocorrelations(hdf5datapath, coarse=1.0, coarse2=0.2,  overwrite=overwrite)
    #
    #
    # # LENGTHSCALES AND REYNOLDS NUMBERS
    # ## TIME-EVOLUTION OF LENGTHSCALES AND REYNOLDS NUMBERS
    # compute_lengthscales(hdf5datapath, indices, overwrite=overwrite, verbose=False)
    #
    # get_lengthscales(hdf5datapath, indices, t0=t, save=save)

    ## COMPARISON OF COMPUTATION METHODS OVER LENGTHSCALES




    # Computationally expensive task 2: Structure Function


    ## SPECTRAL ANALYSIS


    ## INSPECT PIV QUALITIES
    inspect_piv_outputs(hdf5datapath, indices, t0=t, save=save)




    graph.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
    parser.add_argument('-h5data', '--h5data', help='hdf5 file', type=str,
                        default='/Volumes/labshared4/takumi/old_data/sample_piv_cine/hdf5data/PIV_fv_vp_left_micro105mm_fps2000_Dp120p0mm_D25p6mm_piston10p5mm_freq5Hz_v200mms_setting1_inj1p0s_trig5p0s_fx0p0615mmpx.h5')
    parser.add_argument('-overwrite', '--overwrite',
                        help='overwrite pivlab outputs. This is handy if code fails and force code to insert pivlab outputs to hdf5. Default: False',
                        type=bool,
                        default=False)
    parser.add_argument('-verbose', '--verbose',
                        help='True or False',
                        type=bool,
                        default=False)
    parser.add_argument('-t', '--t',
                        help='Time at which data are plotted',
                        type=int,
                        default=0)
    parser.add_argument('-save', '--save',
                        help='Save figures. Default: True',
                        type=bool,
                        default=True)
    args = parser.parse_args()
    main(args.h5data, overwrite=args.overwrite, t=args.t, save=args.save)


