
import numpy as np
import pandas as pd
import tflow.velocity as vel
import tflow.graph as graph
import os, sys, glob
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm as tqdm
import library.basics.formatstring as fs
from importlib import reload
import library.tools.rw_data as rw
import h5py


rho, nu = 0.001, 1.004 # g/mm3, mm2/s

datadir = os.path.join(os.getcwd(), 'data')
data_list = glob.glob(os.path.join(datadir, '*'))
savedir = os.path.join(os.getcwd(), 'figures')





# USER INPUTS
dpaths = [
          '/Users/takumi/Dropbox/Data_transfer/udata/STB_20210210_confinement_transition/l8p2mm_v200mms_f4hz_fr500hz_ext500us_10s_tr00.h5',
          # '/Users/takumi/Dropbox/Data_transfer/udata/Sep2020_NdYLF_four_cameras/l10p5mm_v200mms_f4hz_freq500hz_trig0s_05s_35A_tr00_064pxbinning.h5',
          ]
for dpath in dpaths:
    freq, frate = 4., 500
    R = np.nanmean(vel.estimate_radius_vring_collider(10.5, 200))
    Rblob = np.sqrt(6) * R
    r1s = [
               Rblob*0.5,
               Rblob*1.,
               Rblob*1.5,
               Rblob*2.,
               Rblob*2.5,
               Rblob * 3.,
    ]
    headers = [
                   #   'velocity_pdf_0p5Rblob/',
                   # 'velocity_pdf_1p0Rblob/',
                   # 'velocity_pdf_1p5Rblob/',
                   # 'velocity_pdf_2p0Rblob/',
                   # 'velocity_pdf_2p5Rblob/',
                   #   'velocity_pdf_3p0Rblob/',
        #            'velocity_pdf_whole/',
    ]

    headers2pdf = [
                    'vel_pdf2d_0p5Rblob/',
                    'vel_pdf2d_1p0Rblob/',
                   'vel_pdf2d_1p5Rblob/',
                   'vel_pdf2d_2p0Rblob/',
                   'vel_pdf2d_2p5Rblob/',
                'vel_pdf2d_3p0Rblob/',
                #    'vel_pdf2d_whole/',
    ]
    vmag = 250  # maximum velocity
    nbins = 100  # number of bins for PDF
    overwrite = True
    ###
    xxx, yyy, zzz, xc, yc, zc, tt = vel.read_data_from_h5(dpath, ['x', 'y', 'z', 'xc', 'yc', 'zc', 't'])
    Rblob = R * np.sqrt(6)
    rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)
    r1s.append(np.nanmax(rrr))

    inds = vel.get_suggested_inds(dpath)
    duration = vel.get_udata_dim(dpath)[-1]

    # REYNOLD DECOMPOSITION INGRIDIENTS
    ## CHOICE 1: TIME-AVG FIELD AS A MEAN FIELD
    if 'ux_m' not in vel.get_h5_keys(dpath):
        udata_m = vel.get_mean_flow_field_using_udatapath(dpath)
        vel.add_data2udatapath(dpath, {'ux_m': udata_m[0, ...], 'uy_m': udata_m[1, ...], 'uz_m': udata_m[2, ...]})
    ## CHOICE 2: PHASE-AVG FIELD AS A MEAN FIELD (MORE COMPLICATED)
    if 'ux_pavg' not in vel.get_h5_keys(dpath):
        tt, tp = vel.read_data_from_h5(dpath, ['t', 'tp'])
        tp, udata_pavg = vel.get_phase_averaged_udata_from_path(dpath, freq, tt, 1 / frate)
        vel.add_data2udatapath(dpath, {'tp': tp,
                                       'ux_pavg': udata_pavg[0, ...],
                                       'uy_pavg': udata_pavg[1, ...],
                                       'uz_pavg': udata_pavg[2, ...], })



    # ##############################################################################
    # # Data collection
    # # udata_m = vel.get_mean_flow_field_using_udatapath(dpath)
    #
    # etavg, xxx, yyy, zzz, er, rr, xc, yc, zc, tt = vel.read_data_from_h5(dpath, ['etavg', 'x', 'y', 'z',
    #                                                                              'eTimeThetaPhi_avg', 'r_energy',
    #                                                                              'xc', 'yc', 'zc', 't'])
    # ux_m, uy_m, uz_m = vel.read_data_from_h5(dpath, ['ux_m', 'uy_m', 'uz_m'])
    # udata_m = np.stack((ux_m, uy_m, uz_m))
    # x, y, z = xxx[0, :, 0], yyy[:, 0, 0], zzz[0, 0, :]
    #
    # rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)
    #
    # # r0, r1 = 0, np.nanmax(rrr) # whole
    # # r0, r1 = 0, 1.5*Rblob # whole
    #
    # r0 = 0  # compute pdf for r0<= r < r1
    # for r1, header in zip(r1s, headers):
    #     print(header, 'Ui')
    #     if not header[:-1] in vel.get_h5_keys(dpath) or 'ux_bins' not in vel.get_h5_subkeys(dpath, header[
    #                                                                                                :-1]) or overwrite:
    #         ux_pds, uy_pds, uz_pds = np.empty((nbins, duration)), np.empty((nbins, duration)), np.empty(
    #             (nbins, duration))
    #         for t in tqdm(range(duration)):
    #             udata = vel.get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)[..., 0]
    #             #     udata -= udata_m
    #             cond = np.logical_and(rrr >= r0, rrr < r1)
    #             fig11, ax11, ux_bins, ux_pds[:, t] = graph.pdf(udata[0, ...][cond], fignum=1, subplot=131,
    #                                                            label=f'$U_x$', return_data=True, vmin=-vmag,
    #                                                            vmax=vmag, nbins=nbins)
    #             fig11, ax12, uy_bins, uy_pds[:, t] = graph.pdf(udata[1, ...][cond], fignum=1, subplot=132,
    #                                                            label=f'$U_y$', return_data=True, vmin=-vmag,
    #                                                            vmax=vmag, nbins=nbins)
    #             fig11, ax13, uz_bins, uz_pds[:, t] = graph.pdf(udata[2, ...][cond], fignum=1, subplot=133,
    #                                                            label=f'$U_z$', figsize=(12, 5), return_data=True,
    #                                                            vmin=-vmag, vmax=vmag, nbins=nbins)
    #             plt.close()
    #
    #         datadict = {
    #             header + 'ux_bins': ux_bins,
    #             header + 'uy_bins': uy_bins,
    #             header + 'uz_bins': uz_bins,
    #             header + 'ux_pds': ux_pds,
    #             header + 'uy_pds': uy_pds,
    #             header + 'uz_pds': uz_pds,
    #             #             header + 'ux_t_bins': ux_bins,
    #             #             header + 'uy_t_bins': uy_bins,
    #             #             header + 'uz_t_bins': uz_bins,
    #             #             header + 'ux_t_pds': ux_pds,
    #             #             header + 'uy_t_pds': uy_pds,
    #             #             header + 'uz_t_pds': uz_pds,
    #             header + 't': tt,
    #             header + 'Rblob': Rblob,
    #         }
    #         vel.add_data2udatapath(dpath, datadict, overwrite=overwrite)
    #
    # print('... Ui Done')
    ##
    # ##############################################################################
    # # Data collection (fluctuating field using a time-avg field)
    # ux_m, uy_m, uz_m = vel.read_data_from_h5(dpath, ['ux_m', 'uy_m', 'uz_m'])
    # udata_m = np.stack((ux_m, uy_m, uz_m))
    # x, y, z = xxx[0, :, 0], yyy[:, 0, 0], zzz[0, 0, :]
    #
    # rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)
    #
    # r0 = 0  # compute pdf for r0<= r < r1
    # for r1, header in zip(r1s, headers):
    #     print(header, 'uit')
    #     if not header[:-1] in vel.get_h5_keys(dpath) or 'uz_t_bins' not in vel.get_h5_subkeys(dpath, header[
    #                                                                                                  :-1]) or overwrite:
    #         ux_pds, uy_pds, uz_pds = np.empty((nbins, duration)), np.empty((nbins, duration)), np.empty(
    #             (nbins, duration))
    #         for t in tqdm(range(duration)):
    #             udata = vel.get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)[..., 0]
    #             udata -= udata_m
    #             cond = np.logical_and(rrr >= r0, rrr < r1)  ### Joint PDF: $U_i-U_j$
    #             fig11, ax11, ux_bins, ux_pds[:, t] = graph.pdf(udata[0, ...][cond], fignum=1, subplot=131,
    #                                                            label=f'$U_x$', return_data=True, vmin=-vmag,
    #                                                            vmax=vmag, nbins=nbins)
    #             fig11, ax12, uy_bins, uy_pds[:, t] = graph.pdf(udata[1, ...][cond], fignum=1, subplot=132,
    #                                                            label=f'$U_y$', return_data=True, vmin=-vmag,
    #                                                            vmax=vmag, nbins=nbins)
    #             fig11, ax13, uz_bins, uz_pds[:, t] = graph.pdf(udata[2, ...][cond], fignum=1, subplot=133,
    #                                                            label=f'$U_z$', figsize=(12, 5), return_data=True,
    #                                                            vmin=-vmag, vmax=vmag, nbins=nbins)
    #             plt.close()
    #
    #         datadict = {
    #             #                     header + 'ux_bins': ux_bins,
    #             #                     header + 'uy_bins': uy_bins,
    #             #                     header + 'uz_bins': uz_bins,
    #             #                     header + 'ux_pds': ux_pds,
    #             #                     header + 'uy_pds': uy_pds,
    #             #                     header + 'uz_pds': uz_pds,
    #             header + 'ux_t_bins': ux_bins,
    #             header + 'uy_t_bins': uy_bins,
    #             header + 'uz_t_bins': uz_bins,
    #             header + 'ux_t_pds': ux_pds,
    #             header + 'uy_t_pds': uy_pds,
    #             header + 'uz_t_pds': uz_pds,
    #             header + 't': tt,
    #             header + 'Rblob': Rblob,
    #         }
    #         vel.add_data2udatapath(dpath, datadict, overwrite=overwrite)
    #
    # print('... uit Done')
    #
    # ##############################################################################
    # ux_pavg, uy_pavg, uz_pavg, tp = vel.read_data_from_h5(dpath, ['ux_pavg', 'uy_pavg', 'uz_pavg', 'tp'])
    # udata_pavg = np.stack((ux_pavg, uy_pavg, uz_pavg))
    #
    # # r0, r1 = 0, np.nanmax(rrr) # whole
    # # r0, r1 = 0, 1.5*Rblob # whole
    #
    # r0 = 0  # compute pdf for r0<= r < r1
    # for r1, header in zip(r1s, headers):
    #     print(header, 'uip')
    #     if not header[:-1] in vel.get_h5_keys(dpath) or 'ux_tp_bins' not in vel.get_h5_subkeys(dpath, header[
    #                                                                                                   :-1]) or overwrite:
    #         ux_pds, uy_pds, uz_pds = np.empty((nbins, duration)), np.empty((nbins, duration)), np.empty((nbins, duration))
    #         for t in tqdm(range(duration)):
    #             udata = vel.get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)[..., 0]
    #             phase_ind, _ = vel.find_nearest(tp, tt[t] % tp)
    #             udata -= udata_pavg[..., phase_ind]
    #             cond = np.logical_and(rrr >= r0, rrr < r1)
    #             fig11, ax11, ux_bins, ux_pds[:, t] = graph.pdf(udata[0, ...][cond], fignum=1, subplot=131,
    #                                                            label=f'$U_x$', return_data=True, vmin=-vmag,
    #                                                            vmax=vmag, nbins=nbins)
    #             fig11, ax12, uy_bins, uy_pds[:, t] = graph.pdf(udata[1, ...][cond], fignum=1, subplot=132,
    #                                                            label=f'$U_y$', return_data=True, vmin=-vmag,
    #                                                            vmax=vmag, nbins=nbins)
    #             fig11, ax13, uz_bins, uz_pds[:, t] = graph.pdf(udata[2, ...][cond], fignum=1, subplot=133,
    #                                                            label=f'$U_z$', figsize=(12, 5), return_data=True,
    #                                                            vmin=-vmag, vmax=vmag, nbins=nbins)
    #             plt.close()
    #
    #         datadict = {
    #             #                     header + 'ux_bins': ux_bins,
    #             #                     header + 'uy_bins': uy_bins,
    #             #                     header + 'uz_bins': uz_bins,
    #             #                     header + 'ux_pds': ux_pds,
    #             #                     header + 'uy_pds': uy_pds,
    #             #                     header + 'uz_pds': uz_pds,
    #             header + 'ux_tp_bins': ux_bins,
    #             header + 'uy_tp_bins': uy_bins,
    #             header + 'uz_tp_bins': uz_bins,
    #             header + 'ux_tp_pds': ux_pds,
    #             header + 'uy_tp_pds': uy_pds,
    #             header + 'uz_tp_pds': uz_pds,
    #             header + 'tp': tp,
    #             header + 'Rblob': Rblob,
    #         }
    #         vel.add_data2udatapath(dpath, datadict, overwrite=overwrite)
    #
    # print('... uip Done')
    ########################
    ########################
    ############ JOINT PDF
    nbins = 50
    duration = vel.get_udata_dim(dpath)[-1]
    etavg, xxx, yyy, zzz, er, rr, xc, yc, zc, tt = vel.read_data_from_h5(dpath,
                                                                         ['etavg', 'x', 'y', 'z', 'eTimeThetaPhi_avg',
                                                                          'r_energy', 'xc', 'yc', 'zc', 't'])
    ux_m, uy_m, uz_m = vel.read_data_from_h5(dpath, ['ux_m', 'uy_m', 'uz_m'])
    udata_m = np.stack((ux_m, uy_m, uz_m))
    x, y, z = xxx[0, :, 0], yyy[:, 0, 0], zzz[0, 0, :]

    rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)

    # ucutoff = vel.read_data_from_h5(dpath, ['abs_ui_99'])[0]
    ucutoff = vmag
    inds = vel.get_suggested_inds(dpath)

    for i, header in enumerate(headers2pdf):
        print(header)
        if not header[:-1] in vel.get_h5_keys(dpath) or '2dpdf_ux_uy' not in vel.get_h5_subkeys(dpath, header[
                                                                                                       :-1]) or overwrite:
            r1 = r1s[i]
            rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)

            # initialization
            shape = (nbins, nbins, duration)
            Hxy, Hyz, Hzx = np.empty(shape), np.empty(shape), np.empty(shape)
            for t in tqdm(range(duration)):
                udata = vel.get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)
                cond = rrr < r1
                udata_c = vel.clean_udata(udata, cutoff=ucutoff, median_filter=False, showtqdm=False)[..., 0]

                ux, uy, uz = udata_c[0][cond], udata_c[1][cond], udata_c[2][cond]
                # 2D-PDF: UX-UY
                Hxy[..., t], yedges, xedges = np.histogram2d(uy.flatten(), ux.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])
                # 2D-PDF: UY-UZ
                Hyz[..., t], zedges, yedges = np.histogram2d(uz.flatten(), uy.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])
                # 2D-PDF: UZ-UX
                Hzx[..., t], xedges, zedges = np.histogram2d(ux.flatten(), uz.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])

            datadict = {header + '2dpdf_ux_uy': Hxy,
                        header + '2dpdf_uy_uz': Hyz,
                        header + '2dpdf_uz_ux': Hzx,
                        header + 'ux': xedges,
                        header + 'uy': yedges,
                        header + 'uz': zedges,
                        }
            vel.add_data2udatapath(dpath, datadict, overwrite=overwrite)
    print('2d Ui done')


###################
    nbins = 50
    duration = vel.get_udata_dim(dpath)[-1]
    etavg, xxx, yyy, zzz, er, rr, xc, yc, zc, tt = vel.read_data_from_h5(dpath,
                                                                         ['etavg', 'x', 'y', 'z', 'eTimeThetaPhi_avg',
                                                                          'r_energy', 'xc', 'yc', 'zc', 't'])
    ux_m, uy_m, uz_m = vel.read_data_from_h5(dpath, ['ux_m', 'uy_m', 'uz_m'])
    udata_m = np.stack((ux_m, uy_m, uz_m))
    x, y, z = xxx[0, :, 0], yyy[:, 0, 0], zzz[0, 0, :]

    rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)

    # ucutoff = vel.read_data_from_h5(dpath, ['abs_ui_99'])[0]
    ucutoff = vmag
    inds = vel.get_suggested_inds(dpath)

    for i, header in enumerate(headers2pdf):
        print(header)
        if not header[:-1] in vel.get_h5_keys(dpath) or '2dpdf_uxt_uyt' not in vel.get_h5_subkeys(dpath, header[
                                                                                                         :-1]) or overwrite:
            r1 = r1s[i]
            cond = rrr < r1
            # initialization
            shape = (nbins, nbins, duration)
            Hxy, Hyz, Hzx = np.empty(shape), np.empty(shape), np.empty(shape)
            for t in tqdm(range(duration)):
                udata = vel.get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)
                udata_c = vel.clean_udata(udata, cutoff=ucutoff, median_filter=False, showtqdm=False)[..., 0]
                udata_c -= udata_m

                ux, uy, uz = udata_c[0][cond], udata_c[1][cond], udata_c[2][cond]
                # 2D-PDF: UX-UY
                Hxy[..., t], yedges, xedges = np.histogram2d(uy.flatten(), ux.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])
                # 2D-PDF: UY-UZ
                Hyz[..., t], zedges, yedges = np.histogram2d(uz.flatten(), uy.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])
                # 2D-PDF: UZ-UX
                Hzx[..., t], xedges, zedges = np.histogram2d(ux.flatten(), uz.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])

                # joint-pdf: Ux(r, theta, phi)

            datadict = {header + '2dpdf_uxt_uyt': Hxy,  # Ux-<Ux>_t
                        header + '2dpdf_uyt_uzt': Hyz,  # Uy-<Uy>_t
                        header + '2dpdf_uzt_uxt': Hzx,  # Uy-<Uz>_t
                        header + 'ux': xedges,
                        header + 'uy': yedges,
                        header + 'uz': zedges,
                        }
            vel.add_data2udatapath(dpath, datadict, overwrite=False)
    print('2d uit done')


#############
    nbins = 50
    duration = vel.get_udata_dim(dpath)[-1]
    etavg, xxx, yyy, zzz, er, rr, xc, yc, zc, tt, tp = vel.read_data_from_h5(dpath,
                                                                             ['etavg', 'x', 'y', 'z', 'eTimeThetaPhi_avg',
                                                                              'r_energy', 'xc', 'yc', 'zc', 't', 'tp'])
    ux_pavg, uy_pavg, uz_pavg = vel.read_data_from_h5(dpath, ['ux_pavg', 'uy_pavg', 'uz_pavg'])
    udata_pavg = np.stack((ux_pavg, uy_pavg, uz_pavg))
    x, y, z = xxx[0, :, 0], yyy[:, 0, 0], zzz[0, 0, :]

    rrr, tttheta, ppphi = vel.cart2sph(xxx - xc, yyy - yc, zzz - zc)

    # ucutoff = vel.read_data_from_h5(dpath, ['abs_ui_99'])[0]
    ucutoff = vmag
    inds = vel.get_suggested_inds(dpath)

    for i, header in enumerate(headers2pdf):
        print(header)
        if not header[:-1] in vel.get_h5_keys(dpath) or '2dpdf_uxt_uyt' not in vel.get_h5_subkeys(dpath,
                                                                                                  header[:-1]) or overwrite:
            r1 = r1s[i]
            cond = rrr < r1
            # initialization
            shape = (nbins, nbins, duration)
            Hxy, Hyz, Hzx = np.empty(shape), np.empty(shape), np.empty(shape)
            for t in tqdm(range(duration)):
                udata = vel.get_udata_from_path(dpath, t0=t, t1=t + 1, verbose=False)
                udata_c = vel.clean_udata(udata, cutoff=ucutoff, median_filter=False, showtqdm=False)[..., 0]

                phase_ind, _ = vel.find_nearest(tp, tt[t] % tp)
                udata_c -= udata_pavg[..., phase_ind]

                ux, uy, uz = udata_c[0][cond], udata_c[1][cond], udata_c[2][cond]
                # 2D-PDF: UX-UY
                Hxy[..., t], yedges, xedges = np.histogram2d(uy.flatten(), ux.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])
                # 2D-PDF: UY-UZ
                Hyz[..., t], zedges, yedges = np.histogram2d(uz.flatten(), uy.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])
                # 2D-PDF: UZ-UX
                Hzx[..., t], xedges, zedges = np.histogram2d(ux.flatten(), uz.flatten(), bins=nbins, density=True,
                                                             range=[[-ucutoff, ucutoff], [-ucutoff, ucutoff]])

                # joint-pdf: Ux(r, theta, phi)

            datadict = {header + '2dpdf_uxtp_uytp': Hxy,  # Ux-<Ux>_p
                        header + '2dpdf_uytp_uztp': Hyz,  # Uy-<Uy>_p
                        header + '2dpdf_uztp_uxtp': Hzx,  # Uy-<Uz>_p
                        header + 'ux': xedges,
                        header + 'uy': yedges,
                        header + 'uz': zedges,
                        }
            vel.add_data2udatapath(dpath, datadict, overwrite=False)
    print('2d uip done')
