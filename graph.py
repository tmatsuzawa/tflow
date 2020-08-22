'''
Module for plotting and saving figures
'''
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import mpl_toolkits.axes_grid as axes_grid
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits
import itertools
from scipy import stats
from scipy.optimize import curve_fit
from scipy import interpolate
import numpy as np
import glob
from fractions import Fraction
from math import modf
import pickle
import copy

import h5py
import ilpm.vector as vec
# comment this and plot_fit_curve if it breaks
import library.basics.std_func as std_func

#Global variables
#Default color cycle: iterator which gets repeated if all elements were exhausted
#__color_cycle__ = itertools.cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
__def_colors__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
__color_cycle__ = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])  #matplotliv v2.0
__old_color_cycle__ = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  #matplotliv classic
__fontsize__ = 16
__figsize__ = (8, 8)
cmap = 'magma'

# See all available arguments in matplotlibrc
params = {'figure.figsize': __figsize__,
          'font.size': __fontsize__,  #text
        'legend.fontsize': 12, # legend
         'axes.labelsize': __fontsize__, # axes
         'axes.titlesize': __fontsize__,
         'xtick.labelsize': __fontsize__, # tick
         'ytick.labelsize': __fontsize__,
          'lines.linewidth': 5}


## Save a figure
def save(path, ext='pdf', close=False, verbose=True, fignum=None, dpi=None, overwrite=True, tight_layout=False,
         savedata=True, transparent=True, bkgcolor='w', **kwargs):
    """Save a figure from pyplot
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    if fignum == None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum)
    if dpi is None:
        dpi = fig.dpi

    if tight_layout:
        fig.tight_layout()

    # Separate a directory and a filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # path where the figure is saved
    savepath = os.path.join(directory, filename)
    # if a figure already exists AND you'd like to overwrite, name a figure differently
    ver_no = 0
    while os.path.exists(savepath) and not overwrite:
        # this needs to be fixed. right now, it keeps saving to _000.png
        savepath = directory + '/' + os.path.split(path)[1]+ '_%03d.' % ver_no + ext
        ver_no += 1

    if verbose:
        print(("Saving figure to '%s'..." % savepath))

    # Save the figure
    plt.savefig(savepath, dpi=dpi, transparent=transparent, **kwargs)

    # Save fig instance... This may fail for python2
    if savedata:
        try:
            pickle.dump(fig, open(savepath[:-len(ext)-1] + '_fig.pkl', 'wb'))
        except:
            print('... Could not save a fig instance')

    # Close it
    if close:
        plt.close(fignum)

    if verbose:
        print("... Done")


## Create a figure and axes
def set_fig(fignum, subplot=None, dpi=100, figsize=None, **kwargs):
    """
    Make a plt.figure instance and makes an axes as an attribute of the figure instance
    Returns figure and ax
    Parameters
    ----------
    fignum
    subplot
    dpi
    figsize
    kwargs

    Returns
    -------
    fig
    ax

    """

    if fignum == -1:
        if figsize is not None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
        else:
            fig = plt.figure(dpi=dpi)
    if fignum == 0:
        fig = plt.cla()  #clear axis
    if fignum > 0:
        if figsize is not None:
            fig = plt.figure(num=fignum, dpi=dpi, figsize=figsize)
            fig.set_size_inches(figsize[0], figsize[1])
        else:
            fig = plt.figure(num=fignum, dpi=dpi)
        fig.set_dpi(dpi)

    if subplot is not None:
        # a triplet is expected !
        ax = fig.add_subplot(subplot, **kwargs)
        return fig, ax
    else:
        ax = fig.add_subplot(111, **kwargs)
        return fig, ax


def plotfunc(func, x, param, fignum=1, subplot=111, ax = None, label=None, color=None, linestyle='-', legend=False, figsize=None, **kwargs):
    """
    plot a graph using the function fun
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    # y = func(x, a, b)
    if len(param)==1:
        a=param[0]
        y = func(x, a)
    if len(param) == 2:
        a, b = param[0], param[1]
        y = func(x, a, b)
    if len(param) == 3:
        a, b, c = param[0], param[1], param[2]
        y = func(x, a, b, c)
    if len(param) == 4:
        a, b, c, d = param[0], param[1], param[2], param[3]
        y = func(x, a, b, c, d)
    if not color==None:
        ax.plot(x, y, color=color, linestyle=linestyle, label=label, **kwargs)
    else:
        ax.plot(x, y, label=label, linestyle=linestyle, **kwargs)
    if legend:
        ax.legend()
    return fig, ax

def plot(x, y=None, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False,
         fig=None, ax=None, maskon=False, thd=1, **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    if y is None:
        y = copy.deepcopy(x)
        x = np.arange(len(x))
    # Make sure x and y are np.array
    x, y = np.asarray(x), np.asarray(y)

    if len(x) > len(y):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    elif len(y) > len(x):
        print("Warning : x and y data do not have the same length")
        y = y[:len(x)]

    # remove nans
    keep = ~np.isnan(x) * ~np.isnan(y)
    x, y = x[keep], y[keep]

    if maskon:
        keep = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        keep = [True] * len(x)

    if color is None:
        ax.plot(x[keep], y[keep], label=label, **kwargs)
    else:
        ax.plot(x[keep], y[keep], color=color, label=label, **kwargs)

    if legend:
        ax.legend()
    return fig, ax


def plot_multicolor(x, y=None, colored_by=None, cmap='viridis',
                    fignum=1, figsize=None,
                    subplot=None,
                    fig=None, ax=None, maskon=False, thd=1,
                    linewidth=2, vmin=None, vmax=None, **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed

    org source: https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/multicolored_line.html
    """

    if vmin is None:
        vmin = np.nanmin(colored_by)
    if vmax is None:
        vmax = np.nanmax(colored_by)

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    if y is None:
        y = copy.deepcopy(x)
        # x = np.arange(len(x))
    # Make sure x and y are np.array
    x, y = np.asarray(x), np.asarray(y)

    if colored_by is None:
        print('... colored_by is None. Pass a list/array by which line segments are colored. Using x instead...')
        colored_by = x

    if len(x) > len(y):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    elif len(y) > len(x):
        print("Warning : x and y data do not have the same length")
        y = y[:len(x)]
    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)

    x, y, colored_by = x[mask], y[mask], colored_by[mask]
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)

    # Set the values used for colormapping
    lc.set_array(colored_by)
    lc.set_linewidth(linewidth)
    line = ax.add_collection(lc)

    # autoscale does not work for collection => manually set x/y limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return fig, ax

def plot_with_arrows(x, y=None, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False, fig=None, ax=None, maskon=False, thd=1, **kwargs):
    fig, ax = plot(x, **kwargs)
    lines = ax.get_lines()
    for i, line in enumerate(lines):
        add_arrow_to_line(ax, line)

def plot3d(x, y, z, fignum=1, figsize=None, label='', color=None, subplot=None, fig=None,
           ax=None, labelaxes=True, **kwargs):
    """
    plot a 3D graph using given x, y, z
    """

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize, projection='3d')
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()
    # Make sure x and y are np.array
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    if not len(x)==len(y)==len(z):
        raise ValueError('... x, y, z do not have the same length.')

    # #                     color=color,
    #                     color=color,
    #                    )
    if color is None:
        line, = ax.plot(x, y, z, label=label, **kwargs)
    else:
        line, = ax.plot(x, y, z, color=color, label=label, **kwargs)
    if labelaxes:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

    set_axes_equal(ax)
    return fig, ax



def plot_surface(x, y, z, shade=True, fig=None, ax=None, fignum=1, subplot=None, figsize=None,
                 azdeg=0, altdeg=65):
    """
    plot_surface for the graph module
        ... By default, it enables the shading feature

    Source: https://stackoverflow.com/questions/28232879/phong-shading-for-shiny-python-3d-surface-plots/31754643
    Parameters
    ----------
    x
    y
    z
    shade
    fignum
    subplot
    figsize

    Returns
    -------

    """
    # example
    # x, y = np.mgrid[-3:3:100j,-3:3:100j]
    # z = 3*(1 - x)**2 * np.exp(-x**2 - (y + 1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2 - y**2) - 1./3*np.exp(-(x + 1)**2 - y**2)

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize, projection='3d')
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    # Create light source object.
    ls = mpl.colors.LightSource(azdeg=azdeg, altdeg=altdeg)
    if shade:
        # Shade data, creating an rgb array.
        rgb = ls.shade(z, plt.cm.RdYlBu)
    else:
        rgb = None
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                       antialiased=False, facecolors=rgb)
    return fig, ax, surf

def plot_spline(x, y=None, order=3,
                fignum=1, figsize=None, subplot=None,
                fig=None, ax=None,
                label='', color=None, legend=False,
                maskon=False, thd=1, **kwargs):

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    if y is None:
        y = copy.deepcopy(x)
        x = np.arange(len(x))
    # Make sure x and y are np.array
    x, y = np.asarray(x), np.asarray(y)

    if len(x) > len(y):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    elif len(y) > len(x):
        print("Warning : x and y data do not have the same length")
        y = y[:len(x)]
    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)

    spl_func = interpolate.UnivariateSpline(x[mask], y[mask], k=order)
    x4plot = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    y4plot = spl_func(x4plot)
    if color is None:
        ax.plot(x4plot, y4plot, label=label, **kwargs)
    else:
        ax.plot(x4plot, y4plot, color=color, label=label, **kwargs)

    if legend:
        ax.legend()
    return fig, ax



def plot_saddoughi(fignum=1, fig=None, ax=None, figsize=None, label='Re$_{\lambda} \approx 600 $ \n Saddoughi and Veeravalli, 1994', color='k', alpha=0.6, subplot=None, legend=False, **kwargs):
    """
    plot universal 1d energy spectrum (Saddoughi, 1992)
    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    x = np.asarray([1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    y = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])

    ax.plot(x, y, color=color, label=label, alpha=alpha,**kwargs)
    if legend:
        ax.legend()
    tologlog(ax)
    labelaxes(ax, '$\kappa \eta$', '$E_{11} / (\epsilon\\nu^5)^{1/4}$')
    return fig, ax

def plot_saddoughi_struc_func(fignum=1, fig=None, ax=None, figsize=None,
                              label='Re$_{\lambda} \approx 600 $ \n Saddoughi and Veeravalli, 1994',
                              color='k', alpha=0.6, subplot=None,
                              legend=False,  zorder=0, **kwargs):
    """
    Plots the second order structure function on Saddoughi & Veeravalli, 1994

    Parameters
    ----------
    fignum
    fig
    ax
    figsize
    label
    color
    alpha
    subplot
    legend
    kwargs

    Returns
    -------

    """
    tflow_dir = os.path.split(os.path.realpath(__file__))[0]

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    datapath = tflow_dir + '/reference_data/sv_struc_func.h5'
    # datapath = tflow_dir + '/velocity_ref/sv_struc_func.txt'
    # data = np.loadtxt(datapath, skiprows=1, delimiter=',')
    # r_scaled, dll = data[:, 0], data[:, 1]
    with h5py.File(datapath, 'r') as ff:
        r_scaled = np.asarray(ff['r_s'])
        dll = np.asarray(ff['dll'])
    ax.plot(r_scaled, dll, alpha=alpha, color=color, label=label, **kwargs)
    if legend:
        ax.legend()
    tosemilogx(ax)
    labelaxes(ax, '$r / \eta$', '$D_{LL} / (\epsilon r)^{2/3}$')

def scatter(x, y, ax=None, fig=None,  fignum=1, figsize=None,
            marker='o', fillstyle='full', label=None, subplot=None, legend=False,
            maskon=False, thd=1,
            **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    if figsize is not None:
        fig.set_size_inches(figsize)

    x, y = np.array(x), np.array(y)
    if len(x.flatten()) > len(y.flatten()):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]

    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)

    if fillstyle =='none':
        # Scatter plot with open markers
        facecolors = 'none'
        # ax.scatter(x, y, color=color, label=label, marker=marker, facecolors=facecolors, edgecolors=edgecolors, **kwargs)
        ax.scatter(x[mask], y[mask], label=label, marker=marker, facecolors=facecolors, **kwargs)
    else:
        ax.scatter(x[mask], y[mask], label=label, marker=marker, **kwargs)
    if legend:
        plt.legend()
    return fig, ax

def scatter3d(x, y, z, ax=None, fig=None, fignum=1, figsize=None, marker='o',
            fillstyle='full', label=None, subplot=None, legend=False,
            labelaxes=True, **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize, projection='3d')
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    x, y, z = np.array(x), np.array(y), np.asarray(z)

    if fillstyle =='none':
        # Scatter plot with open markers
        facecolors = 'none'
        # ax.scatter(x, y, color=color, label=label, marker=marker, facecolors=facecolors, edgecolors=edgecolors, **kwargs)
        ax.scatter(x, y, z, label=label, marker=marker, facecolors=facecolors, **kwargs)
    else:
        ax.scatter(x, y, z, label=label, marker=marker, **kwargs)
    if legend:
        plt.legend()

    if labelaxes:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

    set_axes_equal(ax)
    return fig, ax

def pdf(data, nbins=10, return_data=False, vmax=None, vmin=None, fignum=1, figsize=None, subplot=None, density=True, analyze=False, **kwargs):
    def compute_pdf(data, nbins=10):
        # Get a normalized histogram
        # exclude nans from statistics
        hist, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=density)
        # len(bins) = len(hist) + 1
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, hist

    data = np.asarray(data)

    # Use data where values are between vmin and vmax
    if vmax is not None:
        cond1 = np.asarray(data) < vmax # if nan exists in data, the condition always gives False for that data point
    else:
        cond1 = np.ones(data.shape, dtype=bool)
    if vmin is not None:
        cond2 = np.asarray(data) > vmin
    else:
        cond2 = np.ones(data.shape, dtype=bool)
    data = data[cond1 * cond2]

    # compute a pdf
    bins, hist = compute_pdf(data, nbins=nbins)
    fig, ax = plot(bins, hist, fignum=fignum, figsize=figsize, subplot=subplot, **kwargs)

    if analyze:
        bin_width = float(bins[1]-bins[0])
        mean = np.nansum(bins * hist * bin_width)
        mode = bins[np.argmax(hist)]
        var = np.nansum(bins**2 * hist * bin_width)
        text2 = 'mean: %.2f' % mean
        text1 = 'mode: %.2f' % mode
        text3 = 'variance: %.2f' % var
        addtext(ax, text=text2, option='tc2')
        addtext(ax, text=text1, option='tc')
        addtext(ax, text=text3, option='tc3')

    if not return_data:
        return fig, ax
    else:
        return fig, ax, bins, hist


def errorbar(x, y, xerr=0., yerr=0., fignum=1, marker='o', fillstyle='full', linestyle='None', label=None, mfc='white',
             subplot=None, legend=False, figsize=None, maskon=False, thd=1, **kwargs):
    """ errorbar plot

    Parameters
    ----------
    x : array-like
    y : array-like
    xerr: must be a scalar or numpy array with shape (N,1) or (2, N)... [xerr_left, xerr_right]
    yerr:  must be a scalar or numpy array with shape (N,) or (2, N)... [yerr_left, yerr_right]
    fignum
    label
    color
    subplot
    legend
    kwargs

    Returns
    -------
    fig
    ax

    """
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x, y = np.array(x), np.array(y)
    # Make xerr and yerr numpy arrays if they are not scalar. Without this, TypeError would be raised.
    if not (isinstance(xerr, int) or isinstance(xerr, float)):
        xerr = np.array(xerr)
    else:
        xerr = np.ones_like(x) * xerr
    if not (isinstance(yerr, int) or isinstance(yerr, float)):
        yerr = np.array(yerr)
    else:
        yerr = np.ones_like(x) * yerr
    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)
    if fillstyle == 'none':
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, mfc=mfc, linestyle=linestyle,
                    label=label, **kwargs)
    else:
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, fillstyle=fillstyle,
                    linestyle=linestyle, label=label, **kwargs)
    if legend:
        plt.legend()
    return fig, ax

def errorfill(x, y, yerr, fignum=1, color=None, subplot=None, alpha_fill=0.3, ax=None, label=None,
              legend=False, figsize=None, color_cycle=__color_cycle__, maskon=False, thd=1, **kwargs):

    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    x = np.array(x)
    y = np.array(y)

    #ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        yerrdown, yerrup = yerr
        ymin = y - yerrdown
        ymax = y + yerrup
    else:
        ymin = y - yerr
        ymax = y + yerr

    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)


    p = ax.plot(x[mask], y[mask], color=color, label=label, **kwargs)
    color = p[0].get_color()
    ax.fill_between(x[mask], ymax[mask], ymin[mask], color=color, alpha=alpha_fill)

    #patch used for legend
    color_patch = mpatches.Patch(color=color, label=label)
    if legend:
        plt.legend(handles=[color_patch])


    return fig, ax, color_patch


## Plot a fit curve
def plot_fit_curve(xdata, ydata, func=None, fignum=1, subplot=111, ax=None, figsize=None, linestyle='--',
                   xmin=None, xmax=None, add_equation=True, eq_loc='bl', color=None, label='fit',
                   show_r2=False, return_r2=False, p0=None, bounds=(-np.inf, np.inf), maskon=True, thd=1,**kwargs):
    """
    Plots a fit curve given xdata and ydata
    Parameters
    ----------
    xdata
    ydata
    func : Method, assumes a function to be passed
    fignum
    subplot

    Returns
    -------
    fig, ax
    popt, pcov : fit results, covariance matrix
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if len(xdata) != len(ydata):
        print('x and y have different length! Data will be clipped... %d, %d' % (len(xdata), len(ydata)))
        n = min(len(xdata), len(ydata))
        xdata = xdata[:n]
        ydata = ydata[:n]

    if any(np.isnan(ydata)) or any(np.isnan(xdata)):
        print('Original data contains np.nans! Delete them for curve fitting')
        condx, condy = np.isnan(xdata), np.isnan(ydata)
        cond = (~condx * ~condy)
        print('No of deleted data points %d / %d' % (np.sum(~cond), len(xdata)))
        if np.sum(~cond) == len(xdata):
            print('No data points for fitting!')
            raise RuntimeError
        xdata, ydata = xdata[cond], ydata[cond]

    if xmin is None:
        xmin = np.nanmin(xdata)
    if xmax is None:
        xmax = np.nanmax(xdata)

    if maskon:
        mask = get_mask4erroneous_pts(xdata, ydata, thd=thd)
        if any(mask):
            xdata = xdata[mask]
            ydata = ydata[mask]


    x_for_plot = np.linspace(xmin, xmax, 1000)
    if func is None or func=='linear':
        print('Fitting to a linear function...')
        popt, pcov = curve_fit(std_func.linear_func, xdata, ydata, p0=p0, bounds=bounds)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle, ax=ax, **kwargs)
        else:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, ax=ax, **kwargs)

        if add_equation:
            text = '$y=ax+b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.linear_func(xdata, *popt)
    elif func=='power':
        print('Fitting to a power law...')

        popt, pcov = curve_fit(std_func.power_func, xdata, ydata, p0=p0, bounds=bounds)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.power_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle, ax=ax, **kwargs)
        else:
            fig, ax = plot(x_for_plot, std_func.power_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, ax=ax, **kwargs)

        if add_equation:
            text = '$y=ax^b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.power_func(xdata, *popt)
    else:
        popt, pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds)
        if color is None:
            fig, ax = plot(x_for_plot, func(x_for_plot, *popt), fignum=fignum, subplot=subplot, label=label, figsize=figsize,
                           linestyle=linestyle, ax=ax, **kwargs)
        else:
            fig, ax = plot(x_for_plot, func(x_for_plot, *popt), fignum=fignum, subplot=subplot, label=label, figsize=figsize,
                           color=color, linestyle=linestyle, ax=ax, **kwargs)
        y_fit = func(xdata, *popt)
    #plot(x_for_plot, std_func.power_func(x_for_plot, *popt))

    if show_r2 or return_r2:
        # compute R^2
        # residual sum of squares
        ss_res = np.sum((ydata - y_fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        if show_r2:
            addtext(ax, '$R^2: %.2f$' % r2, option='bl3')
        if return_r2:
            return fig, ax, popt, pcov, r2



    return fig, ax, popt, pcov


## 2D plotsFor the plot you showed at group meeting of lambda converging with resolution, can you please make a version with two x axes (one at the top, one below) one pixel spacing, other PIV pixel spacing, and add a special tick on each for the highest resolution point.
# (pcolormesh)
def color_plot(x, y, z, subplot=None, fignum=1, figsize=None, ax=None, vmin=None, vmax=None, log10=False, label=None,
               cbar=True, cmap='magma', symmetric=False, aspect='equal', option='scientific', ntick=5, tickinc=None, **kwargs):
    """  Color plot of 2D array
    Parameters
    ----------
    x 2d array eg. x = np.mgrid[slice(1, 5, dx), slice(1, 5, dy)]
    y 2dd array
    z 2d array
    subplot
    fignum
    vmin
    vmax
    log10
    show
    cbar
    cmap

    Returns
    -------
    fig
    ax
    cc QuadMesh class object

    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
        # fig, ax = set_fig(fignum, subplot, figsize=figsize, aspect=aspect)

    if log10:
        z = np.log10(z)

    # For Diverging colormap, ALWAYS make the color thresholds symmetric
    if cmap in ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']:
        symmetric = True

    if symmetric:
        hide = np.isinf(z)
        keep = ~hide
        if vmin is None and vmax is None:
            v = max(np.abs(np.nanmin(z[keep])), np.abs(np.nanmax(z[keep])))
            vmin, vmax = -v, v
        else:
            arr = np.asarray([vmin, vmax])
            v = np.nanmax(np.abs(arr))
            vmin, vmax = -v, v




    # Note that the cc returned is a matplotlib.collections.QuadMesh
    # print('np.shape(z) = ' + str(np.shape(z)))
    if vmin is None and vmax is None:
        # plt.pcolormesh returns a QuadMesh class object.
        cc = ax.pcolormesh(x, y, z, cmap=cmap, **kwargs)
    else:
        cc = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if cbar:
        if vmin is None and vmax is None:
            add_colorbar(cc, ax=ax, label=label, option=option, ntick=ntick, tickinc=tickinc)
        elif vmin is not None and vmax is None:
            add_colorbar(cc, ax=ax, label=label, option=option, vmin=vmin, ntick=ntick, tickinc=tickinc)
        elif vmin is None and vmax is not None:
            add_colorbar(cc, ax=ax, label=label, option=option, vmax=vmax, ntick=ntick, tickinc=tickinc)
        else:
            add_colorbar(cc, ax=ax, label=label, option=option, vmin=vmin, vmax=vmax, ntick=ntick, tickinc=tickinc)
    ax.set_aspect(aspect)
    # set edge color to face color
    cc.set_edgecolor('face')

    return fig, ax, cc

#imshow
def imshow(griddata, xmin=0, xmax=1, ymin=0, ymax=1, cbar=True, vmin=0, vmax=0, \
           fignum=1, subplot=111, figsize=__figsize__, interpolation='nearest', cmap='bwr', scale=1.0,
           key_kwargs={},
           **kwargs):
    """

    Parameters
    ----------
    griddata
    xmin
    xmax
    ymin
    ymax
    cbar
    vmin
    vmax
    fignum
    subplot
    figsize
    interpolation: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
                   'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
                   'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'.
    cmap

    Returns
    -------

    """
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    if vmin == vmax == 0:
        cax = ax.imshow(griddata, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap)
    else:
        cax = ax.imshow(griddata, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar:
        cc = fig.colorbar(cax, scale=1.0)
    return fig, ax, cax, cc


# quiver
def quiver(x, y, u, v, subplot=None, fignum=1, figsize=None, ax=None,
           inc_x=1, inc_y=1, inc=None, color='k',
           vmin=None, vmax=None,
           key=True, key_loc=[0.8, 1.06], key_length=None,
           key_label=None, key_units='mm/s', key_labelpos='E',
           key_pad=25., key_fmt='.1f',
           key_kwargs={},
           aspect='equal', **kwargs):
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
    if inc is not None:
        inc_x = inc_y = inc
    x_tmp, y_temp = x[::inc_y, ::inc_x], y[::inc_y, ::inc_x]
    u_tmp, v_tmp = u[::inc_y, ::inc_x], v[::inc_y, ::inc_x]
    u_norm = np.sqrt(u_tmp ** 2 + v_tmp ** 2)

    if vmin is None:
        vmin = np.nanmin(u_norm)
    if vmax is None:
        vmax = np.nanmax(u_norm)
    hide1 = u_norm < vmin
    hide2 = u_norm > vmax
    hide3 = np.isinf(u_norm)
    hide = np.logical_or(np.logical_or(hide1, hide2), hide3)
    cond = ~hide

    Q = ax.quiver(x_tmp[cond], y_temp[cond], u_tmp[cond], v_tmp[cond], color=color, **kwargs)

    if key:
        if key_length is None:
            U_rms = np.nanmean(u_norm[cond])
            U_rmedians = np.nanmedian(u_norm[cond])

            # key_length = 10 ** round(np.log10(U_rms))
            # key_length = 10 ** round(np.log10(U_rmedians))
            # key_length = round(U_rmedians, int(-round(np.log10(U_rmedians))) + 1) * 5
            key_length = round(U_rms, int(-round(np.log10(U_rms))) + 1)
        if key_label is None:
            key_label = '{:' + key_fmt + '} '
            key_label = key_label.format(key_length) + key_units
        title(ax, '   ') # dummy title to create space on the canvas
        ax._set_title_offset_trans(key_pad)

        ax.quiverkey(Q, key_loc[0], key_loc[1], key_length, key_label, labelpos=key_labelpos, coordinates='axes',
                     color=color, **key_kwargs)
    ax.set_aspect(aspect)
    return fig, ax, Q

# streamlines
def streamplot(x, y, u, v, subplot=None, fignum=1, figsize=None, ax=None, density=[1., 1.],
               aspect='equal', **kwargs):
    """
    Plots streamlines (2D)

    Parameters
    ----------
    x: 2d array
    y: 2d array
    u: 2d array
    v: 2d array
    subplot: int
    fignum: int
    figsize: tuple
    ax: matplotlib.ax.Axes instance
    density: 1d array-like
        ... density of streamlines
    aspect: str, default: "equal"
        ... options: "equal", "auto"
    kwargs: dict
        ... passed to ax.streamplot()

    Returns
    -------

    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
    ax.streamplot(x, y, u, v, density=density, **kwargs)

    if aspect=='equal':
        ax.set_aspect('equal')
    return fig, ax

def contour(x, y, psi, levels=10,
            vmin=None, vmax=None,
            subplot=None, fignum=1, figsize=None, ax=None,
            clabel=True,
            fontsize=9, inline=True, fmt='%1.3f',
            label_kwargs={},
            **kwargs):
    """
    Plot contours.


    Parameters
    ----------
    x: 2d array
    y: 2d array
    psi: 2d array
    levels: int or 1d array-like
        ... If int (n), it plots n contours.
        ... If array-like, it plots contours at the corresponding levels.
    vmin: int
        ... plots contours at the levels in (vmin, vmax)
    vmax: int
        ... plots contours at the levels in (vmin, vmax)
    subplot: int
    fignum: int
    figsize: tuple
    ax: matplotlib.ax.Axes instance
    fontsize
        ... passed to ax.clabel()
    inline: bool, default: True
        ... passed to ax.clabel()
    fmt: str, default: "%.1.3f"
        ... passed to ax.clabel()
    label_kwargs
        ... passed to ax.clabel()
    kwargs: dict
        ... passed to ax.contour()

    Returns
    -------

    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    if vmin is None:
        vmin = np.nanmin(psi)
    if vmax is None:
        vmax = np.nanmax(psi)
    hide1 = psi < vmin
    hide2 = psi > vmax
    hide = np.logical_or(hide1, hide2)
    psi[hide] = np.nan

    ctrs = ax.contour(x, y, psi, levels, **kwargs)
    if clabel:
        ax.clabel(ctrs, fontsize=fontsize, inline=inline, fmt=fmt, **label_kwargs)

    return fig, ax, ctrs


def get_contours(ctrs, close_ctr=True, thd=0.5, min_length=0,
                 levels=None):
    """
    Returns positions of contours drawn by ax.contour()
    ... each contour has a different length.

    Parameters
    ----------
    ctrs: QuadContourSet instance (output of ax.contour)
    close_ctr: bool
        ... If True, returned points on the contours would be closed.
    thd: float
        ... Relevant parameter if close_ctr is True
        ... Let the beginning and the end of a contour be R1 and R2.
            If |R1 - R2| < thd, it considers the contour closed.
    Returns
    -------
    verts: list
        ... verts[n] stores (x, y) of the n-th contour
        ... xn, yn = verts[n][:, 0], verts[n][:, 1]
    """
    def get_path_length(p):
        """
        Returns arclength of a matplotlib.path.Path instance
        Parameters
        ----------
        p: matplotlib.path.Path instance

        Returns
        -------
        length
        """
        vert = p.vertices
        x, y = vert[:, 0], vert[:, 1]
        xdiff, ydiff = np.diff(x), np.diff(y)
        length = np.nansum(np.sqrt(xdiff ** 2 + ydiff **2))
        return length

    verts = []
    level_list = []
    n_levels = len(ctrs.collections)
    for i in range(n_levels):
        ps = ctrs.collections[i].get_paths()
        n_ctrs = len(ps) # number of contours at the i-th level

        for j in range(n_ctrs):
            vert = ps[j].vertices
            if close_ctr:
                x0, y0 = vert[0, 0], vert[0, 1]
                x1, y1 = vert[-1, 0], vert[-1, 1]
                r = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                if r > thd:
                    vert = np.append(vert, [[x0, y0]], axis=0)
                    ps[j].vertices = vert
            arclength = get_path_length(ps[j])
            if arclength > min_length:
                verts.append(vert)
                if levels is None:
                    level_list.append(i)
                else:
                    level_list.append(levels[i])
    return verts, level_list

## Miscellanies
def show():
    plt.show()


## Lines
def axhline(ax, y, x0=None, x1=None, color='black', linestyle='--', **kwargs):
    """
    Draw a horizontal line at y=y from xmin to xmax
    Parameters
    ----------
    y
    x

    Returns
    -------

    """
    if x0 is not None:
        xmin, xmax = ax.get_xlim()
        xmin_frac, xmax_frac = x0 / float(xmax), x1 / float(xmax)
    else:
        xmin_frac, xmax_frac= 0, 1
    ax.axhline(y, xmin_frac, xmax_frac, color=color, linestyle=linestyle, **kwargs)

def axvline(ax, x, y0=None, y1=None,  color='black', linestyle='--', **kwargs):
    """
    Draw a vertical line at x=x from ymin to ymax
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    if y0 is not None:
        ymin, ymax = ax.get_ylim()
        ymin_frac, ymax_frac = y0 / float(ymax), y1 / float(ymax)
    else:
        ymin_frac, ymax_frac= 0, 1
    ax.axvline(x, ymin_frac, ymax_frac, color=color, linestyle=linestyle, **kwargs)

## Bands
def axhband(ax, y0, y1, x0=None, x1=None, color='C1', alpha=0.2, **kwargs):
    """
        Make a horizontal band between y0 and y1 (highlighting effect)
        Parameters
        ----------
        ax: plt.axes.axes object
        x0: x-coordinate of the left of a band  (x0 < x1). As a default, x0, x1 = ax.get_xlim()
        x1: x-coordinate of the right of a band (x0 < x1)
        y0: y-coordinate of the bottom of a band  (y0 < y1)
        y1: y-coordinate of the top of a band  (y0 < y1)
        color: color of a band
        alpha: alpha of a band
        kwargs: kwargs for ax.fill_between()

        Returns
        -------

        """
    ymin, ymax = ax.get_ylim()
    if x0 is None and x1 is None:
        x0, x1 = ax.get_xlim()
    ax.fill_between(np.linspace(x0, x1, 2), y0, y1, alpha=alpha, color=color, **kwargs)
    ax.set_xlim(x0, x1)
    ax.set_ylim(ymin, ymax)

def axvband(ax, x0, x1, y0=None, y1=None, color='C1', alpha=0.2, **kwargs):
    """
    Make a vertical band between x0 and x1 (highlighting effect)
    Parameters
    ----------
    ax: plt.axes.axes object
    x0: x-coordinate of the left of a band  (x0 < x1)
    x1: x-coordinate of the right of a band (x0 < x1)
    y0: y-coordinate of the bottom of a band  (y0 < y1)
    y1: y-coordinate of the top of a band  (y0 < y1). As a default, y0, y1 = ax.get_ylim()
    color: color of a band
    alpha: alpha of a band
    kwargs: kwargs for ax.fill_between()

    Returns
    -------

    """
    xmin, xmax = ax.get_xlim()
    if y0 is None and y1 is None:
        y0, y1 = ax.get_ylim()
    ax.fill_between(np.linspace(x0, x1, 2), y0, y1, alpha=alpha, color=color, **kwargs)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y0, y1)

# Arrow plots
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def add_arrow_to_line(axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], head_width=15, transform=None, **kwargs):
    if isinstance(line, mlines.Line2D):
        add_arrow_to_line2D(axes, line, arrow_locs=arrow_locs, head_width=head_width, transform=transform, **kwargs)
    else:
        add_arrow_to_line3D(axes, line, arrow_locs=arrow_locs, head_width=head_width, transform=transform, **kwargs)


def add_arrow_to_line2D(
    axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', head_width=15, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes:
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if not isinstance(line, mlines.Line2D):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line.get_xdata(), line.get_ydata()

    arrow_kw = {
        "arrowstyle": arrowstyle,
        "mutation_scale": head_width * line.get_linewidth(),
    }

    color = line.get_color()
    use_multicolor_lines = isinstance(color, np.ndarray)
    if use_multicolor_lines:
        raise NotImplementedError("multicolor lines not supported")
    else:
        arrow_kw['color'] = color

    linewidth = line.get_linewidth()
    if isinstance(linewidth, np.ndarray):
        raise NotImplementedError("multiwidth lines not supported")
    else:
        arrow_kw['linewidth'] = linewidth

    if transform is None:
        transform = axes.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform,
            **arrow_kw)
        axes.add_patch(p)
        arrows.append(p)
    return arrows

def add_arrow_to_line3D(
        axes, line, arrow_locs=[0.2, 0.4, 0.6, 0.8], head_width=15, lw=1, transform=None, **kwargs):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    example:
        # plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        z = np.zeros(11)
        line, = ax.plot(x,y,z, alpha=1, lw=3, color='k')
        add_arrow_to_line3D(ax, line, arrow_locs=np.linspace(0., 1., 5), alpha=0.3)

    Parameters:
    -----------
    axes:
    line: Line2D object as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if not isinstance(line, mpl_toolkits.mplot3d.art3d.Line3D):
        raise ValueError("expected a matplotlib.lines.Line3D object")
    x, y, z = line.get_data_3d()

    length = len(x)
    if length < 2:
        return None
    else:
        arrow_kw = {}

        color = line.get_color()
        use_multicolor_lines = isinstance(color, np.ndarray)
        if use_multicolor_lines:
            raise NotImplementedError("multicolor lines not supported")
        else:
            kwargs['color'] = color

        linewidth = line.get_linewidth()
        if isinstance(linewidth, np.ndarray):
            raise NotImplementedError("multiwidth lines not supported")
        else:
            kwargs['linewidth'] = linewidth

        if transform is None:
            transform = axes.transData

        arrows = []
        for loc in arrow_locs:
            s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2))
            n = np.searchsorted(s, s[-1] * loc)
            arrow_tail = (x[n], y[n], z[n])
            arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]), np.mean(z[n:n + 2]))
            arrow_lines = list(zip(arrow_tail, arrow_head))

            arrow = Arrow3D(
                arrow_lines[0], arrow_lines[1], arrow_lines[2],
                mutation_scale=head_width,
                lw=lw, **kwargs)
            ax.add_artist(arrow)
            arrows.append(arrow)
        return arrows

def arrow3D(x, y, z, dx, dy, dz, lw=3, arrowstyle='-|>', color='r', mutation_scale=20,
            ax=None, fig=None, fignum=1, subplot=111, figsize=None, **kwargs):
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    arrow_obj = Arrow3D([x, x+dx], [y, y+dy],
                        [z, z+dz], mutation_scale=mutation_scale,
                        lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
    ax.add_artist(arrow_obj)
    return fig, ax, arrow_obj

def arrow(x, y, dx, dy,
          ax=None, fig=None, fignum=1, subplot=111, figsize=None, **kwargs):
    """
    Adds an arrow on a canvas
    ... Specify an arrow by its starting point (x, y) and its direction (dx, dy)

    Parameters
    ----------
    x
    y
    dx
    dy
    ax
    fig
    fignum
    subplot
    figsize
    kwargs

    Returns
    -------

    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    ax.arrow(x, y, dx, dy, **kwargs)
    return fig, ax

## Legend
# Legend
def legend(ax, remove=False, **kwargs):
    """
    loc:
    best	0, upper right	1, upper left	2, lower left	3, lower right	4, right	5,
    center left	6, center right	7, lower center	8, upper center	9, center	10
    Parameters
    ----------
    ax
    kwargs

    Returns
    -------

    """
    leg = ax.legend(**kwargs)
    if remove:
        leg.get_frame().set_facecolor('none')


# Colorbar
class FormatScalarFormatter(mpl.ticker.ScalarFormatter):
    """
    Ad-hoc class to subclass matplotlib.ticker.ScalarFormatter
    in order to alter the number of visible digits on color bars
    """
    def __init__(self, fformat="%03.1f", offset=True, mathText=True):
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
        self.set_scientific(True)
        # Scientific notation is used for data < 10^-n or data >= 10^m, where n and m are the power limits set using set_powerlimits((n,m))
        self.set_powerlimits((0, 0))
    def _set_format(self):
        """
        Call this method to change the format of tick labels


        Returns
        -------

        """

        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % mpl.ticker._mathdefault(self.format)
    def _update_format(self, fformat):
        self.fformat = fformat
        self._set_format()

def reset_sfmt(fformat="%03.1f"):
    global sfmt
    sfmt = FormatScalarFormatter() # Default format: "%04.1f"
    # sfmt.fformat = fformat # update format
    # sfmt._set_format() # this updates format for scientific nota
    sfmt._update_format(fformat)

reset_sfmt()

def get_sfmt():
    ""
    global sfmt
    reset_sfmt()
    return sfmt

def add_colorbar_old(mappable, fig=None, ax=None, fignum=None, label=None, fontsize=__fontsize__,
                 vmin=None, vmax=None, cmap='jet', option='normal', **kwargs):
    """
    Adds a color bar (Depricated. replaced by add_colorbar)
    Parameters
    ----------
    mappable : image like QuadMesh object to which the color bar applies (NOT a plt.figure instance)
    ax : Parent axes from which space for a new colorbar axes will be stolen
    label :

    Returns
    -------
    """
    # Get a Figure instance
    if fig is None:
        fig = plt.gcf()
        if fignum is not None:
            fig = plt.figure(num=fignum)
    if ax is None:
        ax = plt.gca()

    # if vmin is not None and vmax is not None:
    #     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # elif vmin is None and vmax is not None:
    #     print 'vmin was not provided!'
    # elif vmin is not None and vmax is None:
    #     print 'vmax was not provided!'

    # fig.colorbar makes a another ax object which colives with ax in the fig instance.
    # Therefore, cb has all attributes that ax object has!

    if option == 'scientific':
        cb = fig.colorbar(mappable, ax=ax, cmap=cmap, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(mappable, ax=ax, cmap=cmap, **kwargs)

    if not label == None:
        cb.set_label(label, fontsize=fontsize)

    return cb


def add_colorbar(mappable, fig=None, ax=None, fignum=None, location='right', label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, aspect='equal', ntick=5, tickinc=None, **kwargs):
    """
    Adds a color bar

    e.g.
        fig = plt.figure()
        img = fig.add_subplot(111)
        ax = img.imshow(im_data)
        colorbar(ax)
    Parameters
    ----------
    mappable
    location

    Returns
    -------

    """
    global sfmt
    def get_ticks_for_sfmt(mappable, n=10, inc=0.5, **kwargs):
        """
        Returns ticks for scientific notation
        ... setting format=smft sometimes fails to use scientific notation for colorbar.
        ... This function should ensure the colorbar object to have appropriate ticks
         to display numbers in scientific fmt once the generated ticks are passed to fig.colorbar().
        Parameters
        ----------
        mappable
        n: int, (ROUGHLY) the number of ticks
        inc: float (0, 0.5]
        ... 0.5 or 0.25 is recommended
        Returns
        -------
        ticks, list, ticks for scientific format
        """
        # ticks for scientific notation
        zmin, zmax = np.nanmin(mappable.get_array()), np.nanmax(mappable.get_array())
        if 'vmin' in kwargs.keys():
            zmin = kwargs['vmin']
        if 'vmax' in kwargs.keys():
            zmax = kwargs['vmax']

        # ticks = np.linspace(zmin, zmax, 2*n)
        exponent = int(np.floor(np.log10(np.abs(zmax))))
        # ticks = np.around(ticks[1::2], decimals=-exponent + 1)
        if tickinc is not None:
            # Specify the increment of ticks!
            dz = inc * 10 ** exponent
            ticks = [i * dz for i in range(int(zmin / dz), int(zmax / dz)+1)]
        else:
            # Specify the number of ticks!
            exp = int(np.floor(np.log10((zmax - zmin) / n)))
            dz = np.round((zmax - zmin) / n, -exp)
            # exp = int(np.ceil(np.log10((zmax - zmin) / n)))
            # dz = (zmax - zmin) / n
            ticks = [i * dz for i in range(int(zmin / dz), int(zmax / dz) + 1)]
            # print(np.log10((zmax - zmin) / n), exp)
            # print((zmax - zmin) / n, dz)
            # print(ticks)

        return ticks

    def remove_vmin_vmax_from_kwargs(**kwargs):
        if 'vmin' in kwargs.keys():
            del kwargs['vmin']
        if 'vmax' in kwargs.keys():
            del kwargs['vmax']
        return kwargs

    # ax = mappable.axes
    # fig = ax.figure
    # Get a Figure instance
    if fig is None:
        fig = plt.gcf()
        if fignum is not None:
            fig = plt.figure(num=fignum)
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()

    reset_sfmt()

    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.15)
    if option == 'scientific_custom':
        ticks = get_ticks_for_sfmt(mappable, n=ntick, inc=tickinc, **kwargs)
        kwargs = remove_vmin_vmax_from_kwargs(**kwargs)
        # sfmt.format = '$\mathdefault{%1.1f}$'
        cb = fig.colorbar(mappable, cax=cax, format=sfmt, ticks=ticks, **kwargs)
        # cb = fig.colorbar(mappable, cax=cax, format=sfmt, **kwargs)
    elif option == 'scientific':
        # old but more robust
        kwargs = remove_vmin_vmax_from_kwargs(**kwargs)
        cb = fig.colorbar(mappable, cax=cax, format=sfmt, **kwargs)
    else:
        kwargs = remove_vmin_vmax_from_kwargs(**kwargs)
        cb = fig.colorbar(mappable, cax=cax, **kwargs)

    if not label is None:
        if fontsize is None:
            cb.set_label(label)
        else:
            cb.set_label(label, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()


    return cb


def add_discrete_colorbar(ax, colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, **kwargs):
    fig = ax.get_figure()
    if vmax is None:
        vmax = len(colors)
    tick_spacing = (vmax - vmin) / float(len(colors))
    ticks = np.linspace(vmin, vmax, len(colors)+1) + tick_spacing / 2. # tick positions

    # if there are too many ticks, just use 3 ticks
    if len(ticks) > 10:
        n = len(ticks)
        ticks = [ticks[0], ticks[n/2], ticks[-2]]
        if ticklabel is not None:
            ticklabel = [ticklabel[0], ticklabel[n/2], ticklabel[-1]]


    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    if option == 'scientific':
        cb = fig.colorbar(sm, ticks=ticks, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(sm, ticks=ticks, **kwargs)

    if ticklabel is not None:
        cb.ax.set_yticklabels(ticklabel)

    if not label is None:
        if fontsize is None:
            cb.set_label(label)
        else:
            cb.set_label(label, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb



def add_colorbar_alone(ax, values, cmap=cmap, label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, location='right', color='k', **kwargs):
    fig = ax.get_figure()

    # number of values
    n = np.asarray(values).size
    # get min/max values
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    # vmin, vmax = 0, len(values)


    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    # make an axis instance for a colorbar
    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.15)

    if option == 'scientific':
        cb = fig.colorbar(sm, cax=cax, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(sm, cax=cax,  **kwargs)

    if ticklabel is not None:
        cb.ax.set_yticklabels(ticklabel)

    if label is not None:
        if fontsize is None:
            cb.set_label(label, color=color)
        else:
            cb.set_label(label, fontsize=fontsize, color=color)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect == 'equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb

# def add_colorbar_alone(fig=None, ax=None, ax_loc=[0.05, 0.80, 0.9, 0.15], vmin=0, vmax=1, cmap=cmap, orientation='horizontal',
#                        label=None, fontsize=__fontsize__, *kwargs):
#     """
#     Add a colorbar alone to a canvas.
#     Use a specified figure and axis object if given. Otherwise, create one at location "ax_loc"
#     Parameters
#     ----------
#     fig
#     ax
#     ax_loc
#     vmin
#     vmax
#     cmap
#     orientation
#     label
#
#     Returns
#     -------
#     ax: axis object
#     cb: colorbarbase object
#
#     """
#
#
#     if fig is None:
#         fig = plt.gcf()
#     if ax is None:
#         ax = fig.add_axes(ax_loc)
#     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#     cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                     norm=norm,
#                                     orientation=orientation)
#     if label is not None:
#         cb.set_label(label, fontsize=fontsize)
#     return ax, cb





def colorbar(fignum=None, label=None, fontsize=__fontsize__):
    """
    Use is DEPRECIATED. This method is replaced by add_colorbar(mappable)
    I keep this method for old codes which might have used this method
    Parameters
    ----------
    fignum :
    label :

    Returns
    -------
    """
    fig, ax = set_fig(fignum)
    c = plt.colorbar()
    if not label==None:
        c.set_label(label, fontsize=fontsize)
    return c


### Axes
# Label
def labelaxes(ax, xlabel, ylabel, **kwargs):
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)
# multi-color labels
def labelaxes_multicolor(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kwargs):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis == 'x' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', **kwargs))
                 for text, color in zip(list_of_strings, list_of_colors)]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad, frameon=False, bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == 'y' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', rotation=90, **kwargs))
                 for text, color in zip(list_of_strings[::-1], list_of_colors)]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.2, 0.4),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)



# Limits
def setaxes(ax, xmin, xmax, ymin, ymax):
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return ax

## Set axes to semilog or loglog
def tosemilogx(ax=None, **kwargs):
    if ax == None:
        ax = plt.gca()
    ax.set_xscale("log", **kwargs)
def tosemilogy(ax=None, **kwargs):
    if ax == None:
        ax = plt.gca()
    ax.set_yscale("log", **kwargs)
def tologlog(ax=None, **kwargs):
    if ax == None:
        ax = plt.gca()
    ax.set_xscale("log", **kwargs)
    ax.set_yscale("log", **kwargs)

# Ticks
def set_xtick_interval(ax, tickint):
    """
    Sets x-tick interval as tickint
    Parameters
    ----------
    ax: Axes object
    tickint: float, tick interval

    Returns
    -------

    """
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tickint))

def set_ytick_interval(ax, tickint):
    """
    Sets y-tick interval as tickint
    Parameters
    ----------
    ax: Axes object
    tickint: float, tick interval

    Returns
    -------

    """
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tickint))



##Title
def title(ax, title, **kwargs):
    """
    ax.set_title(title, **kwargs)
    ... if you want more space for the tile, try "pad=50"

    Parameters
    ----------
    ax
    title
    subplot
    kwargs

    Returns
    -------

    """
    ax.set_title(title, **kwargs)

def suptitle(title, fignum=None,
             tight_layout=True,
             rect=[0, 0.03, 1, 0.95],
             **kwargs):
    """
    Add a centered title to the figure.
    If fignum is given, it adds a title, then it reselects the figure which selected before this method was called.
    ... this is because figure class does not have a suptitle method.
    ...
    Parameters
    ----------
    title
    fignum
    kwargs

    Returns
    -------

    """
    if fignum is not None:
        plt.figure(fignum)
    fig = plt.gcf()

    plt.suptitle(title, **kwargs)
    if tight_layout:
        fig.tight_layout(rect=rect)




##Text
def set_standard_pos(ax):
    """
    Sets standard positions for added texts in the plot
    left: 0.025, right: 0.75
    bottom: 0.10 top: 0.90
    xcenter: 0.5 ycenter:0.5
    Parameters
    ----------
    ax

    Returns
    -------
    top, bottom, right, left, xcenter, ycenter: float, position

    """
    left_margin, right_margin, bottom_margin, top_margin = 0.025, 0.75, 0.1, 0.90

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    width, height = np.abs(xright - xleft), np.abs(ytop - ybottom)

    if ax.get_xscale() == 'linear':
        left, right = xleft + left_margin * width, xleft + right_margin * width
        xcenter = xleft + width/2.
    if ax.get_yscale() == 'linear':
        bottom, top = ybottom + bottom_margin * height, ybottom + top_margin * height
        ycenter = ybottom + height / 2.

    if ax.get_xscale() == 'log':
        left, right = xleft + np.log10(left_margin * width), xleft + np.log10(right_margin * width)
        xcenter = xleft + np.log10(width/2.)

    if ax.get_yscale() == 'log':
        bottom, top = ybottom + np.log10(bottom_margin * height), ybottom + np.log10(top_margin * height)
        ycenter = ybottom + np.log10(height / 2.)

    return top, bottom, right, left, xcenter, ycenter, height, width


def addtext(ax, text='text goes here', x=0, y=0, color='k',
            option=None, npartition=15, **kwargs):
    """
    Adds text to a plot. You can specify the position where the texts will appear by 'option'
    | tl2    tc2    tr2 |
    | tl     tc     tr  |
    | tl3    tc3    tr3 |
    |                   |
    | cl2               |
    | cl     cc     cr  |
    | cl3               |
    |                   |
    | bl2           br2 |
    | bl      bc    br  |
    | bl3           br3 |

    Parameters
    ----------
    ax
    subplot
    text
    x
    y
    fontsize
    color
    option: default locations
    kwargs

    Returns
    ax : with a text
    -------

    """
    top, bottom, right, left, xcenter, ycenter, height, width = set_standard_pos(ax)
    dx, dy = width / npartition,  height / npartition




    if option == None:
        ax.text(x, y, text, color=color, **kwargs)
    if option == 'tr':
        ax.text(right, top, text, color=color, **kwargs)
    if option == 'tr2':
        ax.text(right, top + dy, text,  color=color, **kwargs)
    if option == 'tr3':
        ax.text(right, top - dy, text,  color=color, **kwargs)
    if option == 'tl':
        ax.text(left, top, text,  color=color, **kwargs)
    if option == 'tl2':
        ax.text(left, top + dy, text,  color=color, **kwargs)
    if option == 'tl3':
        ax.text(left, top - dy, text,  color=color, **kwargs)

    if option == 'tc':
        ax.text(xcenter, top, text,  color=color, **kwargs)
    if option == 'tc2':
        ax.text(xcenter, top + dy, text,  color=color, **kwargs)
    if option == 'tc3':
        ax.text(xcenter, top - dy, text,  color=color, **kwargs)
    if option == 'br':
        ax.text(right, bottom, text,  color=color, **kwargs)
    if option == 'br2':
        ax.text(right, bottom + dy, text,  color=color, **kwargs)
    if option == 'br3':
        ax.text(right, bottom - dy, text, color=color, **kwargs)
    if option == 'bl':
        ax.text(left, bottom, text, color=color, **kwargs)
    if option == 'bl2':
        ax.text(left, bottom + dy, text,  color=color, **kwargs)
    if option == 'bl3':
        ax.text(left, bottom - dy, text, color=color, **kwargs)
    if option == 'bc':
        ax.text(xcenter, bottom, text, color=color, **kwargs)
    if option == 'bc2':
        ax.text(xcenter, bottom + dy, text, color=color, **kwargs)
    if option == 'bc3':
        ax.text(xcenter, bottom - dy, text, color=color, **kwargs)
    if option == 'cr':
        ax.text(right, ycenter, text, color=color, **kwargs)
    if option == 'cl':
        ax.text(left, ycenter, text, color=color, **kwargs)
    if option == 'cl2':
        ax.text(left, ycenter + dy, text, color=color, **kwargs)
    if option == 'cl3':
        ax.text(left, ycenter - dy, text, color=color, **kwargs)
    if option == 'cc':
        ax.text(xcenter, ycenter, text,  color=color, **kwargs)
    return ax

def draw_power_triangle(ax, x, y, exponent, w=None, h=None, facecolor='none', edgecolor='r', alpha=1.0, flip=False,
                        fontsize=__fontsize__, set_base_label_one=False, beta=20, zorder=100, **kwargs):
    """
    Draws a triangle which indicates a power law in the log-log plot.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        ... get it like plt.gca()
    x: float / int
        ... x coordinate of the triangle drawn on the plot
    y: float / int
        ... x coordinate of the triangle drawn on the plot
    exponent: float / int
        ... exponent of the power law
        ... Y = X^exponent
    w: float / int
        ... number of decades for the drawn triangle to span on the plot
        ... By default, this function draws a triangle with size of 0.4 times the width of the plot
    h: float / int
        ... number of decades for the drawn triangle to span on the plot
    facecolor: str
        ... face color of the drawn triangle, default: 'none' (transparent)
        ... passed to mpatches.PathPatch object
    edgecolor: str
        ... edge color of the drawn triangle, default: 'r'
        ... passed to mpatches.PathPatch object
    alpha: float [0, 1]
        ... alpha value of the drawn triangle
    flip: bool
        ... If True, it will flip the triangle horizontally.
    fontsize: float / int
        ... fontsize of the texts to indicate the exponent aside the triangle
    set_base_label_one: bool, default: False
        ... If True, it will always annotate the base as '1' and alter the text for the height accordingly.
        ... By default, it will annotate the base and the height using the closest integer pair.
    beta: float / int, default: 20
        ... This is used to control the spacing between the text and the drawn triangle
        ... The higher beta is, the less spacing between the text and the triangle
    zorder: zorder of triangle, default: 0
    kwargs: the other kwargs will be passed to ax.text()

    Returns
    -------

    """
    def simplest_fraction_in_interval(x, y):
        """Return the fraction with the lowest denominator in [x,y]."""
        if x == y:
            # The algorithm will not terminate if x and y are equal.
            raise ValueError("Equal arguments.")
        elif x < 0 and y < 0:
            # Handle negative arguments by solving positive case and negating.
            return -simplest_fraction_in_interval(-y, -x)
        elif x <= 0 or y <= 0:
            # One argument is 0, or arguments are on opposite sides of 0, so
            # the simplest fraction in interval is 0 exactly.
            return Fraction(0)
        else:
            # Remainder and Coefficient of continued fractions for x and y.
            xr, xc = modf(1 / x);
            yr, yc = modf(1 / y);
            if xc < yc:
                return Fraction(1, int(xc) + 1)
            elif yc < xc:
                return Fraction(1, int(yc) + 1)
            else:
                return 1 / (int(xc) + simplest_fraction_in_interval(xr, yr))

    def approximate_fraction(x, e):
        """Return the fraction with the lowest denominator that differs
        from x by no more than e."""
        return simplest_fraction_in_interval(x - e, x + e)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    exp_xmax, exp_xmin = np.log10(xmax), np.log10(xmin)
    exp_ymax, exp_ymin = np.log10(ymax), np.log10(ymin)
    exp_x, exp_y = np.log10(x), np.log10(y)

    # Default size of the triangle is 0.4 times the width of the plot
    if w is None and h is None:
        exp_w = (exp_xmax - exp_xmin) * 0.4
        exp_h = exp_w * exponent
    elif w is None and h is not None:
        exp_h = h
        exp_w = exp_h / exponent
    elif w is not None and h is None:
        exp_w = w
        exp_h = exp_w * exponent
    else:
        exp_w = w
        exp_h = h

    w = 10 ** (exp_x + exp_w) - 10 ** exp_x  # base of the triangle
    h = 10 ** (exp_y + exp_h) - 10 ** exp_y  # height of the triangle
    if not flip:
        path = mpl.path.Path([[x, y], [x + w, y], [x + w, y + h], [x, y]])
    else:
        path = mpl.path.Path([[x, y], [x, y + h], [x + w, y + h], [x, y]])
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)


    # annotate
    # beta = 20. # greater beta corresponds to less spacing between the texts and the triangle edges
    if exponent >= 0 and not flip:
        x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y - (exp_ymax - exp_ymin) / beta)
        x_height, y_height = 10 ** (exp_w + exp_x + 0.4*(exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
    elif exponent < 0 and not flip:
        x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y + 0.3*(exp_ymax - exp_ymin) / beta)
        x_height, y_height = 10 ** (exp_w + exp_x + 0.4*(exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
    elif exponent >= 0 and flip:
        x_base, y_base = 10 ** (exp_x + exp_w * 0.4), 10 ** (exp_y + exp_h + 0.3*(exp_ymax - exp_ymin) / beta)
        x_height, y_height = 10 ** (exp_x - (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
    else:
        x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y + exp_h - (exp_ymax - exp_ymin) / beta)
        x_height, y_height = 10 ** (exp_x - (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.6)


    if set_base_label_one:
        ax.text(x_base, y_base, '1', fontsize=fontsize)
        ax.text(x_height, y_height, '%.2f' % exponent, fontsize=fontsize)
    else:
        # get the numbers to put on the graph to indicate the power
        exponent_rational = approximate_fraction(exponent, 0.0001)
        ax.text(x_base, y_base, str(np.abs(exponent_rational.denominator)), fontsize=fontsize, **kwargs)
        ax.text(x_height, y_height, str(np.abs(exponent_rational.numerator)), fontsize=fontsize, **kwargs)



##Clear plot
def clf(fignum=None):
    plt.figure(fignum)
    plt.clf()
def close(*argv, **kwargs):
    plt.close(*argv, **kwargs)

## Color cycle
def skipcolor(numskip, color_cycle=__color_cycle__):
    """ Skips numskip times in the color_cycle iterator
        Can be used to reset the color_cycle"""
    for i in range(numskip):
        next(color_cycle)
def countcolorcycle(color_cycle = __color_cycle__):
    return sum(1 for color in color_cycle)

def get_default_color_cycle():
    return __color_cycle__

def get_first_n_colors_from_color_cycle(n):
    color_list = []
    for i in range(n):
        color_list.append(next(__color_cycle__))
    return color_list


def create_cmap_using_values(colors=None, color1='greenyellow', color2='darkgreen', n=100):
    """
    Create a colormap instance from a list
    ... same as mpl.colors.LinearSegmentedColormap.from_list()
    Parameters
    ----------
    colors
    color1
    color2
    n

    Returns
    -------

    """
    if colors is None:
        colors = get_color_list_gradient(color1=color1, color2=color2, n=n)
    cmap_name = 'new_cmap'
    newcmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n)

    return newcmap

def get_colors_and_cmap_using_values(values, cmap=None, color1='greenyellow', color2='darkgreen',
                                     vmin=None, vmax=None, n=100):
    """
    Returns colors (list), cmap instance, mpl.colors.Normalize instance
    Parameters
    ----------
    values
    cmap
    color1
    color2
    vmin
    vmax
    n

    Returns
    -------

    """
    values = np.asarray(values)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmin(values)
    if cmap is None:
        cmap = create_cmap_using_values(color1=color1, color2=color2, n=n)
    else:
        cmap = plt.get_cmap(cmap, n)
        # normalize
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(values))
    return colors, cmap, norm

def get_color_list_gradient(color1='greenyellow', color2='darkgreen', color3='darkgreen', n=10):
    """
    Returns a list of colors in RGB between color1 and color2
    Input (color1 and color2) can be RGB or color names set by matplotlib
    Parameters
    ----------
    color1
    color2
    n: length of the returning list

    Returns
    -------
    color_list
    """
    if color3 is None:

        # convert color names to rgb if rgb is not given as arguments
        if not color1[0] == '#':
            color1 = cname2hex(color1)
        if not color2[0] == '#':
            color2 = cname2hex(color2)
        color1_rgb = hex2rgb(color1) / 255.  # np array
        color2_rgb = hex2rgb(color2) / 255.  # np array

        r = np.linspace(color1_rgb[0], color2_rgb[0], n)
        g = np.linspace(color1_rgb[1], color2_rgb[1], n)
        b = np.linspace(color1_rgb[2], color2_rgb[2], n)
        color_list = list(zip(r, g, b))
    else:
        # convert color names to rgb if rgb is not given as arguments
        if not color1[0] == '#':
            color1 = cname2hex(color1)
        if not color2[0] == '#':
            color2 = cname2hex(color2)
        if not color3[0] == '#':
            color3 = cname2hex(color3)
        color1_rgb = hex2rgb(color1) / 255.  # np array
        color2_rgb = hex2rgb(color2) / 255.  # np array
        color3_rgb = hex2rgb(color3) / 255.  # np array

        n_middle = int((n-1)/2)

        r1 = np.linspace(color1_rgb[0], color2_rgb[0], n_middle, endpoint=False)
        g1 = np.linspace(color1_rgb[1], color2_rgb[1], n_middle, endpoint=False)
        b1 = np.linspace(color1_rgb[2], color2_rgb[2], n_middle, endpoint=False)
        color_list1 = list(zip(r1, g1, b1))

        r2 = np.linspace(color2_rgb[0], color3_rgb[0], n-n_middle)
        g2 = np.linspace(color2_rgb[1], color3_rgb[1], n-n_middle)
        b2 = np.linspace(color2_rgb[2], color3_rgb[2], n-n_middle)
        color_list2 = list(zip(r2, g2, b2))
        color_list = color_list1 + color_list2
    return color_list

def get_color_from_cmap(cmap='viridis', n=10, lut=None):
    """
    A simple function which returns a list of RGBA values from a cmap (evenly spaced)
    ... If one desires to assign a color based on values, use get_colors_and_cmap_using_values()
    ... If one prefers to get colors between two colors of choice, use get_color_list_gradient()
    Parameters
    ----------
    cmapname: str, standard cmap name
    n: int, number of colors
    lut, int,
        ... If lut is not None it must be an integer giving the number of entries desired in the lookup table,
        and name must be a standard mpl colormap name.

    Returns
    -------
    colors

    """
    cmap = mpl.cm.get_cmap(cmap, lut)
    colors = cmap(np.linspace(0, 1, n, endpoint=True))
    return colors

def hex2rgb(hex):
    """
    Converts a HEX code to RGB in a numpy array
    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    h = hex.lstrip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))
    return rgb

def cname2hex(cname):
    """
    Converts a color registered on matplotlib to a HEX code
    Parameters
    ----------
    cname

    Returns
    -------

    """
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS) # dictionary. key: names, values: hex codes
    try:
        hex = colors[cname]
        return hex
    except NameError:
        print(cname, ' is not registered as default colors by matplotlib!')
        return None

def set_color_cycle(ax, colors=__def_colors__):
    """
    Sets a color cycle using a list
    Parameters
    ----------
    ax
    colors: list of colors in rgb/cnames/hex codes

    Returns
    -------

    """
    ax.set_prop_cycle(color=colors)

def set_color_cycle_gradient(ax, color1='greenyellow', color2='navy', n=10):
    colors = get_color_list_gradient(color1, color2, n=n)
    ax.set_prop_cycle(color=colors)



# Figure settings
def update_figure_params(params):
    """
    update a default matplotlib setting
    e.g. params = { 'legend.fontsize': 'x-large',
                    'figure.figsize': (15, 5),
                    'axes.labelsize': 'x-large',
                    'axes.titlesize':'x-large',
                    'xtick.labelsize':'x-large',
                    'ytick.labelsize':'x-large'}
    Parameters
    ----------
    params: dictionary

    Returns
    -------

    """
    pylab.rcParams.update(params)

def reset_figure_params():
    pylab.rcParams.update(params)

def default_figure_params():
    mpl.rcParams.update(mpl.rcParamsDefault)

# Use the settings above as a default
reset_figure_params()

## 3D plotting
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



# plotting styles
def show_plot_styles():
    """Prints available plotting styles"""
    style_list = ['default'] + sorted(style for style in plt.style.available)
    print(style_list)
    return style_list
def use_plot_style(stylename):
    """Reminder for me how to set a plotting style"""
    plt.style.use(stylename)

#
def get_markers():
    """Returns a list of available markers for ax.scatter()"""
    filled_markers = list(Line2D.filled_markers)
    unfilled_markers = [m for m, func in Line2D.markers.items()
               if func != 'nothing' and m not in Line2D.filled_markers]
    markers = filled_markers + unfilled_markers
    return markers

# Embedded plots
def add_subplot_axes(ax, rect, axisbg='w', alpha=1):
    """
    Creates a sub-subplot inside the subplot (ax)
    rect: list, [x, y, width, height] e.g. rect = [0.2,0.2,0.7,0.7]
    Parameters
    ----------
    ax
    rect: list, [x, y, width, height]  e.g. rect = [0.2,0.2,0.7,0.7]
    axisbg: background color of the newly created axes object

    Returns
    -------
    subax, Axes class object
    """

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height])
    subax.set_facecolor(axisbg)
    subax.patch.set_alpha(alpha)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.3
    y_labelsize *= rect[3]**0.3
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


# sketches
def draw_circle(ax, x, y, r, linewidth=1, edgecolor='r', facecolor='none', fill=False, **kwargs):
    """
    Draws a circle in a figure (ax)
    Parameters
    ----------
    ax
    x
    y
    r
    linewidth
    edgecolor
    facecolor
    fill

    Returns
    -------

    """
    circle = plt.Circle((x, y), r, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, fill=fill, **kwargs)
    ax.add_artist(circle)
    return circle

def draw_rectangle(ax, x, y, width, height, angle=0.0, linewidth=1, edgecolor='r', facecolor='none', **kwargs):
    """
    Draws a rectangle in a figure (ax)
    Parameters
    ----------
    ax
    x
    y
    width
    height
    angle
    linewidth
    edgecolor
    facecolor
    kwargs

    Returns
    -------

    """
    rect = mpatches.Rectangle((x, y), width, height, angle=angle, linewidth=linewidth, edgecolor=edgecolor,
                              facecolor=facecolor, **kwargs)
    ax.add_patch(rect)
    ax.axis('equal') # this ensures to show the rectangle if the rectangle is bigger than the original size
    return rect


def draw_box(ax, xx, yy, w_box=325., h_box=325., xoffset=0, yoffset=0, linewidth=5,
             scalebar=True, sb_length=50., sb_units='$mm$', sb_loc=(0.95, 0.1), sb_txtloc=(0.0, 0.4), sb_lw=10, sb_txtcolor='white',
             facecolor='k', fluidcolor='skyblue'):
    """
    Draws a box and fills the surrounding area with color (default: skyblue)
    Adds a scalebar by default
    ... drawn box center coincides with the center of given grids(xx, yy)
    ... in order to shift the center of the box, use xoffset any yoffset
    Parameters
    ----------
    ax: matplotlib.axes.Axes instance
    xx: 2d numpy array
        x coordinates
    yy: 2d numpy array
        y coordinates
    w_box: float/int
        width of the box
    h_box: float/int
        height of the box
    xoffset: float/int
        real number to shift the box center in the x direction
    yoffset:
        real number to shift the box center in the x direction
    linewidth: int
        linewidth of drawn box
    scalebar: bool (default: True)
        ... draws a scalebar inside the drawn box
    sb_length: int
        ... length of the scale bar in physical units.
        ...... In principle, this can be float. If you want that, edit the code where ax.text() is called.
        ...... Generalizing to accept the float requires a format which could vary everytime, so just accept integer.
    sb_units: str
        ... units of the sb_length. Default: '$mm$'
    sb_loc: tuple, (x, y)
        ... location of the scale bar. Range: [0, 1]
        ... the units are with respect the width and height of the box
    sb_txtloc: tuple, (x, y)
        ... location of the TEXT of the scale bar. Range: [0, 1]
        ... x=0: LEFT of the scale bar, x=1: RIGHT of the scale bar
        ... y=0: LEFT of the scale bar, x=1: RIGHT of the scale bar

    sb_lw: float
        ... line width of the scale bar

    facecolor
    fluidcolor

    Returns
    -------

    """
    xmin, xmax = np.nanmin(xx), np.nanmax(xx)
    ymin, ymax = np.nanmin(yy), np.nanmax(yy)
    # if np.nanmean(yy) > 0:
    #     xc, yc = xmin + (xmax - xmin) / 2., ymin + (ymax - ymin) / 2.
    # else:
    #     xc, yc = xmin + (xmax - xmin) / 2., ymin - (ymax - ymin) / 2.
    xc, yc = xmin + (xmax - xmin) / 2., ymin + (ymax - ymin) / 2.
    x0, y0 = xc - w_box / 2. + xoffset, yc - h_box / 2. + yoffset
    draw_rectangle(ax, x0, y0, w_box, h_box, linewidth=linewidth, facecolor=facecolor, zorder=0)
    ax.set_facecolor(fluidcolor)

    if scalebar:
        dx, dy = np.abs(xx[0, 1] - xx[0, 0]), np.abs(yy[1, 0] - yy[0, 0]) # mm/px

        #         x0_sb, y0_sb = x0 + 0.8 * w_box, y0 + 0.1*h_box
        x1_sb, y1_sb = x0 + sb_loc[0] * w_box, y0 + sb_loc[1] * h_box
        x0_sb, y0_sb = x1_sb - sb_length, y1_sb
        if sb_loc[1] < 0.5:
            x_sb_txt, y_sb_txt = x0_sb + sb_txtloc[0] * sb_length, y0 + sb_loc[1] * h_box * sb_txtloc[1]
        else:
            x_sb_txt, y_sb_txt = x0_sb + sb_txtloc[0] * sb_length, y0 - (1 - sb_loc[1]) * h_box * sb_txtloc[1] + sb_loc[1] * h_box
        x_sb, y_sb = [x0_sb, x1_sb], [y0_sb, y1_sb]
        xmin, xmax, ymin, ymax = ax.axis()
        width, height = xmax - xmin, ymax - ymin
        ax.plot(x_sb, y_sb, linewidth=sb_lw, color=sb_txtcolor)
        ax.text(x_sb_txt, y_sb_txt, '%d %s' % (sb_length, sb_units), color=sb_txtcolor)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw_cuboid(ax, xx, yy, zz, color='c', lw=2, **kwargs):


    xmin, xmax = np.nanmin(xx), np.nanmax(xx)
    ymin, ymax = np.nanmin(yy), np.nanmax(yy)
    zmin, zmax = np.nanmin(zz), np.nanmax(zz)
    rx = [xmin, xmax]
    ry = [ymin, ymax]
    rz = [zmin, zmax]
    w, h, d = xmax - xmin, ymax - ymin, zmax - zmin
    for s, e in itertools.combinations(np.array(list(itertools.product(rx, ry, rz))), 2):
        dist = np.linalg.norm(s - e)
        if dist in [w, h, d]:
            ax.plot3D(*zip(s, e), color=color, lw=lw, **kwargs)

    ax.set_xlim(rx)
    ax.set_ylim(ry)
    ax.set_zlim(rz)
    set_axes_equal(ax)


def add_color_wheel(fig=None, fignum=1, figsize=__figsize__,
         rect=[0.68, 0.65, 0.2, 0.2],
         cmap=None, cmapname='hsv',
         norm=None, values=[-np.pi, np.pi],
         n=2056,
         ring=True,
         text='Phase',
         fontsize=__fontsize__,
         ratio=1, text_loc_ratio=0.35, text_loc_angle=np.pi*1.07,
         **kwargs
         ):
    if fig is None:
        fig = plt.figure(num=fignum, figsize=figsize)

    subax = fig.add_axes(rect, projection='polar')
    subax._direction = 2*np.pi

    if cmap is None or norm is None:
       colors, cmap, norm = get_colors_and_cmap_using_values(values, cmap=cmapname, n=n)

    cb = mpl.colorbar.ColorbarBase(subax,
                                   cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')

    # aesthetics - get rid of border and axis labels
    cb.outline.set_visible(False)
    subax.set_axis_off()

    if ring:
        w = values[1] - values[0]
        subax.set_rlim([values[0] - w * ratio , values[1]]) # This makes it a color RING not a wheel (filled circle)
    # addtext(subax, text, np.pi*1.07, (values[0] - w/2.9 ), color='w', fontsize=fontsize)
    addtext(subax, text,
            text_loc_angle,
            values[0] - w * ratio * (1/ (1/text_loc_ratio * ratio + 1)), color='w', fontsize=fontsize)


    # subax2 = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    # plot([0, 1], [0, 1], ax=subax2)
    # addtext(subax2, text, 0, 0, color='w', fontsize=fontsize)
    # # print(values[0], values[0] - w)
    return subax, cb
## misc.
def simplest_fraction_in_interval(x, y):
    """Return the fraction with the lowest denominator in [x,y]."""
    if x == y:
        # The algorithm will not terminate if x and y are equal.
        raise ValueError("Equal arguments.")
    elif x < 0 and y < 0:
        # Handle negative arguments by solving positive case and negating.
        return -simplest_fraction_in_interval(-y, -x)
    elif x <= 0 or y <= 0:
        # One argument is 0, or arguments are on opposite sides of 0, so
        # the simplest fraction in interval is 0 exactly.
        return Fraction(0)
    else:
        # Remainder and Coefficient of continued fractions for x and y.
        xr, xc = modf(1/x);
        yr, yc = modf(1/y);
        if xc < yc:
            return Fraction(1, int(xc) + 1)
        elif yc < xc:
            return Fraction(1, int(yc) + 1)
        else:
            return 1 / (int(xc) + simplest_fraction_in_interval(xr, yr))

def approximate_fraction(x, e):
    """Return the fraction with the lowest denominator that differs
    from x by no more than e."""
    return simplest_fraction_in_interval(x - e, x + e)

def get_mask4erroneous_pts(x, y, thd=1):
    """
    Retruns a mask that can be sued to hide erroneous data points for 1D plots
    ... e.g. x[mask] and y[mask] hide the jumps which appear to be false to human eyes
    ... Uses P = dy/dx / y to determine whether data points appear to be false
        If P is high, we'd expect a jump. thd is a threshold of P.

    Parameters
    ----------
    x: 1d array
    y: 1d array
    thd: float, threshold on P (fractional dy/dx)

    Returns
    -------
    mask: 1d bool array

    """
    # remove nans
    keep_x, keep_y = ~np.isnan(x), ~np.isnan(y)
    keep = keep_x * keep_y
    x, y = x[keep], y[keep]


    fractional_dydx =  np.gradient(y, x) / y
    mask = np.abs(fractional_dydx) < thd
    mask = np.roll(mask, 1) # shift the resulting array (the convention of np.gradient)
    return mask

def tight_layout(fig, rect=[0, 0.03, 1, 0.95]):
    """
    Reminder for myself how tight_layout works with the ect option
    fig.tight_layout(rect=rect)
    Parameters
    ----------
    fig
    rect

    Returns
    -------
    """
    fig.tight_layout(rect=rect)

# data extraction from fig
def get_scatter_data_from_fig(fig, axis_number=0):
    """
    Return x, y data of scattered data in a figure instance
    ... It requires a different code to extract scattered data from a Figure instance, compared to plt.plot() output (type: line?)
    ... Scattered data are stored as an collections.collection object.

    Parameters
    ----------
    fig: matplotlib.Figure object
    axis_number: int, number to specify which axis user refers to. ax = fig.axes[axis_number]

    Returns
    -------
    data_list: list, each element of a list is a 2d array which store x,y coordinates of the scattered data points
    """

    n_col = len(fig.axes[axis_number].collections)
    data_list = []
    for i in range(n_col):
        data_list.append(fig.axes[axis_number].collections[i].get_offsets())
    return data_list

def get_plot_data_from_fig(fig, axis_number=0):
    """
    Returns a list of data included in the figure
    ... this function extracts data points for fig.ax.lines
    ... Any other data must be returned

    Parameters
    ----------
    fig
    axis_number

    Returns
    -------

    """
    nlines = len(fig.axes[axis_number].lines)
    xlist, ylist = [], []
    for i in range(nlines):
        x = fig.axes[axis_number].lines[i]._xorig
        y = fig.axes[axis_number].lines[i]._yorig
        xlist.append(x)
        ylist.append(y)
    return xlist, ylist

## Interactive plotting
class LineDrawer(object):
    """
    Class which allows users to draw lines/splines by clicking pts on the plot
        ... Default: lines/splines are closed.
        ... make sure that matplotlib backend is interactive

    Procedure for self.draw_lines() or self.draw_splines:
        It uses plt.ginput()
        1. Add a point by a left click
        2. Remove a point by a right click
        3. Stop interaction (move onto the next line to draw)

    Example
        # Pass matplotlib.axes._subplots.AxesSubplot object whose coordinates are used for extracting pts
        ld = LineDrawer(ax)

        # Draw lines/splines
        ld.draw_lines(n=5) # Draw 5 lines (connecting 5 set of points)
        # ld.draw_splines(n=2) # Or draw 2 splines based on the clicked points

        xs, ys = ld.xs, ld.ys # Retrieve x and y coords of pts used to draw lines/splines
        # xis, yis = ld.xis, ld.yis # Retrieve x and y coords for each spline

        # plot the first contour
        plt.plot(xs[0], ys[0]

        # for example, I could feed this contour to compute a line integral using vel.compute_circulation()


    """

    def __init__(self, ax):
        self.ax = ax

    def get_contour(self, npt=100, close=True):
        ax = self.ax
        xy = plt.ginput(npt)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        #         line = ax.scatter(x,y, marker='x', s=20, zorder=100)
        #         ax.figure.canvas.draw()
        #         self.lines.append(line)

        if close:
            # append the starting x,y coordinates
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

        self.x = x
        self.y = y

        return x, y

    def draw_lines(self, n=1, close=True):
        ax = self.ax
        xs, ys = [], []
        for i in range(n):
            x, y = self.get_contour(close=close)
            xs.append(x)
            ys.append(y)

            ax.plot(x, y)

        self.xs = xs
        self.ys = ys

    def spline_fit(self, x, y, n=1000):
        from scipy import interpolate
        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([x, y], s=0, per=True)

        # evaluate the spline fits for 1000 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, n), tck)

        return xi, yi

    def draw_splines(self, n=1, npt=100, n_sp=1000, close=True):
        ax = self.ax

        xs, ys = [], []
        xis, yis = [], []
        for i in range(n):
            x, y = self.get_contour(npt=npt, close=close)
            xi, yi = self.spline_fit(x, y, n=n_sp)

            xs.append(x)
            ys.append(y)

            xis.append(xi)
            yis.append(yi)

            # ax.plot(x, y)
            ax.plot(xi, yi)

        self.xs = xs
        self.ys = ys
        self.xis = xis
        self.yis = yis

    def return_pts_on_splines(self):
        return self.xis, self.yis

    def close(self):
        plt.close()


class PointFinder(object):
    def __init__(self, ax, xx, yy, weight=None):
        self.ax = ax
        self.xx = xx
        self.yy = yy
        self.ind = None

        if weight is None:
            self.weight = np.ones_like(xx)
        else:
            self.weight = weight

    def get_pts(self, npt=100):
        def find_indices(xx, yy, xs, ys):
            xg, yg = xx[0, :], yy[:, 0]
            xmin, xmax, ymin, ymax = xg.min(), xg.max(), yg.min(), yg.max()
            # i_list, j_list = [], []
            inds = []
            for n in range(len(xs)):
                if xs[n] > xmin and xs[n] < xmax and ys[n] > ymin and ys[n] < ymax:

                    X = np.abs(xg - xs[n])
                    Y = np.abs(yg - ys[n])
                    j = int(np.where(X == X.min())[0])
                    i = int(np.where(Y == Y.min())[0])
                    # i_list.append(i)
                    # j_list.append(j)
                else:
                    i, j = np.nan, np.nan
                inds.append(np.asarray([i, j]))
            return inds


        ax = self.ax
        xy = plt.ginput(npt)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]

        inds = find_indices(self.xx, self.yy, x, y)
        self.ind = inds
        self.x = x
        self.y = y
        return x, y, inds

    def find_local_center_of_mass(self, kernel_radius=2):
        def get_subarray(arr, i, j, kernel_radius):
            arr = np.asarray(arr)
            nrows, ncols = arr.shape

            imax = i + kernel_radius
            imin = i - kernel_radius
            jmax = j + kernel_radius
            jmin = j - kernel_radius


            if imax >= nrows:
                imax = nrows - 1
            if imin < 0:
                imin = 0
            if jmax >= ncols:
                jmax = ncols - 1
            if jmin < 0:
                jmin = 0
            subarr = arr[imin:imax, jmin:jmax]
            return subarr

        xcs, ycs = [], []


        for n, idx in enumerate(self.ind):
            if ~np.isnan(idx[0]):
                xx_sub = get_subarray(self.xx, idx[0], idx[1], kernel_radius=kernel_radius)
                yy_sub = get_subarray(self.yy, idx[0], idx[1], kernel_radius=kernel_radius)
                weight_sub = get_subarray(self.weight, idx[0], idx[1], kernel_radius=kernel_radius)

                xc = np.nansum(xx_sub * weight_sub) / np.nansum(weight_sub)
                yc = np.nansum(yy_sub * weight_sub) / np.nansum(weight_sub)
            else:
                xc, yc = np.nan, np.nan
            xcs.append(xc)
            ycs.append(yc)

            self.ax.scatter([xc], [yc], marker='x', color='k')
        self.xc = xcs
        self.yc = ycs

        return xcs, ycs

    def get_local_center_of_mass(self, npt=100, kernel_radius=2):
        x, y, inds = self.get_pts(npt=npt)
        xcs, ycs = self.find_local_center_of_mass(kernel_radius=kernel_radius)
        return xcs, ycs

    # def get_local_center_of_mass(self, weight, kernel_size=3):
    #     from scipy import ndimage
    #     import numpy as np
    #     arr_conv = ndimage.generic_filter(weight, np.nanmean, size=kernel_size,
    #                                       mode='constant', cval=np.NaN)

## backend
def get_current_backend():
    gui = mpl.get_backend()
    print(gui)
    return gui

def list_available_backends():
    current_backend = mpl.get_backend()

    gui_backends = [i for i in mpl.rcsetup.interactive_bk]
    non_gui_backends = mpl.rcsetup.non_interactive_bk
    # gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']

    backends = gui_backends + non_gui_backends

    available_backends = []

    print ("Non Gui backends are:", non_gui_backends)
    print ("Gui backends I will test for", gui_backends)
    for backend in backends:
        try:
            mpl.use(backend, warn=False, force=True)
            available_backends.append(backend)
        except:
            continue
    print('Available backends:')
    print(available_backends)

    mpl.use(current_backend)
    print("Currently using:", mpl.get_backend() )

def use_backend(name='agg'):
    mpl.use(name)


# smooth a curve using convolution
def smooth(x, window_len=11, window='hanning', log=False):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with a given signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

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
        return y[(window_len//2-1):(window_len//2-1)+len(x)]
    else:
        return np.exp(y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)])