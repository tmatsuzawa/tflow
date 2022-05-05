'''
Module for plotting and saving figures
'''
import os, copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import mpl_toolkits.axes_grid1 as axes_grid
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits
import seaborn as sns
from cycler import cycler
from skimage import measure
import pylab as pl

import itertools
from scipy.optimize import curve_fit
from scipy import interpolate
import numpy as np
from fractions import Fraction
from math import modf
import pickle
from scipy.stats import binned_statistic
from numpy import ma
import scipy, seaborn, h5py

# import ilpm.vector as vec
# comment this and plot_fit_curve if it breaks
import tflow.std_func as std_func
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")



#Global variables
#Default color cycle: iterator which gets repeated if all elements were exhausted
#__color_cycle__ = itertools.cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
__def_colors__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
__color_cycle__ = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])  #matplotliv v2.0
__old_color_cycle__ = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  #matplotliv classic
__fontsize__ = 11
__figsize__ = (7.54, 7.54)
cmap = 'magma'

# See all available arguments in matplotlibrc
params = {'figure.figsize': __figsize__,
          'font.size': __fontsize__,  #text
        'legend.fontsize': __fontsize__, # legend
         'axes.labelsize': __fontsize__, # axes
         'axes.titlesize': __fontsize__,
         'xtick.labelsize': __fontsize__, # tick
         'ytick.labelsize': __fontsize__,
          'lines.linewidth': 3}


## Save a figure
def save(path, ext='pdf', close=False, verbose=True, fignum=None, dpi=None, overwrite=True, tight_layout=False,
         savedata=True, transparent=True, bkgcolor='w', **kwargs):
    """
    Save a figure from pyplot

    Parameters
    ----------
    path: string
        The path (and filename, without the extension) to save the
        figure to.
    ext: string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
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
    fignum
    dpi
    overwrite
    tight_layout
    savedata
    transparent
    bkgcolor
    kwargs

    Returns
    -------

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
    if transparent:
        plt.savefig(savepath, dpi=dpi, transparent=transparent, **kwargs)
    else:
        plt.savefig(savepath, dpi=dpi, facecolor=bkgcolor, **kwargs)

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
default_custom_cycler = {'color': ['r', 'b', 'g', 'y'],
                          'linestyle': ['-', '-', '-', '-'],
                          'linewidth': [3, 3, 3, 3],
                          'marker': ['o', 'o', 'o', 'o'],
                          's': [0,0,0,0]}

def set_fig(fignum, subplot=111, dpi=100, figsize=None,
            custom_cycler=False, custom_cycler_dict=default_custom_cycler, # advanced features to change a plotting style
            **kwargs):
    """
    Returns Figure and Axes instances
    ... a short sniplet for
        plt.figure(fignum, dpi=dpi, figsize=figsize)
        plt.subplot(subplot, **kwargs)

    Parameters
    ----------
    fignum: int, figure number
    subplot: int, A 3-digit integer. The digits are interpreted as if given separately as three single-digit integers, i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5). Note that this can only be used if there are no more than 9 subplots.
    dpi: int,
    figsize: tuple, figure size
    custom_cycler: bool, If True, it enables users to customize a plot style (color cycle, marker cycle, linewidth cycle etc.)
        ... The customized cycler could be passed to custom_cycler_dict.
    custom_cycler_dict: dict, A summary of a plotting style.
        ... E.g.- default_custom_cycler = {'color': ['r', 'b', 'g', 'y'],
                                          'linestyle': ['-', '-', '-', '-'],
                                          'linewidth': [3, 3, 3, 3],
                                          'marker': ['o', 'o', 'o', 'o'],
                                          's': [0,0,0,0]}
        ... The dictionary is turned into a list of cyclers, and passed to ax.set_prop_cycle(custom_cycler).

    kwargs: Visit plt.subplot(**kwargs) for available kwargs

    Returns
    -------
    fig: Figure instance
    ax: Axes instance
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
    if subplot is None:
        subplot = 111
    # >=matplotlib 3.4: fig.add_subplot() ALWAYS creates a new axes instance
    # <matplotlib 3.4: fig.add_subplot() returns an existing Axes instance if it existed
    # ax = fig.add_subplot(subplot, **kwargs, )
    # >matplotlib 3.4 plt.suplot() continues to reuse an existing Axes with a matching subplot spec and equal kwargs.
    ax = plt.subplot(subplot, **kwargs)

    if custom_cycler:
        apply_custom_cyclers(ax, **custom_cycler_dict)

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

def plot(x, y=None, fmt='-', fignum=1, figsize=None, label='', color=None, subplot=None, legend=False,
         fig=None, ax=None, maskon=False, thd=1, xmin=None, xmax=None,
         set_bottom_zero=False, symmetric=False, #y-axis
         set_left_zero=False,
         smooth=False, smoothlog=False, window_len=5, window='hanning',
         custom_cycler=None, custom_cycler_dict=default_custom_cycler,
         return_xy=False, **kwargs):
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
    if custom_cycler:
        apply_custom_cyclers(ax, **custom_cycler_dict)

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
    if xmax is not None:
        keep *= x < xmax
    if xmin is not None:
        keep *= x >= xmin

    if smooth:
        x2plot = x[keep]
        y2plot = smooth1d(y[keep], window_len=window_len, window=window)
    elif smoothlog:
        x2plot = x[keep]
        try:
            logy2plot = smooth1d(np.log10(y[keep]), window_len=window_len, window=window)
            y2plot = 10**logy2plot
        except:
            y2plot = y[keep]
    else:
        x2plot, y2plot = x[keep], y[keep]
    if color is None:
        ax.plot(x2plot, y2plot, fmt, label=label, **kwargs)
    else:
        ax.plot(x2plot, y2plot, fmt, color=color, label=label, **kwargs)

    if legend:
        ax.legend()

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_ylim(-yabs, yabs)
    if return_xy:
        return fig, ax, x2plot, y2plot
    else:
        return fig, ax


def plot_multicolor(x, y=None, colored_by=None, cmap='viridis',
                    fignum=1, figsize=None,
                    subplot=None,
                    fig=None, ax=None, maskon=False, thd=1,
                    linewidth=2, vmin=None, vmax=None, verbose=True, **kwargs):
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
    xminOrg, xmaxOrg = ax.get_xlim()
    yminOrg, ymaxOrg = ax.get_ylim()

    if y is None:
        y = copy.deepcopy(x)
        # x = np.arange(len(x))
    # Make sure x and y are np.array
    x, y = np.asarray(x), np.asarray(y)

    if colored_by is None:
        if verbose:
            print('... colored_by is None. Pass a list/array by which line segments are colored. Using x instead...')
        colored_by = x

    if len(x) > len(y):
        if verbose:
            print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    elif len(y) > len(x):
        if verbose:
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
    ax.set_xlim(min(xminOrg, x.min()), max(xmaxOrg, x.max()))
    ax.set_ylim(min(yminOrg, y.min()), max(ymaxOrg, y.max()))

    return fig, ax



def plot_with_varying_alphas(x, y=None, color=next(__color_cycle__), alphas=None,
                    fignum=1, figsize=None,
                    subplot=None,
                    fig=None, ax=None,
                    xmin=None, xmax=None,
                    maskon=False, thd=1, # Filter out erroneous data by threasholding
                    linewidth=2, **kwargs):
    """
    Plots a curve with varying alphas (e.g. fading curves)
    ... plt.plot(x, y, alpha=alpha) does not allow varying alpha.
    ... A workaround for this is to use LineCollection. i.e. create lines for each segment, then assign different alpha values

    Parameters
    ----------
    x
    y
    color: color of the line
    alphas: list/array with the same length as x and y
        ... default:  alphas = 1 - np.linspace(0, 1, len(x))  (linearly fade)

    Parameters
    ----------
    x
    y
    color
    alphas
    fignum
    figsize
    subplot
    fig
    ax
    xmin
    xmax
    maskon
    thd
    linewidth
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

    if y is None:
        y = copy.deepcopy(x)
        # x = np.arange(len(x))
    if alphas is None:
        alphas = 1 - np.linspace(0, 1, len(x)) # default alphas
    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1
    alphas[np.isnan(alphas)] = 0

    # Make sure x and y are np.array
    x, y, alphas = np.asarray(x), np.asarray(y), np.asarray(alphas)

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
    if xmin is not None:
        cond = x > xmin
        mask = mask * cond
    if xmax is not None:
        cond = x < xmax
        mask = mask * cond


    x, y, alphas= x[mask], y[mask], alphas[mask]
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Get RGBA values of the specified color
    if type(color) == str:
        try:
            rgb = hex2rgb(cname2hex(color))
        except:
            rgb = hex2rgb(color) # Returned values are [0-255, 0-255, 0-255]
        rgba = np.append(rgb/255, 1).astype(float) # RGBA values must be between 0-1
        # Prepare an array to specify a color for each segment
        colors = np.tile(rgba, (len(x), 1))
    elif type(color) in [tuple, list, np.array]:
        if len(color) == 3:
            colors = np.tile(np.append(color, 1), (len(x), 1))
        elif len(color) == 4:
            colors = np.tile(color, (len(x), 1))
        else:
            raise ValueError('plot_with_varying_alphas: color must be a tuple/list/1d array with 3 or 4 elements (rgb or rgba)')
    # Insert the alphas specified by users
    colors[:, -1] = alphas
    # Create a line collection instead of a single line
    lc = LineCollection(segments, color=colors)

    lc.set_linewidth(linewidth)
    lines = ax.add_collection(lc)

    # autoscale does not work for collection => manually set x/y limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    return fig, ax

def plot_with_arrows(x, y=None, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False, fig=None, ax=None, maskon=False, thd=1, **kwargs):
    """
    Add doc later
    Parameters
    ----------
    x
    y
    fignum
    figsize
    label
    color
    subplot
    legend
    fig
    ax
    maskon
    thd
    kwargs

    Returns
    -------

    """
    fig, ax = plot(x, **kwargs)
    lines = ax.get_lines()
    for i, line in enumerate(lines):
        add_arrow_to_line(ax, line)
    return fig, ax


def plot3d(x, y, z, fignum=1, figsize=None, label='', color=None, subplot=None, fig=None,
           ax=None, labelaxes=True, aspect='auto', **kwargs):
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
    if aspect=='equal':
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


def plot_isosurface(qty, isovalue, xxx, yyy, zzz, cmap='Spectral',
                    r=None, xc=0, yc=0, zc=0, fill_value=0,
                    fignum=1, subplot=None, lw=1,
                    figsize=(8, 8), labelaxes=True, **kwargs):
    """
    Plots a isosurface given a 3D data, value at which isosurface is defined, 3D grids in Cartesian coordinates)
    ... the isosurface is extracted by the marching cube algorithm

    Parameters
    ----------
    qty: 3D array, a scalar field
    isovalue: float, the value at which the isosurface is extracted
    xxx: 3D array, x component of the positional grid
    yyy: 3D array, y component of the positional grid
    zzz: 3D array, z component of the positional grid
    cmap: str, name of the color map used to plot the isosurface
    r: float,
    ... If provided, it treats the qty[R > r] = fill_value where R = np.sqrt(xx**2 + yy**2 + zz**2)
    ... This is used for a simple fitering the problematic outliers near the boundaries
    xc: float/int, x-coordinate of the origin in case r is not None
    yc: float/int, y-coordinate of the origin in case r is not None
    zc: float/int, z-coordinate of the origin in case r is not None
    fill_value: float/int
    ... If r is not None, it sets qty[R > r] = fill_value where R = np.sqrt(xx**2 + yy**2 + zz**2)
    fignum: int, figure number (>=1)
    subplot: int, e.g. 121, 111, 331, default=None
    ... the three digit notaton for the matplotlib
    figsize: tuple, figure size in inches e.g.- (8, 8)
    labelaxes: bool, default True
    ... If True, it labels each axis as x(mm), y(mm), z(mm)

    Returns
    -------
    fig, ax: matplotlib.figure.Figure instance, matplotlib.axes.Axes instance,
    """
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

    def cart2sph(x, y, z):
        """
        Transformation: cartesian to spherical
        z = r cos theta
        y = r sin theta sin phi
        x = r sin theta cos phi

        Parameters
        ----------
        x: array / float /int
        y: array / float /int
        z: array / float /int

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

    if np.sum(np.isnan(qty)) > 0:
        raise ValueError(
            'plot_isosurface: qty contains np.nan. skimage.measure.marching_cubes_lewiner does not work with nans.')
    dx, dy, dz = get_grid_spacing(xxx, yyy, zzz)

    qty_ = copy.deepcopy(qty)

    if r is not None:
        rrr, tttheta, ppphi = cart2sph(xxx - xc, yyy - yc, zzz - zc)
        qty_[rrr > r] = fill_value
    verts, faces, normals, vals = measure.marching_cubes_lewiner(qty_, isovalue, spacing=(dy, dx, dz))

    verts[:, 0] += np.min(yyy)
    verts[:, 1] += np.min(xxx)
    verts[:, 2] += np.min(zzz)

    fig, ax = set_fig(fignum, subplot, figsize=figsize, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                    cmap=cmap, lw=lw, **kwargs)
    set_axes_equal(ax)
    if labelaxes:
        ax.set_xlabel('$x~(mm)$')
        ax.set_ylabel('$y~(mm)$')
        ax.set_zlabel('$z~(mm)$')
    return fig, ax

def plot_spline(x_, y_, order=3,
                fignum=1, figsize=None, subplot=None,
                fig=None, ax=None, log=False,
                label='', color=None, legend=False,
                maskon=False, thd=1., **kwargs):
    """
    Plots a spline representation of a curve (x against y)

    Parameters
    ----------
    x: 1d array-like
    y: 1d array-like
    order: int, order of spline interpolation
    fignum: int, figure number, default=1
    figsize: tuple, figure size e.g. (8, 8) in inch
    subplot# int, e.g.- 121- matplotlib shorthand notation
    fig: matplotlib.figure.Figure instance, default: None
    ax: matplotlib.axes.Axes instance, default: None
        ... If passed, this function plots a curve on the given ax.
    label: str, label of the curve
    color: str, color e.g.- 'r' for red, 'b' for blue. Consult mpl website for the full color code.
    legend: bool, If True, ax.legend() is called.
    maskon: bool, If True, it uses get_mask4erroneous_pts() to spot potentially erroneous values, and hides them.
    thd: float, This argument is only relevant if maskon=True. This is a parameter which controls the tolerance of the jumpiness of hte plot.
        ... The higher thd is, the less inputs gets hide.
    kwargs: dict, The other keyword arguments gets passed to ax.plot()

    Returns
    -------
    fig, ax: matplotlib.figure.Figure instance, matplotlib.axes.Axes instance
    """

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    if log:
        x = np.log10(copy.deepcopy(x_))
        y = np.log10(copy.deepcopy(y_))
    else:
        x, y = x_, y_

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
    cond1, cond2 = ~np.isnan(x), ~np.isnan(y)
    keep = cond1 * cond2
    x, y = x[keep], y[keep]


    try:
        if maskon:
            mask = get_mask4erroneous_pts(x, y, thd=thd)
        else:
            mask1 = ~np.isnan(x)
            mask2 = ~np.isnan(y)
            mask = mask1 * mask2
        spl_func = interpolate.UnivariateSpline(x[mask], y[mask], k=order)
    except:
        x, y, yerr = get_binned_stats(x, y, n_bins=len(x))
        if maskon:
            mask = get_mask4erroneous_pts(x, y, thd=thd)
        else:
            mask = [True] * len(x)
        spl_func = interpolate.UnivariateSpline(x[mask], y[mask], k=order)
    x2plot = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    y2plot = spl_func(x2plot)

    if log:
        x2plot, y2plot = 10**x2plot, 10**y2plot

    if color is None:
        ax.plot(x2plot, y2plot, label=label, **kwargs)
    else:
        ax.plot(x2plot, y2plot, color=color, label=label, **kwargs)

    if legend:
        ax.legend()
    return fig, ax

def plot_date(dates, y,
            fignum=1, figsize=None, label='', color=None, subplot=None, legend=False,
            fig=None, ax=None, set_bottom_zero=False, **kwargs):
    """
    A function to plot values against dates with format "2020-01-01"

    Parameters
    ----------
    dates: 1d array-like of dates- each entry must be in the format "YYYY-MM-DD"
    y: 1d array-like
    fignum: int, fignure number
    figsize: tuple, figure size e.g.- (8, 8)
    label: label kwarg in plt.plot_date
    color: str,  color kwarg in plt.plot_date
    subplot: int, 3-digit notation to specify a subplot
    legend: bool
    fig: mpl.figure.Figure instance- if given, it will just return this instance at the end
    ax: mpl.axes.Axes instance- if given, it plots the given inputs on this subplot.
    set_bottom_zero: bool, if True, it sets the ymin=0
    kwargs: the keyword arguments will be passed to plt.plot_date()

    Returns
    -------
    fig, ax: matplotlib.figure.Figure instance, matplotlib.axes.Axes instance

    """

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    # Make sure x and y are np.array
    if len(dates) > len(y):
        print("Warning : x and y data do not have the same length")
        dates = dates[:len(y)]
    elif len(y) > len(dates):
        print("Warning : x and y data do not have the same length")
        y = y[:len(dates)]

    # remove nans
    keep = ~np.isnan(dates) * ~np.isnan(y)
    dates, y = dates[keep], y[keep]

    ax.plot_date(dates, y, label=label, color=color, **kwargs)
    if legend:
        ax.legend()

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    return fig, ax


def pie(sizes, labels=None, explode=None, autopct='%1.1f%%', startangle=90, shadow=False, sort=True,
        fignum=1, figsize=None, subplot=None,
        fig=None, ax=None, **kwargs):
    """
    A wrapper for plt.pie
    ... a main difference from the original plt.plot is the sorting feature. It automatically sorts the portions from the largest to smallest.
    ... If one
    """

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

    if sort:
        if explode is None:
            explode = [0] * len(sizes)
        if labels is None:
            labels_dummy = [''] * len(sizes)
            sizes, labels_dummy, explode = sort_n_arrays_using_order_of_first_array([sizes, labels_dummy, explode])
        else:
            sizes, labels, explode = sort_n_arrays_using_order_of_first_array([sizes, labels, explode])

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    ax.pie(sizes, explode=explode, labels=labels, autopct=autopct,
           shadow=shadow, startangle=startangle, **kwargs)
    ax.axis('equal')

    return fig, ax


def plot_saddoughi(fignum=1, fig=None, ax=None, figsize=None,
                   # label='Re$_{\lambda} \\approx 600 $ \n Saddoughi and Veeravalli, 1994',
                   label='Re$_{\lambda} \\approx 600 $ \n SV, 1994',
                   color='k', alpha=0.6, subplot=None, cc=1, legend=False,
                   plotEk=False,
                   **kwargs):
    """
    plot universal 1d energy spectrum (Saddoughi, 1992)

    E(k)=C epsilon^(2/3)k^(-5/3), E11(k)=C1 epsilon^(2/3)k^(-5/3)
    # # In iso, homo, turbulence, C1 = 18/55 C. (Pope 6.242)
    # c = 1.6
    # c1 = 18. / 55. * c

    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()

    x = np.asarray([1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    y = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])
    y *= cc

    if plotEk:
        x, y_ = resample(x, y, n=100, mode='loglog')
        y = 0.5 * x ** 3 * np.gradient(1 / x * np.gradient(y_, x), x)

    ax.plot(x, y, color=color, label=label, alpha=alpha,**kwargs)
    if legend:
        ax.legend()
    tologlog(ax)
    if plotEk:
        labelaxes(ax, '$\kappa \eta$', '$E / (\epsilon\\nu^5)^{1/4}$')
    else:
        labelaxes(ax, '$\kappa_1 \eta$', '$E_{11} / (\epsilon\\nu^5)^{1/4}$')
    return fig, ax

def plot_saddoughi_struc_func(fignum=1, fig=None, ax=None, figsize=None,
                              label='Re$_{\lambda} \approx 600 $ \n Saddoughi and Veeravalli, 1994',
                              color='k', alpha=0.6, subplot=None,
                              legend=False,  **kwargs):
    """
    Plots the second order structure function on Saddoughi & Veeravalli, 1994

    Parameters
    ----------
    fignum
    fig
    ax
    figsize
    label
    color: str, array-like (1d)
    alpha
    subplot
    legend
    marker: str or list
        ... Unlike the plt.scatter(), this accepts a list for markers.
        A useful feature if one wants to plot with different markers
    kwargs

    Returns
    -------
    fig, ax
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
            xmin=None, xmax=None, alpha=1.,
            set_bottom_zero=False, symmetric=False,
            set_left_zero=False,
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
        keep = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        keep = [True] * len(x)
    if xmax is not None:
        keep *= x < xmax
    if xmin is not None:
        keep *= x >= xmin

    if type(marker) == list:
        marker_list = [m for i, m in enumerate(marker) if mask[i]]
        marker = None
    else:
        marker_list = None

    if fillstyle =='none':
        # Scatter plot with open markers
        facecolors = 'none'
        if type(alpha) == float or type(alpha) == int:
            # ax.scatter(x, y, color=color, label=label, marker=marker, facecolors=facecolors, edgecolors=edgecolors, **kwargs)
            sc = ax.scatter(x[keep], y[keep], label=label, marker=marker, facecolors=facecolors, alpha=alpha, **kwargs)
        else:
            for i, alpha_ in enumerate(alpha[keep]):
                if i != 0:
                    label=None
                sc = ax.scatter(x[keep][i], y[keep][i], label=label, marker=marker, facecolors=facecolors, alpha=alpha_, **kwargs)
    else:
        if type(alpha) == float or type(alpha) == int:
            sc = ax.scatter(x[keep], y[keep], label=label, marker=marker, alpha=alpha, **kwargs)
        else:
            for i, alpha_ in enumerate(alpha[keep]):
                if i != 0:
                    label=None
                sc = ax.scatter(x[keep][i], y[keep][i], label=label, marker=marker, alpha=alpha_, **kwargs)
    if legend:
        plt.legend()

    if type(marker_list) == list:
        paths = []
        for marker in marker_list:
            if isinstance(marker, mpl.markers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mpl.markers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        xmin, xmax = ax.get_xlim()
        xabs = np.abs(max(-xmin, xmax))
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_xlim(-xabs, xabs)
        ax.set_ylim(-yabs, yabs)
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

def pdf(data, nbins=100, return_data=False, vmax=None, vmin=None,
        fignum=1, figsize=None, subplot=None, density=True, analyze=False, **kwargs):
    """
    Plots a probability distribution function of ND data
    ... a wrapper for np.histogram and matplotlib
    ... Returns fig, ax, (optional: bins, hist)

    Parameters
    ----------
    data: nd-array, list, or tuple, data used to get a histogram/pdf
    nbins: int, number of bins
    return_data: bool, If True, it returns  fig, ax, bins (centers of the bins), hist (counts or probability density values)
    vmax: float, data[data>vmax] will be ignored during counting.
    vmin: float, data[data<vmin] will be ignored during counting.
    fignum: int, figure number (the argument called "num" in matplotlib)
    figsize: tuple, figure size in inch (width x height)
    subplot: int, matplotlib subplot notation. default: 111
    density: bool, If True, it plots the probability density instead of counts.
    analyze: bool If True, it adds mean, mode, variane to the plot.
    kwargs: other kwargs passed to plot() of the velocity module

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    (Optional)
    bins: 1d array, bin centers
    hist: 1d array, probability density vales or counts
    """
    def compute_pdf(data, nbins=10, density=density):
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
        vmax = np.nanmax(data)
    if vmin is not None:
        cond2 = np.asarray(data) > vmin
    else:
        vmin = np.nanmin(data)
        cond2 = np.ones(data.shape, dtype=bool)
    data = data[cond1 * cond2]
    delta = (vmax - vmin) / nbins
    bins = np.arange(vmin, vmax+delta, delta)

    # compute a pdf
    bins, hist = compute_pdf(data, nbins=bins)
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


def cdf(data, nbins=100, return_data=False, vmax=None, vmin=None,
        fignum=1, figsize=None, subplot=None, **kwargs):
    """
    Plots a cummulative distribution function of ND data
    ... a wrapper for np.histogram and matplotlib
    ... Returns fig, ax, (optional: bins, hist)

    Parameters
    ----------
    data: nd-array, list, or tuple, data used to get a histogram/pdf
    nbins: int, umber of bins
    return_data: bool, If True, it returns  fig, ax, bins (centers of the bins), hist (counts or probability density values)
    vmax: float, data[data>vmax] will be ignored during counting.
    vmin: float, data[data<vmin] will be ignored during counting.
    fignum: int, figure number (the argument called "num" in matplotlib)
    figsize: tuple, figure size in inch (width x height)
    subplot: int, matplotlib subplot notation. default: 111
    density: bool, If True, it plots the probability density instead of counts.
    analyze: bool If True, it adds mean, mode, variane to the plot.
    kwargs: other kwargs passed to plot() of the velocity module

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    (Optional)
    bins: 1d array, bin centers
    hist: 1d array, probability density vales or counts
    """
    def compute_pdf(data, nbins=10):
        # Get a normalized histogram
        # exclude nans from statistics
        pdf, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=True)
        # len(bins) = len(hist) + 1
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, pdf

    def compute_cdf(data, nbins=10):
        """compute cummulative probability distribution of data"""
        bins, pdf = compute_pdf(data, nbins=nbins)
        cdf = np.cumsum(pdf) * np.diff(bins, prepend=0)
        return bins, cdf

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

    # compute a cdf
    bins, cdf = compute_cdf(data, nbins=nbins)
    fig, ax = plot(bins, cdf, fignum=fignum, figsize=figsize, subplot=subplot, **kwargs)

    if not return_data:
        return fig, ax
    else:
        return fig, ax, bins, cdf


def errorbar(x, y, xerr=0., yerr=0., fignum=1, marker='o', fillstyle='full', linestyle='None', label=None, mfc='white',
             subplot=None, legend=False, legend_remove_bars=False, figsize=None, maskon=False, thd=1, capsize=10,
             xmax=None, xmin=None, ax=None, **kwargs):
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
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
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
    xerr[xerr==0] = np.nan
    yerr[yerr==0] = np.nan
    if maskon:
        keep = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        keep = [True] * len(x)
    if xmax is not None:
        keep *= x < xmax
    if xmin is not None:
        keep *= x >= xmin
    if fillstyle == 'none':
        ax.errorbar(x[keep], y[keep], xerr=xerr[keep], yerr=yerr[keep], marker=marker, mfc=mfc, linestyle=linestyle,
                    label=label, capsize=capsize, **kwargs)
    else:
        ax.errorbar(x[keep], y[keep], xerr=xerr[keep], yerr=yerr[keep], marker=marker, fillstyle=fillstyle,
                    linestyle=linestyle, label=label, capsize=capsize,  **kwargs)

    if legend:
        ax.legend()

        if legend_remove_bars:
            from matplotlib import container
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    return fig, ax

def errorfill(x, y, yerr, fignum=1, color=None, subplot=None, alpha_fill=0.3, ax=None, label=None,
              legend=False, figsize=None, maskon=False, thd=1,
              xmin=None, xmax=None, smooth=False, smoothlog=False, window_len=5, window='hanning',
              set_bottom_zero=False, set_left_zero=False, symmetric=False, return_xy=False,
              **kwargs):

    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)

    #ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = color_cycle.next()
    if maskon:
        keep = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        keep = [True] * len(x)
    if xmax is not None:
        keep *= x < xmax
    if xmin is not None:
        keep *= x >= xmin

    mask2removeNans = ~np.isnan(x) * ~np.isnan(y)
    keep = keep * mask2removeNans

    if smooth:
        x2plot = x[keep]
        y2plot = smooth1d(y[keep], window_len=window_len, window=window)
    elif smoothlog:
        x2plot = x[keep]
        try:
            logy2plot = smooth1d(np.log10(y[keep]), window_len=window_len, window=window)
            y2plot = 10**logy2plot
        except:
            y2plot = y[keep]
    else:
        x2plot, y2plot = x[keep], y[keep]
    if len(yerr) == len(y):
        ymin = y2plot - yerr[keep]
        ymax = y2plot + yerr[keep]
    elif len(yerr) == 2:
        yerrdown, yerrup = yerr
        ymin = y2plot - yerrdown
        ymax = y2plot + yerrup
    else:
        ymin = y2plot - yerr
        ymax = y2plot + yerr


    p = ax.plot(x2plot, y2plot, color=color, label=label, **kwargs)
    color = p[0].get_color()
    ax.fill_between(x2plot, ymax, ymin, color=color, alpha=alpha_fill)

    #patch used for legend
    color_patch = mpatches.Patch(color=color, label=label)
    if legend:
        plt.legend(handles=[color_patch])

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_ylim(-yabs, yabs)

    if not return_xy:
        return fig, ax, color_patch
    else:
        return fig, ax, color_patch, x2plot, y2plot


def bin_and_errorbar(x_, y_, xerr=None,
                     n_bins=100, mode='linear', bin_center=True, return_std=False,
                     fignum=1, ax=None, marker='o', fillstyle='full',
                     linestyle='None', linewidth=1, label=None, mfc='white',
                     subplot=None, legend=False, figsize=None, maskon=False, thd=1, capsize=5,
                     set_bottom_zero=False, symmetric=False, #y-axis
                     set_left_zero=False,
                     return_stats=False, **kwargs):
    """
    Takes scattered data points (x, y), bin them (compute avg and std), then plots the results with error bars

    Parameters
    ----------
    x : array-like
    y : array-like
    xerr: must be a scalar or numpy array with shape (N,1) or (2, N)... [xerr_left, xerr_right]
        ... if xerr==0, it removes the error bars in x.
    yerr:  must be a scalar or numpy array with shape (N,) or (2, N)... [yerr_left, yerr_right]
    n_bins: int, number of bins used to compute a histogram between xmin and xmax
    mode: str, default: 'linear', options are 'linear' and 'log'. Select either linear binning or logarithmic binning
        ... If "linear", it computes statistics using evenly separated bins between xmin and xmax.
        ... If "log", it uses bins evenly separted in the log space. (It assumes that xmin>0)
            i.e. The bin edges are like (10^-1.0, 10^-0.5),  (10^-0.5, 10^0),  (10^0, 10^0.5), and so on.
    bin_center: bool, default: True.
        ... passed to get_binned_stats()
    return_std: bool, default: False.
        ... passed to get_binned_stats()
        ... If False, it uses standard errors as error bars, instead of using standard deviations
    fignum: int, figure number
    ax: Axes object, default: None
        ... If given, this becomes the Axes on which the results are plotted
    marker: str, default: 'o', marker style
    fillstyle: str, default: 'full'. Options: 'full', 'none'. See matplotlib scatter for more details
    linestyle: str, default:'None'
    linewidth: float, linewidth of the error bars
    label: str, label for a legend
    mfc: str, default:'white', marker face color
        ... Use this with fillstyle='none' in order to change the face color of the marker.
        ... Common usage: empty circles- fillstyle='none', mfc='white'
    subplot: int, three-digit number. e.g.-111
    legend: bool, default: False. If True, ax.legend is called at the end.
    figsize: tuple, figure size in inches
    maskon: bool, default: False
        ... This hides "suspicious" data points / outliers.
        ... See the docstr of get_mask4erroneous_pts() for more details
    thd: float, threshold value used for get_mask4erroneous_pts() to determine the outliers
    capsize: float, width of the error bars
    return_stats: bool, default: False
        ... If True, it returns the binned results (that are being plotted): x[mask], y[mask], xerr[mask], yerr[mask]
    kwargs: passed to ax.errorbar()

    Returns
    -------
    If not return_stats (default),
        fig, ax: a Figure instance, an Axes instance
    If return_stats:
        fig, ax, x[mask], y[mask], xerr[mask], yerr[mask]: a Figure instance, an Axes instance, binned results (x, y, x_err, y_err)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x_, y_ = np.array(x_), np.array(y_)
    x, y, yerr = get_binned_stats(x_, y_, n_bins=n_bins, mode = mode, bin_center=bin_center, return_std = return_std)
    if xerr is None:
        xerr = np.ones_like(x) * (x[1] - x[0]) / 2.
    elif type(xerr) in [int, float]:
        xerr = np.ones_like(x) * xerr
    xerr[xerr == 0] = np.nan
    yerr[yerr == 0] = np.nan

    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)
    if fillstyle == 'none':
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, mfc=mfc, linestyle=linestyle,
                    label=label, capsize=capsize, linewidth=linewidth, **kwargs)
    else:
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, fillstyle=fillstyle,
                    linestyle=linestyle, label=label, capsize=capsize, linewidth=linewidth,   **kwargs)
    if legend:
        ax.legend()

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        xmin, xmax = ax.get_xlim()
        xabs = np.abs(max(-xmin, xmax))
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_xlim(-xabs, xabs)
        ax.set_ylim(-yabs, yabs)

    if not return_stats: # default
        return fig, ax
    else:
        return fig, ax, x[mask], y[mask], xerr[mask], yerr[mask]

## Plot a fit curve
def plot_fit_curve(xdata, ydata, func=None, fignum=1, subplot=111, ax=None, figsize=None, linestyle='--',
                   xmin=None, xmax=None, add_equation=True, eq_loc='bl', color=None, label='fit',
                   show_r2=False, return_r2=False, p0=None, bounds=(-np.inf, np.inf), maskon=True, thd=1,**kwargs):
    """
    Plots a fit curve given xdata and ydata
    Parameters
    ----------
    xdata: 1d array
    ydata: 1d array
    func : a function to be fit- e.g. lambda x, a, b: a*x+b
    fignum: int, figure number
    subplot: int, three-digit number to specify a subplot location
    ax: Axes instance- If given, it plots on the
    figsize
    linestyle
    xmin
    xmax
    add_equation
    eq_loc
    color
    label
    show_r2
    return_r2
    p0
    bounds
    maskon
    thd
    kwargs

    Returns
    -------
    fig, ax: A Figure object, an Axes object
    popt, pcov : fit results, a covariance matrix

    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    xdata = np.array(xdata).squeeze()
    ydata = np.array(ydata).squeeze()

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

    if maskon and func!='power2':
        mask = get_mask4erroneous_pts(xdata, ydata, thd=thd)
        if any(mask):
            xdata = xdata[mask]
            ydata = ydata[mask]


    x_for_plot = np.linspace(xmin, xmax, 1000)
    if func is None or func == 'linear':
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
            try:
                addtext(ax, text, option=eq_loc)
            except:
                pass
        y_fit = std_func.linear_func(xdata, *popt)
    elif func == 'power':
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
            try:
                addtext(ax, text, option=eq_loc)
            except:
                pass
        y_fit = std_func.power_func(xdata, *popt)
    elif func == 'power2':
        print('Fitting to a linear function to the log-log plot')
        xdata[xdata<10**-16], ydata[xdata<10**-16] = np.nan, np.nan
        xdata_log, ydaya_log = np.log(xdata), np.log(ydata)

        popt, pcov = curve_fit(std_func.linear_func, xdata_log, ydaya_log, p0=p0, bounds=bounds)

        y_fit = np.exp(popt[1]) * x_for_plot ** popt[0]

        if color is None:
            fig, ax = plot(x_for_plot, y_fit, fignum=fignum, subplot=subplot,
            label = label, figsize = figsize, linestyle = linestyle, ax = ax, ** kwargs)
        else:
            fig, ax = plot(x_for_plot, y_fit, fignum=fignum, subplot=subplot,
            label = label, figsize = figsize, color = color, linestyle = linestyle, ax = ax, ** kwargs)

        if add_equation:
            text = '$y=ax^b$: a=%.2f, b=%.2f' % (np.exp(popt[1]) , popt[0])
            try:
                addtext(ax, text, option=eq_loc)
            except:
                pass
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


def plot_interpolated_curves(x, y, zoom=2, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False,
         fig=None, ax=None, maskon=False, thd=1, return_interp_func=False, **kwargs):
    """
    plot a graph using given x, y
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
    # f = scipy.interpolate.interp1d(x[keep], y[keep], fill_value="extrapolate")
    # FOR A SMOOTHER CURVE
    x_ = scipy.ndimage.zoom(x[keep], zoom)
    y_ = scipy.ndimage.zoom(y[keep], zoom)
    f = scipy.interpolate.interp1d(x_, y_, fill_value="extrapolate")

    fig, ax = plot(x_, f(x_), label=label, color=color, ax=ax, legend=legend, **kwargs)
    if return_interp_func:
        return fig, ax, f
    else:
        return fig, ax

## 2D plotsFor the plot you showed at group meeting of lambda converging with resolution, can you please make a version with two x axes (one at the top, one below) one pixel spacing, other PIV pixel spacing, and add a special tick on each for the highest resolution point.
# (pcolormesh)
def color_plot(x, y, z,
               subplot=None, fignum=1, figsize=None, ax=None,
               vmin=None, vmax=None, log10=False, label=None,
               cbar=True, cmap='magma', symmetric=False, enforceSymmetric=True,
               aspect='equal', option='scientific', ntick=5, tickinc=None,
               crop=None, fontsize=None, ticklabelsize=None, cb_kwargs={}, return_cb=False,
               **kwargs):
    """

    Parameters
    ----------
    x: 2d array
    y: 2d array
    z: 2d array
    subplot: int, default is 111
    fignum
    figsize
    ax
    vmin
    vmax
    log10
    label
    cbar
    cmap
    symmetric
    aspect: str, 'equal' or 'auto
    option
    ntick
    tickinc
    crop
    kwargs
    cb_kwargs: dict, kwargs for add_colorbar()
        ... e.g. {"fformat": %.0f}
    return_cb: bool, default: False
        ... if True, this function returns fig, ax, cc, cb (colorbar instance)

    Returns
    -------
    fig:
    ax:
    cc: QuadMesh instance
    cb: colorbar instance (optional)
    """

    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
        # fig, ax = set_fig(fignum, subplot, figsize=figsize, aspect=aspect)
    if crop is not None:
        x = x[crop:-crop, crop:-crop]
        y = y[crop:-crop, crop:-crop]
        z = z[crop:-crop, crop:-crop]


    if log10:
        z = np.log10(z)

    # For Diverging colormap, ALWAYS make the color thresholds symmetric
    if cmap in ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'] \
            and enforceSymmetric:
        symmetric = True

    if symmetric:
        hide = np.isinf(z)
        keep = ~hide
        if vmin is None and vmax is None:
            v = max(np.abs(np.nanmin(z[keep])), np.abs(np.nanmax(z[keep])))
            vmin, vmax = -v, v
        elif vmin is not None and vmax is not None:
            arr = np.asarray([vmin, vmax])
            v = np.nanmax(np.abs(arr))
            vmin, vmax = -v, v
        elif vmin is not None and vmax is None:
            vmax = -vmin
        else:
            vmin = -vmax




    # Note that the cc returned is a matplotlib.collections.QuadMesh
    # print('np.shape(z) = ' + str(np.shape(z)))
    if vmin is None and vmax is None:
        # plt.pcolormesh returns a QuadMesh class object.
        cc = ax.pcolormesh(x, y, z, cmap=cmap, **kwargs)
    else:
        cc = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)

    if cbar:
        if vmin is None and vmax is None:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
        elif vmin is not None and vmax is None:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, vmin=vmin, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
        elif vmin is None and vmax is not None:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, vmax=vmax, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
        else:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, vmin=vmin, vmax=vmax, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
    ax.set_aspect(aspect)
    # set edge color to face color
    cc.set_edgecolor('face')

    if return_cb and cbar:
        return fig, ax, cc, cb
    else:
        return fig, ax, cc

#imshow
def imshow(arr, xmin=0, xmax=1, ymin=0, ymax=1, cbar=True, vmin=0, vmax=0, \
           fignum=1, subplot=111, figsize=__figsize__, ax=None, interpolation='nearest', cmap='bwr',
           cb_kwargs={}, **kwargs):
    """

    Parameters
    ----------
    arr: array-like or PIL image
    xmin: float [0., 1.)- extent=(xmin, xmax, ymin, ymax) The bounding box in data coordinates that the image will fill.
    xmax: float (0., 1.]- extent=(xmin, xmax, ymin, ymax) The bounding box in data coordinates that the image will fill.
    ymin: float [0., 1.)- extent=(xmin, xmax, ymin, ymax) The bounding box in data coordinates that the image will fill.
    ymax:  float (0., 1.]- extent=(xmin, xmax, ymin, ymax) The bounding box in data coordinates that the image will fill.
    cbar: bool, If True, fig.colorbar(ImageAxes instance, **cb_kwargs) is called.
    vmin: float, image intensity ranges from vmin to vmax
    vmax: float, image intensity ranges from vmin to vmax
    fignum: int, figure number
    subplot: int, three-digit integer to specify a subplot
    figsize: tuple, figure size- e.g. (8,8)
    interpolation: 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
                   'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
                   'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'.
    cmap: str, color map used for plt.imshow
    cb_kwargs: dict,  color bar keyward arguments can be passed in this dictionary like {'shrink': 0.5, 'pad':0.05}
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html

    Returns
    -------
    fig, ax, ima, cc: Figure, Axes, AxesImage, Colorbar instances
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
    if vmin == vmax == 0:
        ima = ax.imshow(arr, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap, **kwargs)
    else:
        ima = ax.imshow(arr, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if cbar:
        cc = fig.colorbar(ima, **cb_kwargs)
    else:
        cc = None
    return fig, ax, ima, cc


def imgScatter(x, y, imgs, img_x=None, img_y=None, subax_size=0.06,
               cmap='viridis', vmin=None, vmax=None,
               cbar=True, cb_pad='2%', cb_size='5%', cb_option='scientific', cb_label=None,
               axLim=(None, None, None, None),
               fignum=1, figsize=__figsize__,
               **kwargs):
    """
    Scatter plots images (2d arrays)
    ... This function creates additional Axes on top of the master Axes instance.

    To do1: one should be able to plot on the master Axes; however, this fails even with altering the zorder values.
    To do2: Should one be able to pass a list of imag_x as well?- this would enable plotting imgs with different resolutions
    Parameters
    ----------
    x: 1d array-like, x-coordinates of the image locations
    y: 1d array-like, y-coordinates of the image locations
    imgs: list, list of images (2d arrays
    img_x: 2d array (optional), x grid of the images- if given, this calls color_plot(img_x, img_y, imgs[i]).
        ... Otherwise, it calls imshow(imgs[i]).
    img_y: 2d array (optional), y grid of the images- if given, this calls color_plot(img_x, img_y, imgs[i]).
        ... Otherwise, it calls imshow(imgs[i]).
    subax_size: float (0, 1], default:0.06, the size of the subaxes (inner plots)
    cmap: str/cmap object- a color map of the images
    vmin: float, default:None- color bar range [vmin, vmax]
    vmax: float, default:None- color bar range [vmin, vmax]
    cbar: boolean, default:False- To toggle a color bar, vmin and vmax must be given. This is because each image could be drawn with a different color bar range.
    cb_pad: str, default:"2%" (with respect to the figure width)- Do not pass a float.
    cb_size: str, default:"2%" (with respect to the figure width)- Do not pass a float.
    cb_option: str, color bar notation, choose from 'normal' and 'scientific'
    cb_label: str, label of the color bar
    axLim: 1d array-like (xmin, xmax, ymin, ymax)- x- and y-limits of the master axes
    kwargs: dict, this gets passed to either imshow() or color_plot()

    Returns
    -------
    fig, ax, axes, cb: Figure, Axes (master), list of Axes, a color bar object
    """

    def pc2float(x):
        return float(x.strip('%')) / 100
    #
    # def float2pc(x):
    #     return "{0}%".format(x * 100)

    subaxes = []

    fig, ax = scatter(x, y, s=1, zorder=0, fignum=fignum, figsize=figsize)  # dummy plot
    if cbar:
        if vmin is None and vmax is None:
            print('imgScatter: To toggle a universal color bar, provide vmin and vmax. Color range: [vmin, vmax]')
            cb = None
            cbar = False
        else:
            cb = add_colorbar_alone(ax, [vmin, vmax], cmap=cmap, option=cb_option, label=cb_label)
    else: cb = None

    if any([a is not None for a in axLim]):
        ax.set_xlim(axLim[:2])
        ax.set_ylim(axLim[2:])
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    pad, size = pc2float(cb_pad), pc2float(cb_size)

    for i, (x_, y_) in enumerate(zip(x, y)):
        aspectRatio = imgs[i].shape[0] / imgs[i].shape[1]
        if cbar:
            subax = add_subplot_axes(ax, [(x_ - xmin) / ((xmax - xmin) * (1. + pad + size)) - subax_size / 2.,
                                                (y_ - ymin) / (ymax - ymin) - subax_size / 2.,
                                                subax_size, subax_size*aspectRatio],
                                           zorder=0, )
        else:
            subax = add_subplot_axes(ax, [(x_ - xmin) / (xmax - xmin) - subax_size / 2.,
                                                (y_ - ymin) / (ymax - ymin) - subax_size / 2.,
                                                subax_size, subax_size*aspectRatio],
                                           zorder=0, )
        if img_x is not None and img_y is not None:
            color_plot(img_x, img_y, imgs[i], ax=subax,
                             cbar=False, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        else:
            imshow(imgs[i], ax=subax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False,
                   aspect='equal',
                   xmax=1, ymax=aspectRatio,
                   **kwargs)
        subax.axis('off')
        subaxes.append(subax)

    return fig, ax, subaxes, cb


# quiver
def quiver(x, y, u, v, subplot=None, fignum=1, figsize=None, ax=None,
           inc_x=1, inc_y=1, inc=None, color='k',
           vmin=None, vmax=None, units='inches', scale=None,
           key=True, key_loc=[0.08, 1.06], key_length=None,
           key_label=None, key_units='mm/s', key_labelpos='E',
           key_pad=25., key_fmt='.1f',
           key_kwargs={},
           aspect='equal',
           **kwargs):
    """
    Wrapper for plt.quiver()

    Some tips:
    ... plt.quiver() autoscales the arrows. This may be problematic if you want to show them with absolute scales.
    ... I got a workaround for you. You can control this by toggling a boolean "absolute"
    ...... If "absolute" is False, it autoscales.
    ...... If "absolute" is True, you must supply "scale" (float)
    ......... e.g. Plot two quiver plots with the same scale
            fig1, ax1, Q1 = quiver(x1, y1, u1, v1, scale=4, fignum=1, key_length=50)
            quiver(x2, y2, u2, v2, scale=4, fignum=2, key_length=50) # scale could be Q1.scale
    ............ This ensures to plot the arrows with the same scale with the same quiver key.
                This is essential to create an animation of quiver plots to avoid distraction.

    Parameters
    ----------
    x
    y
    u
    v
    subplot
    fignum
    figsize
    ax
    inc_x
    inc_y
    inc
    color
    vmin
    vmax
    absolute
    u_ref
    key
    key_loc
    key_length
    key_label
    key_units
    key_labelpos
    key_pad
    key_fmt
    key_kwargs
    aspect
    kwargs

    Returns
    -------

    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
    if inc is not None:
        inc_x = inc_y = inc
    x_tmp, y_temp = x[::inc_y, ::inc_x], y[::inc_y, ::inc_x]
    u_tmp, v_tmp = u[::inc_y, ::inc_x], v[::inc_y, ::inc_x]
    u_norm = np.sqrt(u_tmp ** 2 + v_tmp ** 2)
    u_rms = np.sqrt(np.nanmean(u_tmp ** 2 + v_tmp ** 2))

    if vmin is None:
        vmin = np.nanmin(u_norm)
    if vmax is None:
        vmax = np.nanmax(u_norm)
    hide1 = u_norm < vmin
    hide2 = u_norm > vmax
    hide3 = np.isinf(u_norm)
    hide = np.logical_or(np.logical_or(hide1, hide2), hide3)
    cond = ~hide
    if type(color) != str:
        color = np.asarray(color)
        color = color[::inc_y, ::inc_x, :]
        if color.shape != cond.shape:
            for d in range(color.shape[-1]):
                color[..., d] = color[..., d][cond].reshape(cond.shape)
        color = color.reshape((-1, color.shape[-1]))


    if units=='inches':
        if key_length is None:
            U_absMean = np.nanmean(u_norm[cond])
            scale = U_absMean
        else:
            scale = key_length
    Q = ax.quiver(x_tmp[cond], y_temp[cond], u_tmp[cond], v_tmp[cond], color=color, units=units, scale=scale, **kwargs)

    if key:
        U_absMean = np.nanmean(u_norm[cond])
        if key_length is None:
            # key_length = 10 ** round(np.log10(U_rms))
            # key_length = 10 ** round(np.log10(U_rmedians))
            # key_length = round(U_rmedians, int(-round(np.log10(U_rmedians))) + 1) * 5
            key_length = round(u_rms, int(-round(np.log10(U_absMean))) + 2)
        if key_label is None:
            key_label = '{:' + key_fmt + '} '
            key_label = key_label.format(key_length) + key_units
        title(ax, '   ') # dummy title to create space on the canvas
        ax._set_title_offset_trans(key_pad)
        # print(key_length)
        # print(Q.scale)
        ax.quiverkey(Q, key_loc[0], key_loc[1], U=key_length, label=key_label, labelpos=key_labelpos, coordinates='axes',
                     color=color, **key_kwargs)

    ax.set_aspect(aspect)
    return fig, ax, Q


def quiver3d(udata, normalize=False, mag=1, inc=1, xinc=None, yinc=None, zinc=None,
             umin=0, umax=None, # data range to show quiver
             vmin=0, vmax=None, # colorbar range
             add_bounding_box=True, notebook=True,
             show=True,
             save=False, savepath='./vectorfield.png', verbose=True, **kwargs):
    """
    3D Quiver plot using pyvista

    Parameters
    ----------
    udata: 4d array with shape (3, y, x, z)
    normalize: bool, default: False. If True, it ignores the magnitude in udata. All vectors have the magnitude of 1.
        ... This is handy if you would like to assess the directions of the field.
    mag: float greater than 0, default:1. udata*mag gets plotted. Sometimes, it is necessary to multiply a scalar to see the quivers.
    inc: int, default:1. Increment of quivers to be plotted- if inc=1, it plots all vectors in udata.
        If inc=2, it plots vectors every 2 xsteps, 2ysteps, and 2zsteps. i.e. 1/8 of vectors in udata gets plotted
    xinc: int, default:1. Increment of quivers to be plotted along the x-axis (the third index of udata)
    yinc: int, default:1. Increment of quivers to be plotted along the y-axis (the second index of udata)
    zinc: int, default:1. Increment of quivers to be plotted along the z-axis (the fourth index of udata)
    vmin: float, default: 0. The color range is specified by [vmin, vmax]
    vmax: float, default: None. The default is the maximum value in udata
    add_bounding_box: bool, default: True. If True, it draws a bounding box of udata
    save: bool, default: False. If True, it saves an image (png) at savepath.
    savepath: str, a path where an image gets saved if save is True.
    verbose: bool, default: True. If False, it suppresses print outputs.

    Returns
    -------
    plobj, a: pyvista.Plotter instance,  returned object of pyvista.Plotter.add_arrows()

    """
    import pyvista
    if notebook:
        pyvista.set_jupyter_backend('ipygany')

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
                nrows, ncols, nstacks, duration = ux.shape
                return udata
            except:
                nrows, ncols, nstacks = ux.shape
                duration = 1
                ux = ux.reshape((ux.shape[0], ux.shape[1], ux.shape[2], duration))
                uy = uy.reshape((uy.shape[0], uy.shape[1], uy.shape[2], duration))
                uz = uz.reshape((uz.shape[0], uz.shape[1], uz.shape[2], duration))
                return np.stack((ux, uy, uz))
    def compute_direction_from_udata(udata, normalize=False, t=0):
        udata = fix_udata_shape(udata)
        dim, height, width, depth, duration = udata.shape
        ux, uy, uz = udata[0, ..., t].ravel('F'), udata[1, ..., t].ravel('F'), udata[2, ..., t].ravel('F')
        umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
        direction = np.empty((len(ux), 3))
        if normalize:
            direction[:, 0] = ux / umag
            direction[:, 1] = uy / umag
            direction[:, 2] = uz / umag
        else:
            direction[:, 0] = ux
            direction[:, 1] = uy
            direction[:, 2] = uz
        direction[np.isnan(direction)] = 0
        direction[umag==0, :] = 0
        return direction

    def get_speed(udata):
        """Returns speed from udata"""
        speed = np.zeros_like(udata[0, ...])
        dim = udata.shape[0]
        for d in range(dim):
            speed += udata[d, ...] ** 2
        speed = np.sqrt(speed)
        return speed

    udata = fix_udata_shape(udata)
    dim, height, width, depth, duration = udata.shape

    # set up coordinates
    if xinc is None: xinc = inc
    if yinc is None: yinc = inc
    if zinc is None: zinc = inc
    x, y, z = np.meshgrid(np.arange(0, width, xinc),
                          np.arange(0, height, yinc),
                          np.arange(0, depth, zinc)
                          )
    udata = udata[:, ::yinc, ::xinc, ::zinc]

    points = np.empty((x.size, 3))
    points[:, 0] = x.ravel('F')
    points[:, 1] = y.ravel('F')
    points[:, 2] = z.ravel('F')

    # data range [umin, umax]
    if umin!=0 or umax is not None:
        speed = get_speed(udata)
        keep = np.logical_and(umin <= speed, speed <= umax)
        for d in range(udata.shape[0]):
            udata[d, ~keep] = 0
    # color bar range
    if vmax is None:
        vmax = np.nanmax(udata) * mag

    # Compute a direction for the vector field
    direction = compute_direction_from_udata(udata, normalize=normalize)
    # plot using the plotting class
    plobj = pyvista.Plotter()

    a = plobj.add_arrows(points, direction, mag=mag, **kwargs)
    plobj.update_scalar_bar_range([vmin, vmax])
    if add_bounding_box:
        plobj.add_bounding_box()
    if not save and show:
        plobj.show()
    else:
        savedir = os.path.split(savepath)[0]
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if show:
            plobj.show(screenshot=savepath)
        if verbose:
            print('... A vector field image was saved at ', savepath)
    return plobj, a

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
    fig, ax, ctrs
    ... ctrs: a QuadContourSet instance

    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    if vmin is None:
        vmin = np.nanmin(psi)
    if vmax is None:
        vmax = np.nanmax(psi)
    hide1 = psi <= vmin
    hide2 = psi > vmax
    hide = np.logical_or(hide1, hide2)
    psi2plot = copy.deepcopy(psi)
    psi2plot[hide] = np.nan

    ctrs = ax.contour(x, y, psi2plot, levels, **kwargs)
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



def embed_2dplot_in_3d(pp, qq, qty, zdir='z', offset=0, cmap='viridis', vmin=None, vmax=None,
                       fignum=1, subplot=111, ax=None, figsize=None, zorder=0, alpha=1.,
                       edgecolors='none', lw=0):
    """
    This embeds a 2d plot/image in 3D plot. The 2d plot points in +z(+y, or +x) direction.
    ... This function should be used for a visualization purposes since it is not scientifically informative!
    ... Use ax.plot_surface to insert a 2d plot on an arbitrary surface embedded in 3D.
    ... This function can be usd to show three 2D projections of a 3D data.

    Parameters
    ----------
    pp: 2d array, x coordinate of the 2d plot
    qq: 2d array, y coordinate of the 2d plot
    qty: 2d array, data of the 2d plot
    zdir: str, direction of the normal vector. choices are 'z', 'x', 'y'
    offset: float,
    cmap: str, cmap instance,
    vmin: float, colormap spans from vmin to vmap
    vmax: float, colormap spans from vmin to vmap
    fignum: int, number of the figure
    subplot: int, subplot of the figure (three digit notation, default: 111)
    figsize: tuple, figure size in inches
    zorder: int, zorder of the produced surface
    alpha: float, alpha of the embedded plot
    edgecolors: str, edge color that is passed to ax.plot_surface()- default: 'none'
    lw: float, linewidth that is passed to ax.plot_surface()- default: 0

    Returns
    -------
    fig, ax, surf

    """
    if zdir == 'z':
        xx, yy = pp, qq
        zz = np.zeros_like(pp) + offset  # dummy height for plot_surface
    elif zdir == 'x':
        yy, zz = pp, qq
        xx = np.zeros_like(pp) + offset
    else:
        zz, xx = pp, qq
        yy = np.zeros_like(pp) + offset

    cmap = plt.get_cmap(cmap, 100)
    if vmin is None:
        vmin = np.nanmin(qty)
    if vmax is None:
        vmax = np.nanmax(qty)

    if ax is None:
        fig, ax = set_fig(fignum=fignum, subplot=subplot, figsize=figsize, projection='3d')
    else:
        fig = plt.gcf()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(qty))

    surf = ax.plot_surface(xx, yy, zz, cstride=1, rstride=1,
                           facecolors=colors, shade=False,
                           zorder=zorder, alpha=alpha,
                           edgecolors=edgecolors, lw=lw, antialiased=False,
                           )
    #     graph.add_colorbar_alone(ax, [vmin, vmax], cmap=cmap, label=label)
    #     ax.contourf(xx, yy, rr, color=colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    set_pane_invisible(ax)
    set_grid_invisible(ax)

    return fig, ax, surf


def set_pane_invisible(ax):
    """
    Makes the background pane invisible

    Parameters
    ----------
    ax: axes.Axes instance

    Returns
    -------
    None
    """
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


def set_grid_invisible(ax):
    """
    Makes the background grid invisible
    Parameters
    ----------
    ax: axes.Axes instance

    Returns
    -------
    None
    """
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)




## Miscellanies
def show():
    plt.show()


## Lines
def axhline(ax, y, x0=None, x1=None, color='black', linestyle='--', linewidth=1, zorder=0, **kwargs):
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
        x1 = xmax
    elif x1 is not None:
        xmin, xmax = ax.get_xlim()
        x0 = xmin
    if x0 is not None or x1 is not None:
        if ax.get_xscale()=='linear':
            xmin_frac, xmax_frac = (x0-float(xmin)) / (float(xmax)-float(xmin)), (x1-float(xmin)) / (float(xmax)-float(xmin))
        if ax.get_xscale()=='log':
            xmin, xmax = np.log(xmin), np.log(xmax)
            x0, x1 = np.log(x0), np.log(x1)
            xmin_frac, xmax_frac = (x0 - xmin) / (xmax - xmin), (x1 - xmin) / (xmax - xmin)
    else:
        xmin_frac, xmax_frac= 0, 1
    handle = ax.axhline(y, xmin_frac, xmax_frac, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, **kwargs)
    return handle
def axvline(ax, x, y0=None, y1=None,  color='black', linestyle='--', linewidth=1, zorder=0, **kwargs):
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
    handle = ax.axvline(x, ymin_frac, ymax_frac, color=color, linestyle=linestyle, linewidth=linewidth,  zorder=zorder,
               **kwargs)
    return handle
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
            ax=None, fig=None, fignum=1, subplot=111, figsize=None,
            xlabel='x (mm)', ylabel='y (mm)', zlabel='z (mm)',
            **kwargs):

    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize, projection='3d')
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()


    if isinstance(x, (int, float)):
        arrow_obj = Arrow3D([x, x+dx], [y, y+dy],
                            [z, z+dz], mutation_scale=mutation_scale,
                            lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
        ax.add_artist(arrow_obj)
    elif isinstance(x, (np.ndarray, list)):
        if not len(x) == len(y) == len(z) == len(dx) == len(dy) == len(dz):
            raise ValueError('graph.arrow3D: x, y, z, dx, dy, dz must have the same length')
        for i, x_ in enumerate(x):
            arrow_obj = Arrow3D([x[i], x[i] + dx[i]], [y[i], y[i] + dy[i]],
                                [z[i], z[i] + dz[i]], mutation_scale=mutation_scale,
                                lw=lw, arrowstyle=arrowstyle, color=color, **kwargs)
            ax.add_artist(arrow_obj)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
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
    try:
        ax.arrow(x, y, dx, dy, **kwargs)
    except:
        n = len(x)
        for i in range(n):
            ax.arrow(x[i], y[i], dx[i], dy[i], **kwargs)
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
            # self.format = '$%s$' % mpl.ticker._mathdefault(self.format) # matplotlib < 3.1
            self.format = '$%s$' % self.format


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
                 tight_layout=True, ticklabelsize=None, aspect='equal', ntick=5, tickinc=None,
                 size='5%', pad=0.15, caxAspect=None, fformat="%03.1f", labelpad=1, **kwargs):

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

    reset_sfmt(fformat=fformat)

    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    if caxAspect is not None:
        cax.set_aspect(caxAspect)
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
            cb.set_label(label, labelpad=labelpad)
        else:
            cb.set_label(label, labelpad=labelpad, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)
        cb.ax.yaxis.get_offset_text().set_fontsize(ticklabelsize)
    # ALTERNATIVELY
    # global __fontsize__
    # cb.ax.tick_params(axis='both', which='major', labelsize=__fontsize__, length=5, width=0.2)
    # cb.ax.yaxis.get_offset_text().set_fontsize(__fontsize__) # For scientific format

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb


def add_discrete_colorbar(ax, colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, useMiddle4Ticks=False,**kwargs):
    fig = ax.get_figure()
    if vmax is None:
        vmax = len(colors)
    tick_spacing = (vmax - vmin) / float(len(colors))
    if not useMiddle4Ticks:
        vmin, vmax = vmin -  tick_spacing / 2., vmax -  tick_spacing / 2.
    ticks = np.linspace(vmin, vmax, len(colors) + 1) + tick_spacing / 2.  # tick positions

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



def add_colorbar_alone(ax, values, cmap=cmap, label=None, fontsize=None, option='normal', fformat=None,
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, location='right', color='k',
                 size='5%', pad=0.15, **kwargs):
    """
    Add a colorbar to a figure without a mappable
    ... It creates a dummy mappable with given values

    ... LOCATION OF CAX
    fig, ax = graph.set_fig(1, 111)
    w, pad, size = 0.1, 0.05, 0.05
    graph.add_colorbar_alone(ax, [0, 1], pad=float2pc(pad), size=float2pc(size), tight_layout=False)
    graph.add_subplot_axes(ax, [1-w-(1-1/(1.+pad+size)), 0.8, w, 0.2])


    Parameters
    ----------
    ax: Axes instance
    values: 1D array-like- min and max values of values are found from this array
    cmap: str, cmap instance
    label: str, label of the color bar
    fontsize: float, fontsize of the label
    option: str, choose from 'normal' and 'scientific'
    ... if 'scientific', the color bar is shown in a scientific format like 1x10^exponent
    fformat: str, default: None equivalent to "%03.1f"
    tight_layout: bool, if True, fig.tight_layout() is called.
    ticklabelsize: float
    ticklabel: 1d array-like
    aspect:
    ...  Adding a color bar may distort the aspect ratio. Fix it.
    if aspect == 'equal':
        ax.set_aspect('equal')
    location
    color
    kwargs

    Returns
    -------
    cb:
    """
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
    ## divider.append_axes(location, size=size, pad=pad) creates an Axes
    ## s.t. the size of the cax becomes 'size' (e.g.'5%') of the ax.
    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)


    if option == 'scientific':
        if fformat is not None:
            global sfmt
            sfmt.fformat = fformat

        cb = fig.colorbar(sm, cax=cax, format=sfmt, **kwargs)
        reset_sfmt()
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


def colorbar(fignum=None, label=None, fontsize=__fontsize__):
    """
    Use is DEPRICATED. This method is replaced by add_colorbar(mappable)
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

def create_colorbar(values, cmap='viridis', figsize=None, orientation='vertical', label='qty (mm)', fontsize=11,
                    labelpad=0, ticks=None, **kwargs):
    """
    Creates a horizontal/vertical colorbar for reference using pylab.colorbar()

    Parameters
    ----------
    values: 1d array-like, used to specify the min and max of the colorbar
    cmap: cmap instance or str, default: 'viridis'
    figsize: tuple, figure size in inches, default: None
    orientation: str, 'horizontal' or 'vertical'
    label: str, label of the color bar
    fontsize: fontsize for the label and the ticklabel
    labelpad: float, padding for the label
    ticks: 1d array, tick locations

    Returns
    -------

    """
    values = np.array(values)
    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=0)
    #     values = np.array([[-1.5, 1]])
    if orientation == 'horizontal' and figsize is None:
        figsize = (7.54 * 0.5, 1)
    elif orientation == 'vertical' and figsize is None:
        figsize = (1, 7.54 * 0.5)
    fig = pl.figure(figsize=figsize)
    img = pl.imshow(values, cmap=cmap)
    ax = pl.gca()
    ax.set_visible(False)

    if orientation == 'horizontal':
        cax = pl.axes([0.1, 0.8, 0.8, 0.1])
    else:
        cax = pl.axes([0.8, 0.1, 0.1, 0.8])
    cb = pl.colorbar(orientation=orientation, cax=cax, **kwargs)
    cb.set_label(label=label, fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(np.arange(-1.5, 1.5, 0.5))
    cb.ax.tick_params(labelsize=fontsize)
    fig.tight_layout()
    return fig, ax, cb

def dummy_scalarMappable(values, cmap):
    """
    Returns a dummy scalarMappable that can be used to make a stand-alone color bar
    e.g.
        sm = dummy_scalarMappable([0, 100], 'viridis')
        fig = plt.figure(1)
        fig.colorbar(sm, pad=0.1)
    Parameters
    ----------
    values: list, array, this is used to specify the range of the color bar
    cmap: str, cmap object

    Returns
    -------
    sm
    """
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable
    return sm

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
def setaxes(ax, xmin, xmax, ymin, ymax, **kwargs):
    ax.set_xlim(xmin, xmax, **kwargs)
    ax.set_ylim(ymin, ymax, **kwargs)
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

def force2showLogMinorTicks(ax, subs='all', numticks=9, axis='both'):
    """
    Force to show the minor ticks in the logarithmic axes
    ... the minor ticks could be suppressed due to the limited space
    Parameters
    ----------
    ax: Axes instance
    subs: str, list, or np.array, 'all' is equivalent to np.arange(1, 10)
    numticks: int, make this integer high to show the minor ticks

    Returns
    -------

    """
    if axis in ['x', 'both']:
        ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position
    if axis in ['y', 'both']:
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position

def force2showLogMajorTicks(ax, subs=[1.], numticks=9, axis='both'):
    """
    Force to show the minor ticks in the logarithmic axes
    ... the minor ticks could be suppressed due to the limited space
    Parameters
    ----------
    ax: Axes instance
    subs: str, list, or np.array, 'all' is equivalent to np.arange(1, 10)
    ... to
    numticks: int, make this integer high to show the minor ticks

    Returns
    -------

    """
    if axis in ['x', 'both']:
        ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position
    if axis in ['y', 'both']:
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position


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

    if type(option) in [tuple or list or np.ndarray]:
        x, y = option[0], option[1]
        option = None

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
                        fontsize=__fontsize__, set_base_label_one=False, beta=20, zorder=100,
                        x_base=None, y_base=None, x_height=None, y_height=None,
                        **kwargs):
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
    if xmin < 0:
        xmin = 1e-16
    if ymin < 0:
        ymin = 1e-16
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
    if any([item is None for item in [x_base, y_base, x_height, y_height]]):
        if exponent >= 0 and not flip:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y - (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_w + exp_x + 0.4 * (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
        elif exponent < 0 and not flip:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y + 0.3 * (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_w + exp_x + 0.4 * (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
        elif exponent >= 0 and flip:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.4), 10 ** (exp_y + exp_h + 0.3 * (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_x - (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
        else:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y + exp_h - (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_x - 0.6 * (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.6)

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

def get_first_n_default_colors(n):
    return __def_colors__[:n]


def apply_custom_cyclers(ax, color=['r', 'b', 'g', 'y'], linestyle=['-', '-', '-', '-'], linewidth=[3, 3, 3, 3],
                         marker=['o', 'o', 'o', 'o'], s=[0,0,0,0], **kwargs):

    """
    This is a simple example to apply a custom cyclers for particular plots.
    ... This simply updates the rcParams so one must call this function BEFORE ceration of the plots.
    ... e.g.
            fig, ax = set_fig(1, 111)
            apply_custom_cyclers(ax, color=['r', 'b', 'g', 'y'])
            ax.plot(x1, y1)
            ax.plot(x2, y2)
            ...

    Parameters
    ----------
    ax: mpl.axes.Axes instance
    color: list of strings, color
    linewidths: list of float values, linewidth
    linestyles: list of strings, linestyle
    marker: list of strings, marker
    s: list of float values, marker size

    Returns
    -------
    None

    """
    custom_cycler = cycler(color=color) + cycler(linestyle=linestyle) + cycler(lw=linewidth) + cycler(marker=marker) + cycler(markersize=s)
    ax.set_prop_cycle(custom_cycler)


def create_cmap_using_values(colors=None, color1='greenyellow', color2='darkgreen', color3=None, n=100):
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
        colors = get_color_list_gradient(color1=color1, color2=color2, color3=color3, n=n)
    cmap_name = 'new_cmap'
    newcmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    return newcmap


def create_cmap_from_colors(colors_list, name='newmap'):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors_list)

def get_colors_and_cmap_using_values(values, cmap=None, color1='greenyellow', color2='darkgreen', color3=None,
                                     vmin=None, vmax=None, n=100):
    """
    Returns colors (list), cmap instance, mpl.colors.Normalize instance assigned by the
    ...

    Parameters
    ----------
    values: 1d array-like,
    cmap: str or  matplotlib.colors.Colormap instance
    color1:
    color2
    vmin
    vmax
    n

    Returns
    -------
    colors, cmap, norm

    """
    values = np.asarray(values)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    if cmap is None:
        cmap = create_cmap_using_values(color1=color1, color2=color2, color3=color3, n=n)
    else:
        cmap = plt.get_cmap(cmap, n)
    # normalize
    # vmin, vmax = np.nanmin(values), np.nanmax(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(values))
    return colors, cmap, norm

def get_color_list_gradient(color1='greenyellow', color2='darkgreen', color3=None, n=100, return_cmap=False):
    """
    Returns a list of colors in RGB between color1 and color2
    Input (color1 and color2) can be RGB or color names set by matplotlib
    ... color1-color2-color3

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
    if return_cmap:
        cmap = create_cmap_using_values(colors=color_list, n=n)
        return color_list, cmap
    else:
        return color_list

def get_color_from_cmap(cmap='viridis', n=10, lut=None, reverse=False):
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
    if reverse:
        cmap = cmap.reversed()
    colors = cmap(np.linspace(0, 1, n, endpoint=True))
    return colors

def create_weight_shifted_cmap(cmapname, ratio=0.75, vmin=None, vmax=None, vcenter=None, n=500):
    """
    Creates a cmap instance of a weight-shifted colormap

    Parameters
    ----------
    cmapname: str
    ratio
    vmin
    vmax
    vcenter
    n

    Returns
    -------

    """
    if vmin is not None and vmax is not None and vcenter is not None:
        if vmax <= vmin:
            raise ValueError('... vmax must be greater than vmin')
        if vcenter <= vmin or vcenter >= vmax:
            raise ValueError('vcenter must take a value between vmin and vmax')
        vrange = vmax - vmin
        ratio = (vcenter - vmin) / vrange

    cmap_ = mpl.cm.get_cmap(cmapname, n)
    colorNeg = cmap_(np.linspace(0, 0.5, int(n * ratio)))
    colorPos = cmap_(np.linspace(0.5, 1, n - int(n * ratio)))
    newcolors = np.concatenate((colorNeg, colorPos), axis=0)
    newcmap = mpl.colors.ListedColormap(newcolors, name='shifted_' + cmapname)  # custom cmap
    return newcmap


def choose_colors(**kwargs):
    """
    Equivalent of sns.choose_cubehelix_palette()

    Example: COLOR CURVES BASED ON A QUANTITY 'Z'
        # What is Z?
        z = [0, 0.25, 0.5, 0.75, 1]
        # Choose colors
        colors = graph.choose_colors()
        # Set the colors as a default color cycle
        set_default_color_cycle(n=len(z), colors=colors)
        # Plot your data...
        plot(x1, y1) # color1
        plot(x2, y2) # color2
        ...
        # Add a colorbar (On the same figure)
        add_colorbar_alone(plt.gca(), z, colors=colors) # plot a stand-alone colorbar in the figure
        # Add a colorbar (On the different figure)
        plot_colorbar(z, colors=colors, fignum=2) # plot a stand-alone colorbar on a new figure

    Parameters
    ----------
    kwargs

    Returns
    -------
    colors
    """
    colors = sns.choose_cubehelix_palette(**kwargs)
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
    h = hex.strip('#')
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

def set_default_color_cycle(name='tab10', n=10, colors=None, reverse=False):
    """
    Sets a color cycle for plotting

    sns_palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'] # sns_palettes
    matplotlab cmap names: 'tab10' (default cmap of mpl), 'tab20', 'Set1', 'Set2' etc.
    (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    ... One may specify the color cycles using the existing color maps (seaborn and matplotlib presets)
        or a list of colors specified by a user.
    ... For the presets, pass a name of the colormap like "tab10" (mpl default), "muted" (seaborn defualt)
    ... For a more customized color cycle, pass a list of colors to 'colors'.

    Parameters
    ----------
    name: str, name of the cmap
    n: int, number of colors
    colors: list, a list of colors like ['r', 'b', 'g', 'magenta']

    Returns
    -------
    None
    """
    if colors is None:
        colors = sns.color_palette(name, n_colors=n)
        if reverse:
            colors.reverse()
    sns.set_palette(colors)
    return colors

def set_color_cycle(cmapname='tab10', ax=None, n=10, colors=None):
    """
    Sets a color cycle of a particular Axes instance

    sns_palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'] # sns_palettes
    matplotlab cmap names: 'tab10' (default cmap of mpl), 'tab20', 'Set1', 'Set2' etc.
    (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    ... One may specify the color cycles using the existing color maps (seaborn and matplotlib presets)
        or a list of colors specified by a user.
    ... For the presets, pass a name of the colormap like "tab10" (mpl default), "muted" (seaborn defualt)
    ... For a more customized color cycle, pass a list of colors to 'colors'.

    Parameters
    ----------
    cmapname: str, name of the cmap like 'viridis', 'jet', etc.
    n: int, number of colors
    colors: list, a list of colors like ['r', 'b', 'g', 'magenta']

    Returns
    -------
    None
    """
    if colors is None:
        colors = sns.color_palette(cmapname, n_colors=n)
    if ax is None:
        sns.set_palette(colors)
    else:
        ax.set_prop_cycle(color=colors)

def set_color_cycle_custom(ax, colors=__def_colors__):
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
    ... pylab.rcParams.update(params)
    Parameters
    ----------
    params: dictionary

    Returns
    -------
    None
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
def add_subplot_axes(ax, rect, axisbg='w', alpha=1, **kwargs):
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
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], **kwargs)
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
    Draws a circle on the axes (ax)

    Parameters
    ----------
    ax: matplotlib axes object
    x: float
    y: float
    r: float
    linewidth: float
    edgecolor:
    facecolor
    fill
    kwargs

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


def draw_box(ax, xx, yy, w_box=351., h_box=351., xoffset=0, yoffset=0, linewidth=5,
             scalebar=True, sb_length=50., sb_units='$mm$', sb_loc=(0.95, 0.1), sb_txtloc=(0.0, 0.4),
             sb_lw=10, sb_txtcolor='white', fontsize=None,
             facecolor='k', fluidcolor=None,
             bounding_box=True, bb_lw=1, bb_color='w'):
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
        width of the box- used to be set as 325
    h_box: float/int
        height of the box- used to be set as 325
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
    if fluidcolor is not None:
        ax.set_facecolor(fluidcolor)

    if bounding_box:
        w, h = xmax-xmin, ymax-ymin
        draw_rectangle(ax, xmin, ymin, width=w, height=h, edgecolor=bb_color, linewidth=bb_lw)

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
        if fontsize is None or fontsize>0:
            ax.text(x_sb_txt, y_sb_txt, '%d%s' % (sb_length, sb_units), color=sb_txtcolor, fontsize=fontsize)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw_cuboid(ax, xx, yy, zz, color='c', lw=2, **kwargs):
    """Draws a cuboid (projection='3d')"""

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

def draw_sphere(ax, xc, yc, zc, r, color='r', **kwargs):
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = xc + r * np.outer(np.cos(u), np.sin(v))
    y = yc + r * np.outer(np.sin(u), np.sin(v))
    z = zc + r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    surf = ax.plot_surface(x, y, z, color=color, **kwargs)
    set_axes_equal(ax)

def draw_sphere_wireframe(ax, xc, yc, zc, r, color='r', lw=1, **kwargs):
    """
    Draws a sphere using a wireframe
    Parameters
    ----------
    ax
    xc
    yc
    zc
    r
    color
    lw
    kwargs

    Returns
    -------

    """
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u) * np.sin(v) + xc
    y = r * np.sin(u) * np.sin(v) + yc
    z = r * np.cos(v) + zc
    ax.plot_wireframe(x, y, z, color=color, lw=lw, **kwargs)
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
    reasonable_rate_of_change = np.abs(fractional_dydx) < thd # len(reasonable_rate_of_change) is not necessarily equal to len(keep)
    reasonable_rate_of_change = np.roll(reasonable_rate_of_change, 1) # shift the resulting array (the convention of np.gradient)
    keep[keep] = reasonable_rate_of_change
    return keep

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

def get_plot_data_from_fig(*args, **kwargs):
    """Depricated. Use get_data_from_fig_plot"""
    get_data_from_fig_plot(*args, **kwargs)

def get_data_from_fig_plot(fig, axis_number=0):
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

def get_data_from_fig_scatter(ax):
    """Retrieves x, y from a scatter plot"""
    ndata = len(ax.collections)
    xs, ys = [], []
    for item in ax.collections:
        xs.append(item._offsets.data[:, 0])
        ys.append(item._offsets.data[:, 1])
    return xs, ys

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
def smooth1d(x, window_len=11, window='hanning', log=False):
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

def add_secondary_xaxis(ax, functions=None, loc='top', label='', log=False, **kwargs):
    """
    Adds a secondary x-axis at the top
    ... Must pass a pair of mapping functions between a current x and a new x

    e.g.
        def deg2rad(x):
            return x * np.pi / 180
        def rad2deg(x):
            return x * 180 / np.pi
        add_secondary_xaxis(ax, functions=(deg2rad, rad2deg))

    Parameters
    ----------
    ax
    functions

    Returns
    -------
    secax

    """
    if functions is None:
        print('add_secondary_xaxis: supply a mapping function (Current X to New X) and its inverse function')
        print('... e.g. (deg2rad, rad2deg)')

        def f1(x):
            return 2 * x

        def f2(x):
            return x / 2

        functions = (f1, f2)
    secax = ax.secondary_xaxis(location=loc, functions=functions)
    secax.set_xlabel(label, **kwargs)
    if log:
        secax.set_xscale("log")
    return secax

def add_secondary_yaxis(ax, functions=None, loc='right', label='', log=False, **kwargs):
    """
    Adds a secondary yaxis at the top
    ... Must pass a pair of mapping functions between a current x and a new x

    e.g.
        def deg2rad(y):
            return y * np.pi / 180
        def rad2deg(y):
            return y * 180 / np.pi
        add_secondary_yaxis(ax, functions=(deg2rad, rad2deg))

    Parameters
    ----------
    ax
    functions

    Returns
    -------
    secax

    """
    if functions is None:
        print('add_secondary_xaxis: supply a mapping function (Current X to New X) and its inverse function')
        print('... e.g. (deg2rad, rad2deg)')

        def f1(x):
            return 2 * x

        def f2(x):
            return x / 2

        functions = (f1, f2)
    secax = ax.secondary_yaxis(location=loc, functions=functions)
    secax.set_ylabel(label, **kwargs)
    if log:
        secax.set_yscale("log")
    return secax


def use_symmetric_ylim(ax):
    bottom, top = ax.get_ylim()
    if bottom * top < 0:
        bottom, top = -np.max([-bottom, top]), np.max([-bottom, top])
        ax.set_ylim(bottom=bottom, top=top)
    return ax


def get_binned_stats(arg, var, n_bins=100, mode='linear', bin_center=True, return_std=False):
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
        If True, it returns the STD of the statistics instead of the error = STD / np.sqrt(N-1)
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
    if return_std:
        return arg_bins, var_mean, var_err
    else:
        return arg_bins, var_mean, var_err / np.sqrt(counts-1)


def make_ax_symmetric(ax, axis='y'):
    """Makes the plot symmetric about x- or y-axis"""
    """
    Makes the plot symmetric about the x- or y-axis
    
    Parameters
    ----------
    ax: axes.Axes instance
    axis: str, Choose from 'x', 'y', 'both'
    
    Returns
    -------
    None

    """
    if axis in ['y', 'both']:
        ymin, ymax = ax.get_ylim()
        yabs = max(-ymin, ymax)
        ax.set_ylim(-yabs, yabs)
    if axis in ['x', 'both']:
        xmin, xmax = ax.get_xlim()
        xabs = max(-xmin, xmax)
        ax.set_xlim(-xabs, xabs)


def make_ticks_scientific(ax, axis='both', **kwargs):
    """
    Make tick labels display in a scientific format

    Some other useful lines about tick formats
        ax.set_xticks(np.arange(0, 1.1e-3, 0.5e-3))
        ax.set_yticks(np.arange(0, 1.1e-3, 0.25e-3))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.offsetText.set_fontsize(20)
        ax.yaxis.offsetText.set_fontsize(20)

    Parameters
    ----------
    ax: axes.Axes instance

    Returns
    -------

    """
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis=axis, **kwargs)

def color_axis(ax, locs=['bottom', 'left', 'right'], colors=['k', 'C0', 'C1'],
               xlabel_color=None, ylabel_color=None,
               xtick_color=None, ytick_color=None):
    """
    Colors the axes (axis, ticks, and a label)

    Parameters
    ----------
    ax: axes.Axes instance
    locs: list of strings, locations of the axes. choose from 'bottom', 'left', 'right', 'top'
    colors: list of strings, colors of the axes. e.g. ['k', 'C0', 'C1']
    xlabel_color: str, color of xlabel. If None, the same colors as "colors" are used.
    ylabel_color: str, color of ylabel. If None, the same colors as "colors" are used.
    xtick_color: str, color of xtick. If None, the same colors as "colors" are used.
    ytick_color: str, color of ytick. If None, the same colors as "colors" are used.

    Returns
    -------

    """
    for loc, color in zip(locs, colors):
        ax.spines[loc].set_color(color)
        if loc in ['top', 'bottom'] and xlabel_color is None:
            xlabel_color = color
            if xlabel_color is None: xlabel_color = 'k'
        elif loc in ['right', 'left'] and ylabel_color is None:
            ylabel_color = color

    if xlabel_color is None: xlabel_color = 'k'
    if ylabel_color is None: ylabel_color = 'k'

    # match tick colors with the label colors
    if xtick_color is None: xtick_color = xlabel_color
    if ytick_color is None: ytick_color = ylabel_color

    ax.xaxis.label.set_color(xlabel_color)
    ax.tick_params(axis='x', colors=xtick_color)
    ax.xaxis.label.set_color(xlabel_color)
    ax.tick_params(axis='y', colors=ytick_color)




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

def pc2float(s):
    """
    Converts a percentage expression (str) to float
    e.g. pc2float(5.2%) returns 0.0052
    Parameters
    ----------
    s: str, e.g. "5.2%"

    Returns
    -------
    a floating number  (e.g. 0.0052)
    """
    return float(s.strip('%'))/100.

def float2pc(x):
    """
    Converts a float into a percentage expression
    Parameters
    ----------
    x

    Returns
    -------
    a string in float (e.g. 0.0052)
    """
    return "{0}%".format(x * 100.)



def simple_legend(ax, facecolor='white', **kwargs):
    "Removes the errorbars from the legend"
    from matplotlib import container
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    lg = ax.legend(handles, labels, **kwargs)
    frame = lg.get_frame()
    frame.set_color(facecolor)

# adjuster/reminders
## colorbar stuff
def adjust_colorbar(cb,
                    fontsize=__fontsize__,
                    label=None, labelpad=1,
                    tick_fontsize=__fontsize__,
                    ticks=None, ):
    """A helper to modify basic features of a matplotlib Colorbar object """
    if label is None:
        label = cb.ax.get_ylabel()
    cb.set_label(label, fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    cb.ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)


def plot_colorbar(values, cmap='viridis', colors=None, ncolors=100,
                  fignum=1, figsize=None,
                  orientation='vertical', label=None, labelpad=5,
                  fontsize=__fontsize__, option='normal', fformat=None,
                  ticks=None, tick_params=None, **kwargs):
    """
    Plots a stand-alone colorbar

    Parameters
    ----------
    values: 1d array-like, these values are used to create a colormap
    cmap: str, cmap object
    colors: list of colors, if given, it overwrites 'cmap'.
        ... For custom colors/colormaps, the functions below could be handy.
        ...... colors = graph.choose_colors() # sns.choose_cubehelix_palette()
        ...... colors = graph.get_color_list_gradient(color1='red', color2='blue') # linearly segmented colors
    ncolors
    fignum
    figsize
    orientation
    label
    labelpad
    fontsize
    option: str, if 'scientific', it uses a scientific format for the ticks
    fformat
    ticks
    tick_params
    kwargs

    Returns
    -------
    fig, cax, cb: Figure instance, axes.Axes instance, Colorbar instance
    """
    global sfmt
    if figsize is None:
        if orientation == 'horizontal':
            figsize =(7.54 * 0.5, 1)
        else:
            figsize = (1, 7.54 * 0.5)

    if orientation == 'horizontal':
        cax_spec = [0.1, 0.8, 0.8, 0.1]
    else:
        cax_spec = [0.1, 0.1, 0.1, 0.8]

    if colors is not None:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('cutom_cmap', colors, N=ncolors)

    fig = pl.figure(fignum, figsize=figsize)
    img = pl.imshow(np.array([values]), cmap=cmap)
    pl.gca().set_visible(False)

    cax = pl.axes(cax_spec)

    if option == 'scientific':
        if fformat is not None:
            sfmt.fformat = fformat
        fmt = sfmt
    else:
        if fformat is not None:
            fmt = fformat
        else:
            fmt = None
    cb = pl.colorbar(orientation=orientation, cax=cax, format=fmt, **kwargs)
    cb.set_label(label=label,
                 fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(ticks)

    if tick_params is None:
        tick_params = {'labelsize': fontsize}
    cb.ax.tick_params(**tick_params)
    fig.tight_layout()
    return fig, cax, cb



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

    # x, y = copy.deepcopy(x_), copy.deepcopy(y_)
    x, y = np.array(x), np.array(y) # np.array creates a new object unlike np.asarray

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
            y_rs = 10**flog(logx_new)
            return x_new, y_rs
    else:
        x_new = np.linspace(xmin, xmax, n, endpoint=True)
        #         y_rs = scipy.signal.resample(y, n)
        f = interpolate.interp1d(x, y)
        y_rs = f(x_new)

        return x_new, y_rs


def set_fontsize_scientific_text(ax, fontsize):
    """
    Set fontsize for the scientific format

    Parameters
    ----------
    fontsize: int

    Returns
    -------

    """
    ax.yaxis.get_offset_text().set_fontsize(fontsize)