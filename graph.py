import numpy as np
from scipy.optimize import curve_fit
# import library.basics.std_func as std_func

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
import itertools
from scipy import stats
import numpy as np
import glob
from fractions import Fraction
from math import modf
import pickle
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
        plt.close()

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
        ax = fig.add_subplot(111)
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

def plot(x, y, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False, fig=None, ax=None, maskon=False, thd=1, **kwargs):
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

    if x is None:
        x = np.range(len(y))
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

    if color is None:
        ax.plot(x[mask], y[mask], label=label, **kwargs)
    else:
        ax.plot(x[mask], y[mask], color=color, label=label, **kwargs)



    if legend:
        ax.legend()
    return fig, ax


def plot_saddoughi(fignum=1, fig=None, ax=None, figsize=None, label='', color='k', alpha=0.6, subplot=None, legend=False, **kwargs):
    """
    plot universal 1d energy spectrum (Saddoughi, 1992)
    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif ax is not None and fig is None:
        fig = plt.gcf()
    elif fig is not None and ax is None:
        ax = plt.gca()

    x = np.asarray([1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    y = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])

    ax.plot(x, y, color=color, label=label, alpha=alpha,**kwargs)
    if legend:
        ax.legend()
    tologlog(ax)
    labelaxes(ax, '$\kappa \eta$', '$E_{11} / (\epsilon\\nu^5)^{1/4}$')
    return fig, ax


def scatter(x, y, ax=None, fignum=1, figsize=None, marker='o', fillstyle='full', label=None, subplot=None, legend=False,
            maskon=False, thd=1,
            **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    Use the homemade function refresh() to draw and plot the figure, no matter the way python is called (terminal, script, notebook)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = ax.get_figure()
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


def errorbar(x, y, xerr=0, yerr=0, fignum=1, marker='o', fillstyle='full', linestyle='None', label=None, mfc='white',
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
    if not (isinstance(yerr, int) or isinstance(yerr, float)):
        yerr = np.array(yerr)
    if maskon:
        mask = get_mask4erroneous_pts(x, y, thd=thd)
    else:
        mask = [True] * len(x)
    if fillstyle == 'none':
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, mfc=mfc, linestyle=linestyle, label=label, **kwargs)
    else:
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, fillstyle=fillstyle, linestyle=linestyle, label=label, **kwargs)
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
                   show_r2=False, return_r2=False, **kwargs):
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
        xmin = np.min(xdata)
    if xmax is None:
        xmax = np.max(xdata)

    x_for_plot = np.linspace(xmin, xmax, 1000)
    if func is None or func=='linear':
        print('Fitting to a linear function...')
        popt, pcov = curve_fit(std_func.linear_func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle, ax=ax)
        else:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, ax=ax, **kwargs)

        if add_equation:
            text = '$y=ax+b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.linear_func(xdata, *popt)
    elif func=='power':
        print('Fitting to a power law...')

        popt, pcov = curve_fit(std_func.power_func, xdata, ydata)
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
        popt, pcov = curve_fit(func, xdata, ydata)
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
               cbar=True, cmap='magma', aspect='equal', option='scientific', ntick=5, tickinc=None, **kwargs):
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
    if aspect == 'equal':
        ax.set_aspect('equal')
    # set edge color to face color
    cc.set_edgecolor('face')

    return fig, ax, cc

#imshow
def imshow(griddata, xmin=0, xmax=1, ymin=0, ymax=1, cbar=True, vmin=0, vmax=0, \
           fignum=1, subplot=111, figsize=__figsize__, interpolation='nearest', cmap='bwr', scale=1.0):
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
def reset_sfmt():
    global sfmt
    # Scientific format for Color bar- set format=sfmt to activate it
    sfmt = mpl.ticker.ScalarFormatter(useMathText=True)
    # Scientific notation is used for data < 10^-n or data >= 10^m, where n and m are the power limits set using set_powerlimits((n,m))
    sfmt.set_scientific(True)
    sfmt.set_powerlimits((0, 0))  # (n,m): 10^m <= data < 10 ^ -n]
    # sfmt.format = '$\mathdefault{%1.1f}$'
reset_sfmt()

def get_sfmt():
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
            print(np.log10((zmax - zmin) / n), exp)
            print((zmax - zmin) / n, dz)
            print(ticks)

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

    if not label is None:
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
def title(ax, title, subplot=111, **kwargs):
    ax.set_title(title, **kwargs)

def suptitle(title, fignum=None, **kwargs):
    """
    Add a centered title to the figure.
    If fignum is given, it adds a title, then it reselects the figure which selected before this method was called.
    ... this is because figure class does not have a suptitle method.
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


    plt.suptitle(title, **kwargs)




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

def get_colors_and_cmap_using_values(values, cmap=None, color1='greenyellow', color2='darkgreen', vmin=None, vmax=None, n=100):
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


def get_color_list_gradient(color1='greenyellow', color2='darkgreen', n=10):
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

    """
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
    return color_list


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
    subax = fig.add_axes([x, y, width,height])
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
    if np.nanmean(yy) > 0:
        xc, yc = (xmax - xmin) / 2., (ymax - ymin) / 2.
    else:
        xc, yc = (xmax - xmin) / 2., -(ymax - ymin) / 2.
    x0, y0 = xc - w_box / 2. + xoffset, yc - h_box / 2. + yoffset
    draw_rectangle(ax, x0, y0, w_box, h_box, linewidth=linewidth, facecolor=facecolor, zorder=0)
    ax.set_facecolor(fluidcolor)

    if scalebar:
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
    ... e.g. x[mask], y[mask] hide the jumps which appear to be false to human eyes
    ... Uses P = dy/dx / y to determine whether data points appear to be false
        If P is high, we'd expect a jump. thd is a threashold of P.
    Parameters
    ----------
    x: 1d array
    y: 1d array
    thd: float, threshold on P (fractional dy/dx)

    Returns
    -------
    mask: 1d bool array

    """
    fractional_dydx =  np.gradient(y, x) / y
    mask = np.abs(fractional_dydx) < thd
    mask = np.roll(mask, 1) # due to the convention of np.gradient
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
        ylist.apppend(y)
    return xlist, ylist
