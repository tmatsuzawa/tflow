import numpy as np
from scipy.optimize import curve_fit
import library.basics.std_func as std_func

'''
Module for plotting and saving figures

The up-to-date version is on Takumi's github.
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
import fapm.formatarray as fa
import numpy as np
import glob


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
         'ytick.labelsize': __fontsize__}


## Save a figure
def save(path, ext='pdf', close=False, verbose=True, fignum=None, dpi=None, overwrite=True, tight_layout=False, **kwargs):
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
        savepath = directory + os.path.split(path)[1] + '_{n:03d.}'.format(n=ver_no) + ext
        ver_no += 1


    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Save the figure
    plt.savefig(savepath, dpi=dpi, **kwargs)

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

def plot(x, y, fignum=1, figsize=None, label='', color=None, subplot=None, legend=False, ax=None,  **kwargs):
    """
    plot a graph using given x,y
    fignum can be specified
    any kwargs from plot can be passed
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    fig = plt.gcf()

    if len(x) > len(y):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    if color is None:
        ax.plot(x, y, label=label, **kwargs)
    else:
        ax.plot(x, y, color=color, label=label, **kwargs)
    if legend:
        ax.legend()
    return fig, ax


def plot_saddoughi(fignum=1, figsize=None, label='', color=None, subplot=None, legend=False, **kwargs):
    """
    plot universal 1d energy spectrum (Saddoughi, 1992)
    """
    fig, ax = set_fig(fignum, subplot, figsize=figsize)

    x = np.asarray([1.27151, 0.554731, 0.21884, 0.139643, 0.0648844, 0.0198547, 0.00558913, 0.00128828, 0.000676395, 0.000254346])
    y = np.asarray([0.00095661, 0.0581971, 2.84666, 11.283, 59.4552, 381.78, 2695.48, 30341.9, 122983, 728530])

    if color is None:
        plt.plot(x, y, label=label, **kwargs)
    else:
        plt.plot(x, y, color=color, label=label, **kwargs)
    if legend:
        plt.legend()
    return fig, ax


def scatter(x, y, ax=None, fignum=1, figsize=None, marker='o', fillstyle='full', label=None, subplot=None, legend=False, **kwargs):
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
    if fillstyle =='none':
        # Scatter plot with open markers
        facecolors = 'none'
        # ax.scatter(x, y, color=color, label=label, marker=marker, facecolors=facecolors, edgecolors=edgecolors, **kwargs)
        ax.scatter(x, y, label=label, marker=marker, facecolors=facecolors, **kwargs)
    else:
        ax.scatter(x, y, label=label, marker=marker, **kwargs)
    if legend:
        plt.legend()
    return fig, ax


def pdf(data, nbins=10, return_data=False, vmax=None, vmin=None, fignum=1, figsize=None, subplot=None, density=True, analyze=False, ax=None, **kwargs):
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
    fig, ax = plot(bins, hist, fignum=fignum, figsize=figsize, subplot=subplot, ax=ax, **kwargs)

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


def errorbar(x, y, xerr=0, yerr=0, fignum=1, marker='o', fillstyle='full', linestyle='None', label=None, mfc='white', subplot=None, legend=False, figsize=None, **kwargs):
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

    if fillstyle == 'none':
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, mfc=mfc, linestyle=linestyle, label=label, **kwargs)
    else:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, marker=marker, fillstyle=fillstyle, linestyle=linestyle, label=label, **kwargs)
    if legend:
        plt.legend()
    return fig, ax

def errorfill(x, y, yerr, fignum=1, color=None, subplot=None, alpha_fill=0.3, ax=None, label=None,
              legend=False, figsize=None, color_cycle=__color_cycle__, **kwargs):
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


    if color is not None:
        ax.plot(x, y, color=color, label=label, **kwargs)
        ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
    else:
        ax.plot(x, y,label=label, **kwargs)
        ax.fill_between(x, ymax, ymin, alpha=alpha_fill)

    #patch used for legend
    color_patch = mpatches.Patch(color=color, label=label)
    if legend:
        plt.legend(handles=[color_patch])


    return fig, ax, color_patch


## Plot a fit curve
def plot_fit_curve(xdata, ydata, func=None, fignum=1, subplot=111, figsize=None, linestyle='--',
                   xmin=None, xmax=None, add_equation=True, eq_loc='bl', color=None, label='fit',
                   show_r2=False, **kwargs):
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

    xdata = np.array(xdata)
    ydata = np.array(ydata)

    if any(np.isnan(ydata)) or any(np.isnan(xdata)):
        print 'Original data contains np.nans! Delete them for curve fitting'
        condx, condy = np.isnan(xdata), np.isnan(ydata)
        cond = (~condx * ~condy)
        print 'No of deleted data points %d / %d' % (np.sum(~cond), len(xdata))
        if np.sum(~cond) == len(xdata):
            print 'No data points for fitting!'
            raise RuntimeError
        xdata, ydata = xdata[cond], ydata[cond]

    if xmin is None:
        xmin = np.min(xdata)
    if xmax is None:
        xmax = np.max(xdata)

    x_for_plot = np.linspace(xmin, xmax, 1000)
    if func is None or func=='linear':
        print 'Fitting to a linear function...'
        popt, pcov = curve_fit(std_func.linear_func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle)
        else:
            fig, ax = plot(x_for_plot, std_func.linear_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, **kwargs)

        if add_equation:
            text = '$y=ax+b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.linear_func(xdata, *popt)
    elif func=='power':
        print 'Fitting to a power law...'

        popt, pcov = curve_fit(std_func.power_func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, std_func.power_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, linestyle=linestyle, **kwargs)
        else:
            fig, ax = plot(x_for_plot, std_func.power_func(x_for_plot, *popt), fignum=fignum, subplot=subplot,
                           label=label, figsize=figsize, color=color, linestyle=linestyle, **kwargs)

        if add_equation:
            text = '$y=ax^b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
            addtext(ax, text, option=eq_loc)
        y_fit = std_func.power_func(xdata, *popt)
    else:
        popt, pcov = curve_fit(func, xdata, ydata)
        if color is None:
            fig, ax = plot(x_for_plot, func(x_for_plot, *popt), fignum=fignum, subplot=subplot, label=label, figsize=figsize,
                           linestyle=linestyle, **kwargs)
        else:
            fig, ax = plot(x_for_plot, func(x_for_plot, *popt), fignum=fignum, subplot=subplot, label=label, figsize=figsize,
                           color=color, linestyle=linestyle, **kwargs)
        y_fit = func(xdata, *popt)
    #plot(x_for_plot, std_func.power_func(x_for_plot, *popt))

    if show_r2:
        # compute R^2
        # residual sum of squares
        ss_res = np.sum((ydata - y_fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        addtext(ax, '$R^2: %.2f$' % r2, option='bl3')



    return fig, ax, popt, pcov


## 2D plotsFor the plot you showed at group meeting of lambda converging with resolution, can you please make a version with two x axes (one at the top, one below) one pixel spacing, other PIV pixel spacing, and add a special tick on each for the highest resolution point.
# (pcolormesh)
def color_plot(x, y, z, subplot=None, fignum=1, figsize=None, ax=None, vmin=None, vmax=None, log10=False, show=False,
               cbar=False, cmap='magma', aspect='equal', linewidth=0,  **kwargs):
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
        plt.colorbar()

    if aspect == 'equal':
        ax.set_aspect('equal')
    # set edge color to face color
    cc.set_edgecolor('face')

    return fig, ax, cc

#imshow
def imshow(griddata, xmin=0, xmax=1, ymin=0, ymax=1, cbar=True, vmin=0, vmax=0, \
           fignum=1, subplot=111, figsize=__figsize__, interpolation='linear', cmap='bwr'):
    fig, ax = set_fig(fignum, subplot, figsize=figsize)
    if vmin == vmax == 0:
        cax = ax.imshow(griddata, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap)
    else:
        cax = ax.imshow(griddata, extent=(xmin, xmax, ymin, ymax),\
                   interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar:
        cc = fig.colorbar(cax)
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
# Scientific format for Color bar- set format=sfmt to activate it
sfmt=mpl.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))

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
                 tight_layout=True, ticklabelsize=None, aspect='equal', **kwargs):
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


    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.15)
    if option == 'scientific':
        cb = fig.colorbar(mappable, cax=cax, format=sfmt, **kwargs)
    else:
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

def add_colorbar_alone(fig=None, ax=None, ax_loc=[0.05, 0.80, 0.9, 0.15], vmin=0, vmax=1, cmap=cmap, orientation='horizontal',
                       label=None, fontsize=__fontsize__, *kwargs):
    """
    Add a colorbar alone to a canvas.
    Use a specified figure and axis object if given. Otherwise, create one at location "ax_loc"
    Parameters
    ----------
    fig
    ax
    ax_loc
    vmin
    vmax
    cmap
    orientation
    label

    Returns
    -------
    ax: axis object
    cb: colorbarbase object

    """


    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = fig.add_axes(ax_loc)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation=orientation)
    if label is not None:
        cb.set_label(label, fontsize=fontsize)
    return ax, cb



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




##Clear plot
def clf(fignum=None):
    plt.figure(fignum)
    plt.clf()

## Color cycle
def skipcolor(numskip, color_cycle=__color_cycle__):
    """ Skips numskip times in the color_cycle iterator
        Can be used to reset the color_cycle"""
    for i in range(numskip):
        color_cycle.next()
def countcolorcycle(color_cycle = __color_cycle__):
    return sum(1 for color in color_cycle)

def get_default_color_cycle():
    return __color_cycle__

def get_first_n_colors_from_color_cycle(n):
    color_list = []
    for i in range(n):
        color_list.append(next(__color_cycle__))
    return color_list

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
    color_list = zip(r, g, b)
    return color_list


def hex2rgb(hex):
    """
    Converts HEX code to RGB in a numpy array
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

    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS) # dictionary. key: names, values: hex codes
    try:
        hex = colors[cname]
        return hex
    except NameError:
        print cname, ' is not registered as default colors by matplotlib!'
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
    e.g. params = {'legend.fontsize': 'x-large',
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
def add_subplot_axes(ax, rect, axisbg='w'):
    """
    Creates a sub-subplot inside the subplot (ax)
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
    subax = fig.add_axes([x, y, width,height], axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


# sketches
def draw_rectangle(ax, x, y, width, height, angle=0.0, linewidth=1, edgecolor='r', facecolor='none', **kwargs):
    rect = mpatches.Rectangle((x, y), width, height, angle=angle, linewidth=linewidth, edgecolor=edgecolor,
                              facecolor=facecolor, **kwargs)
    ax.add_patch(rect)
    return rect

