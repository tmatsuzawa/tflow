import numpy as np
from scipy.optimize import curve_fit

###############################################################
# Standard functions
def linear_func(x, a, b):
    """a * x + b"""
    return a * x + b

def quad_func(x, a, b, c):
    """ a*x**2 + b*x + c """
    return a*x**2 + b*x + c
def quad_func2(x, a, b, c):
    """a*(x-b)**2 + c"""
    return a*(x-b)**2 + c

def power_func(x, a, b):
    """a * x ** b"""
    return a * x ** b
def power_func2(x, x0, a, b):
    """a * (x-x0) ** b"""
    return a * (x-x0) ** b


def exp(x, a, b):
    """a * np.exp(-b * x)"""
    return a * np.exp(-b * x)

def exp_func(x, a, b, x0):
    """a * np.exp(-b * (x-x0)) + c"""
    return a * np.exp(-b * (x-x0))

def gaussian(x, a, x0, sigma, b):
    """ a * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.)) + b"""
    return a * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.)) + b

def gaussian_norm(x, x0, sigma):
    """ 1. / (np.sqrt(2 * np.pi * sigma ** 2.)) * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.)) """
    return 1. / (np.sqrt(2 * np.pi * sigma ** 2.)) * np.exp(- (x - x0) ** 2. / (2. * sigma ** 2.))

def double_gaussian(x, a1, x1, sigma1, b1, a2, x2, sigma2, b2):
    """a1 * np.exp(- (x - x1) ** 2. / (2. * sigma1)) + b1 + a2 * np.exp(- (x - x2) ** 2. / (2. * sigma2)) + b2"""
    return a1 * np.exp(- (x - x1) ** 2. / (2. * sigma1)) + b1 + a2 * np.exp(- (x - x2) ** 2. / (2. * sigma2)) + b2


def lorentzian(x, x0, gamma, alpha):
    """alpha * gamma / ((x-x0)**2 + gamma ** 2)"""
    return alpha * gamma / ((x-x0)**2 + gamma ** 2)

def lorentzian_norm(x, x0, gamma):
    """1. / np.pi * gamma / ((x-x0)**2 + gamma ** 2)"""
    return 1. / np.pi * gamma / ((x-x0)**2 + gamma ** 2)


###############################################################

def fit(func, xdata, ydata, **kwargs):
    if func is None:
        popt, pcov = curve_fit(linear_func, xdata, ydata, **kwargs)
    else:
        popt, pcov = curve_fit(func, xdata, ydata, **kwargs)
    return popt, pcov
