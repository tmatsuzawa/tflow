#!/usr/bin/env python

################################################################################

# CINE library
# Includes simple multipage TIFF reader/writer and MJPEG AVI writer
# Author: Dustin Kleckner
# dkleckner@uchicago.edu

################################################################################

from cine import *
from tiff import *
from mjpeg import *
from sparse import *
from plot_to_image import *
import os.path

def open(fn):
    base, ext = os.path.splitext(fn)
    ext = ext.lower()
    if ext == '.cine':
        return Cine(fn)
    elif ext == '.sparse':
        return Sparse(fn)
    elif ext in ('.tiff', '.tif'):
        return Tiff(fn)
    elif ext == '.s4d':
        return Sparse4D(fn)
    else:
        raise ValueError('can only open .tif(f), .sparse, .s4d or .cine files')

import svn_info