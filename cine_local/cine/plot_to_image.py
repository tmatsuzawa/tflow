#!/usr/bin/env python
import numpy as N
import pylab
from matplotlib.backends.backend_agg import FigureCanvasAgg
try:
    from PIL import Image
except:
    import Image
import os

def plot2array(w=None, h=None, canvas=None, fig=None, fast=False):
    if fig is None: fig = pylab.gcf()
    if canvas is None: canvas = pylab.get_current_fig_manager().canvas
    
    if not fast:
        win, hin = fig.get_size_inches()
        if h: #If height is specified, change DPI to accomidate
            dpi = float(h) / hin
            fig.set_dpi(dpi)
            if w: #If width is also specified, change image width in inches
                fig.set_size_inches(float(w)/dpi, hin)
        elif w: #If only width is specified, again change DPI
            fig.set_dpi(float(w) / win)
        
        canvas = canvas.switch_backends(FigureCanvasAgg) #ensure backend is one we can get string from

    canvas.draw() #Force it to draw
    w, h = canvas.get_width_height()
    return N.fromstring(canvas.tostring_rgb(), 'u1').reshape((h, w, 3)) #convert plot canvas into an array

def plot2image(*args, **kwargs):
    return Image.fromarray(plot2array(*args, **kwargs))
    
def ensure_rgb(a):
    a = N.asarray(a)
    
    if len(a.shape) == 3:
        if a.shape[2] == 1: a = a[..., 0]
        elif a.shape[2] == 3: return a
        else: raise ValueError('I can only convert images with shape (x, y) or (x, y, 1) or (x, y, 3)')
    
    if len(a.shape) == 2:
        return N.transpose(N.array([a, a, a]), (1, 2, 0))
    else: raise ValueError('I can only convert images with shape (x, y) or (x, y, 1) or (x, y, 3)')

def plot_next_to_image(img, aspect=1, fast=False):
    img = ensure_rgb(img)
    h, w = img.shape[:2]

    ph = h
    if aspect: pw = int(round(ph * aspect))
    else: pw = w

    return N.hstack([img, plot2array(w=pw, h=ph, fast=False)])

if __name__ == '__main__':
    x = N.linspace(0, 10, 100)
    y = N.sin(x)
    pylab.plot(x, y)
    pylab.xlabel('x')
    pylab.ylabel('y = sin(x)')
    a = plot2array(500, 500)
    pylab.close()
    
    for n in range(5):
        pylab.imshow(a)
        a = plot2array(500, 500)
        pylab.close()
        
    count, bins = N.histogram(a, N.arange(257))
    pylab.semilogy(bins[:-1], count)
    pylab.xticks([0, 64, 128, 192, 255])
    pylab.xlim([0, 255])
#    pylab.show()

    a2 = plot_next_to_image(a)

    Image.fromarray(a2).save(os.path.expanduser('~/plot_test.png'))
    
