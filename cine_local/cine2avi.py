#!/usr/bin/env python
import cine
from numpy import *
import os, sys
import argparse
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Convert CINE file(s) to an AVI.  Also works on TIFFs.")
parser.add_argument('cines', metavar='cines', type=str, nargs='+', help='input cine file(s), append [start:end:skip] (python slice notation) to filename to convert only a section')
parser.add_argument('-o', dest='output', type=str, default='%s.avi', help='output filename, may use %%s for input filename w/o extension or %%d for input file number')
parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')
parser.add_argument('-f', dest='framerate', type=int, default=30, help='frames per second [default: 30]')
parser.add_argument('-q', dest='quality', type=int, default=75, help='JPEG quality (0-100) [default: 75]')
parser.add_argument('-c', dest='clip', type=float, default=0, help='histogram clip, clip the specified fraction to pure black and pure white, and interpolate the rest; applied before gamma; recommended is 1E-4 - 1E-3 [default: 0]')
parser.add_argument('-s', dest='hist_skip', type=int, default=10, help='only check every Nth frame when computing histogram clip [default: 5]')
parser.add_argument('-r', dest='rotate', type=int, default=0, help='amount to rotate in counterclockwise direction, must be multiple on 90 [default: 0]')
parser.add_argument('-t', dest='timestamp', default=False, action='store_true', help='write a timestamp on each frame')
parser.add_argument('--font', dest='font', type=str, default=os.path.join(script_dir, 'Helvetica.ttf'), help='font (ttf) used for the timestamp')
parser.add_argument('--ts', dest='ts', type=int, default=25, help='text size for timestamp in pixels [25]')
parser.add_argument('--tx', dest='tx', type=int, default=25, help='timestamp x origin [25]')
parser.add_argument('--ty', dest='ty', type=int, default=25, help='timestamp y origin [25]')
parser.add_argument('--td', dest='td', type=int, default=None, help='digits in timestamp [determined from framerate]')
parser.add_argument('--tb', dest='tb', type=int, default=255, help='Test brightness, 0-255 [255=white].')
args = parser.parse_args()


def fmt_time(t):
    return '%d:%02d:%02d' % (int(t/3600.), int(t/60.)%60, int(t)%60)


#------------------------------------------------------------------------------
# Class for printing pretty status counters
#------------------------------------------------------------------------------

class StatusPrinter(object):
    def __init__(self, count, msg='Calculating...', min_time=1./30):
        if hasattr(count, '__len__'):
            self.data = count
            self.count = len(count)
        elif hasattr(count, '__iter__'):
            self.data = list(count)
            self.count = len(self.data)
        else:
            self.count = count
        
        self.msg = msg
        self.current = -1
        self.start = time.time()
        self.last_time = time.time()
        self.max_len = 0
        self.min_time = min_time
        self.extra_msg = ''

        if not hasattr(self, 'data'): self.update()
    
    def message(self, msg):
        self.extra_msg = msg
    
    def print_msg(self, msg, final=False):
        if len(msg) > self.max_len: self.max_len = len(msg)
        
        msg = ('%%-%ds' % self.max_len) % msg
        
        if final: print '\r' + msg
        else: print '\r' + msg,
            
        sys.stdout.flush()        


    def update(self, extra_msg=''):
        self.current += 1
        t = time.time()
        
        if self.current < self.count and (t - self.last_time) < self.min_time:
            return None
        
        self.last_time = t
        
        percent = int(100 * self.current / self.count + 0.5)
        
        if not extra_msg: extra_msg = self.extra_msg
        if extra_msg: extra_msg = ' [%s]' % extra_msg
        
        elapsed = t - self.start

        if self.current == self.count:
            self.print_msg(self.msg + ' done in %s. ' % fmt_time(elapsed) + extra_msg, final=True)
        elif self.current <= 0:
            self.print_msg(self.msg + ' %2d%% ' % percent + extra_msg)
        elif self.current < self.count:
            est = elapsed / self.current * (self.count - self.current)
                
            self.print_msg(self.msg + ' %2d%% (%s remaining) ' % (percent, fmt_time(est)) + extra_msg)

    def __iter__(self):
        return self
    
    def next(self):
        self.update()
        
        if self.current < self.count:
            if hasattr(self, 'data'): return self.data[self.current]
            else: return self.current
        else:
            raise StopIteration

def noneint(s):
    return None if not s else int(s)

if args.timestamp:
    import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(args.font, args.ts)

for i, fn in enumerate(args.cines):
    fn = fn.strip()
    
    frame_slice = slice(None)
    if '[' in fn:
        if fn[-1] == ']':
            fn, s = fn.split('[')
            try:
                frame_slice = slice(*map(noneint, s[:-1].split(':')))
            except:
                raise ValueError("Couldn't convert '[%s' to slice notation" % s)

        else:
            print "Warning, found '[' in input, but it didn't end with ']', so I'll assume you didn't mean to give a frame range."
    
    base, ext = os.path.splitext(fn)
    ext = ext.lower()
    
    if not os.path.exists(fn):
        print "File %s not found, ignoring." % fn
        continue
    
    output = args.output
    if '%s' in args.output: output = output % base
    elif '%' in args.output: output = output % i
    
    bpp = None
    
    if ext in ('.cin', '.cine'):
        input = cine.Cine(fn)
        bpp = input.real_bpp
        if bpp < 8 or bpp > 16: bpp = None #Just in case

        td = args.td if args.td else int(ceil(log10(input.frame_rate)))
        frame_text = lambda i: 't: %%.%df s' % td % (i/float(input.frame_rate))
        
    elif ext in ('.tif', '.tiff'):
        input = cine.Tiff(fn)
        frame_text = lambda i: str(i)
        
        
    bpps = input[0].dtype.itemsize * 8
    if bpp is None: bpp = bpps
    
    frames = range(*frame_slice.indices(len(input)))
    
    if args.clip == 0:
        map = linspace(0., 2.**(bpps - bpp), 2**bpps)
    else:
        counts = 0
        bins = arange(2**bpps + 1)
        
        for i in frames[::args.hist_skip]:
            c, b = histogram(input[i], bins)
            counts += c
        
        counts = counts.astype('d') / counts.sum()
        counts = counts.cumsum()
        
        bottom_clip = where(counts > args.clip)[0]
        if not len(bottom_clip): bottom_clip = 0
        else: bottom_clip = bottom_clip[0]

        top_clip = where(counts < (1 - args.clip))[0]
        if not len(top_clip): top_clip = 2**bpps
        else: top_clip = top_clip[-1]

        #print bottom_clip, top_clip
        #import pylab
        #pylab.plot(counts)
        #pylab.show()
        #sys.exit()

        m = 1. / (top_clip - bottom_clip)
        map = clip(-m * bottom_clip + m * arange(2**bpps, dtype='f'), 0, 1)
            
    map = map ** (1./args.gamma)
    
    map = clip(map * 255, 0, 255).astype('u1')

    #print '%s -> %s' % (fn, output)
    
    ofn = output
    output = cine.Avi(output, framerate=args.framerate, quality=args.quality)
    
    #print frames
    for i in StatusPrinter(frames, os.path.basename(ofn)):
        frame = input[i]
        if args.rotate:
            frame = rot90(frame, (args.rotate%360)//90)
            
        frame = map[frame]
        
        if args.timestamp:
            frame = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame)
            draw.text((args.tx, args.ty), frame_text(i), font=font, fill=args.tb)
        
        output.add_frame(frame)
        
    output.close()
        
    