#!/usr/bin/env python
import cine
from numpy import *
import os, sys
import argparse
import time

parser = argparse.ArgumentParser(description="Convert CINE files to an AVI, horizontally stacking them.  Image manipulations applied per image.")
parser.add_argument('cines', metavar='cines', type=str, nargs='+', help='input cine files, can append slice notation to indidually clip them.')
parser.add_argument('-o', dest='output', type=str, default='%s.avi', help='output fileame, %s is base which joins the non-common parts of the names [default: %%s.avi]')
parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')
parser.add_argument('-f', dest='framerate', type=int, default=30, help='frames per second [default: 30]')
parser.add_argument('-q', dest='quality', type=int, default=75, help='JPEG quality (0-100) [default: 75]')
parser.add_argument('-c', dest='clip', type=float, default=0, help='histogram clip, clip the specified fraction to pure black and pure white, and interpolate the rest; applied before gamma; recommended is 1E-4 - 1E-3 [default: 0]')
parser.add_argument('-s', dest='hist_skip', type=int, default=10, help='only check every Nth frame when computing histogram clip [default: 5]')
parser.add_argument('-r', dest='rotate', type=int, default=0, help='amount to rotate in counterclockwise direction, must be multiple of 90 [default: 0]')
parser.add_argument('--frames',dest = 'frames', default=':', help='list of frames in slice notation [defult: :]')
args = parser.parse_args()

def noneint(s):
    return None if not s else int(s)


class FrameGenerator(object):
    def __init__(self, fn, input_args):

        self.frame_slice = slice(None)
        if '[' in fn and fn[-1] == ']':
            fn, s = fn.split('[')
            try:
                self.frame_slice = slice(*map(noneint, s[:-1].split(':')))
            except:
                raise ValueError("Couldn't convert '[%s' to slice notation for file '%s'" % (s, fn))
        
        self.fn = fn
        self.source = cine.Cine(fn)
        self.frames = range(*self.frame_slice.indices(len(self.source)))
        
        self.gamma = getattr(input_args, 'gamma', 1)
        self.rotate = getattr(input_args, 'rotate', 0)
        self.clip = getattr(input_args, 'clip', 0)
        self.hist_skip = getattr(input_args, 'hist_skip', 0)
        
        test_frame = self.source[0]
        bpps = test_frame.dtype.itemsize * 8
        bpp = self.source.real_bpp
        
        if args.clip == 0:
            self.map = linspace(0., 2.**(bpps - bpp), 2**bpps)
        else:
            counts = 0
            bins = arange(2**bpps + 1)
            
            for i in self.frames[::args.hist_skip]:
                c, b = histogram(self.source[i], bins)
                counts += c
            
            counts = counts.astype('d') / counts.sum()
            counts = counts.cumsum()
            
            bottom_clip = where(counts > args.clip)[0]
            if not len(bottom_clip): bottom_clip = 0
            else: bottom_clip = bottom_clip[0]
    
            top_clip = where(counts < (1 - args.clip))[0]
            if not len(top_clip): top_clip = 2**bpps
            else: top_clip = top_clip[-1]
            
            #print bottom_clip
            #print top_clip
    
            m = 1. / (top_clip - bottom_clip)
            self.map = clip(-m * bottom_clip + m * arange(2**bpps, dtype='f'), 0, 1)

        self.map = self.map ** (1./args.gamma)
    
        self.map = clip(self.map * 255, 0, 255).astype('u1')            
        
    def __len__(self):
        return len(self.frames)

    len = __len__

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.__getitem__, range(self.image_count)[key])

        else:
            frame = self.map[self.source[self.frames[key]]]
            if self.rotate:
                frame = rot90(frame, (self.rotate%360)//90)            
            return frame

start = time.time()

fns = map(lambda x: os.path.splitext(x)[0], args.cines)
for i in range(min(map(len, fns))):
    broken = False
    for j in range(len(fns) - 1):
        if fns[j][i] != fns[j+1][i]:
            broken = True
            break
    if broken: break
basename = fns[0][:i] + '_AND_'.join(map(lambda x: x[i:], fns))

        
frame_slice = slice(None)
try:
    frame_slice = slice(*map(noneint, args.frames.split(':')))
except:
    raise ValueError("Couldn't convert '%s' to slice notation" % s)


inputs = [FrameGenerator(fn, args) for fn in args.cines]
shapes = array([input[0].shape for input in inputs])

h = shapes[:, 0].max()
w = shapes[:, 1].sum()

max_frames = min([len(input) for input in inputs])

ofn = args.output.replace('%s', basename)
for input in inputs:
    print input.fn
print '  --> %s' % ofn

#print shapes
#print h, w
#print [len(input) for input in inputs]
#print max_frames
#
#print ofn
#print range(*frame_slice.indices(max_frames))
#sys.exit()


output = cine.Avi(ofn, framerate=args.framerate, quality=args.quality)
frames = range(*frame_slice.indices(max_frames))
N = len(frames)

fmt = '\r  %%%dd/%%%dd' % (len(str(N)), len(str(N)))
print fmt % (0, N),

for j, i in enumerate(frames):
    frame = zeros((h, w), 'u1')
    
    x0 = 0
    for input in inputs:
        subframe = input[i]
        sh, sw = subframe.shape
        frame[:sh, x0:x0+sw] = subframe
        x0 += sw
        print fmt % (j+1, N),
        sys.stdout.flush()
        
    
    output.add_frame(frame)
        
print '\r  done in %.1fs' % (time.time() - start)

output.close()
        
    