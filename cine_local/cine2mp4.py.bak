#!/usr/bin/env python
import cine
from numpy import *
import os, sys
import argparse
try:
    from PIL import Image
except:
    import Image
import glob
import subprocess

parser = argparse.ArgumentParser(description="Convert CINE file(s) to h264 MP4.  Also works on TIFFs.")
parser.add_argument('cines', metavar='cines', type=str, nargs='+', help='input cine file(s), append [start:end:skip] (python slice notation) to filename to convert only a section')
parser.add_argument('-o', dest='output', type=str, default='%s.mp4', help='output filename, may use %%s for input filename w/o extension or %%d for input file number')
parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')
parser.add_argument('-f', dest='framerate', type=int, default=30, help='frames per second [default: 30]')
parser.add_argument('-b', dest='bitrate', type=str, default='', help='ffmpeg bitrate [default: None - ffmpeg default]')
parser.add_argument('-k', dest='keep_images', action='store_true', default=False, help="don't delete images files after video is created.")
parser.add_argument('-c', dest='clip', type=float, default=0, help='histogram clip, clip the specified fraction to pure black and pure white, and interpolate the rest; applied before gamma; recommended is 1E-4 - 1E-3 [default: 0]')
parser.add_argument('-s', dest='hist_skip', type=int, default=10, help='only check every Nth frame when computing histogram clip [default: 5]')
parser.add_argument('-r', dest='rotate', type=int, default=0, help='amount to rotate in counterclockwise direction, must be multiple on 90 [default: 0]')
parser.add_argument('-W', dest='width', type=int, default=0, help='resize image to this width')
parser.add_argument('-H', dest='height', type=int, default=0, help='resize image to this height')
args = parser.parse_args()

def noneint(s):
    return None if not s else int(s)

def bfn(p):
    return os.path.splitext(p)[0]

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
        
    elif ext in ('.tif', '.tiff'):
        sys.setrecursionlimit(10**5)
        input = cine.Tiff(fn)
        
    test_frame = input[0]
    bpps = test_frame.dtype.itemsize * 8
    if bpp is None: bpp = bpps
    
    oh, ow = test_frame.shape
    
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

    print '%s -> %s' % (fn, output)
    
    #output = cine.Avi(output, framerate=args.framerate, quality=args.quality)
    
    #print frames
    output_dir = '%s-%s-frames' % (bfn(fn), os.path.basename(bfn(sys.argv[0])))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    aspect_ratio = float(ow)/oh    
    if args.width > 0:
        nw = args.width
        if args.height > 0:
            nh = args.height
        else:
            nh = int(round(nw / aspect_ratio))
    elif args.height > 0:
        nh = args.height
        nw = int(round(nh * aspect_ratio))
        
    else:
        resize = None
    
    if resize is not None:
        if (nw + nh) > (ow + oh): resize = Image.BILINEAR
        else: resize = Image.ANTIALIAS
        print "  Resizing frames: (%d, %d) -> (%d, %d)" % (ow, oh, nw, nh)
    
    for j, i in enumerate(frames):
        frame = input[i]
        if args.rotate:
            frame = rot90(frame, (args.rotate%360)//90)
        
        frame = Image.fromarray(map[frame])
        
        if resize:
            frame = frame.resize((nw, nh), resize)
        
        ofn = os.path.join(output_dir, '%08d.tga' % j)
        frame.save(ofn)
       
        print '\r%6d/%d' % ((j+1), len(frames)),
        sys.stdout.flush()
        
    print
        
    cmd = ['ffmpeg',
               '-i', os.path.join(output_dir, '%08d.tga'),
               '-vcodec', 'libx264',
               '-r', str(args.framerate), 
               '-threads', '0']
    
    if args.bitrate:
        cmd += ['-b', args.bitrate]
    
    cmd .append(output)
        
        
    print '----- Running: %s -----' % (' '.join(cmd))
        
    subprocess.Popen(cmd).wait()
        
    print '-'*40
    
    if not args.keep_images:
        for fn in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, fn))
        os.rmdir(output_dir)

    