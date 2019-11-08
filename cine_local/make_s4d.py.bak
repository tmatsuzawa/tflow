#!/usr/bin/env python
import argparse, sys, os, glob
import cine, time
from numpy import clip, arange, array, zeros
import ast

DEFAULT_3D_SETUP = "#3d setup file\n#Required fields: 'cine_depth', 'display_frames', 'bottom_clip', 'u1_top', 'u1_gamma', 'scale', 'rate', 'x_func', 'y_func', 'z_func'\n\n#-------------------------------------------------------------------------------\n#For converting from cine/sparse to S4D\ncine_depth = 420\nx_size = 384\nframe_shape = (384, 384)\ndisplay_frames = range(6, 390) \nbottom_clip = 80\n\n#For conversion to 8 bit only\nu1_top = 2000\nu1_gamma = 2.0\n\n#-------------------------------------------------------------------------------\n\n# Scale is mm/pixel\nscale = 0.4\n# Rate is volumes/sec\nrate = 16E6/211/420\n\n#-------------------------------------------------------------------------------\n\n# Half-edge length, used as scale for dimensionless units\na = x_size * scale / 2.\na_in = a / 25.4\nn = 1.33\n\n# Effective distance (in water!) from center of airfoil to camera, scaled by half-edge length\nL = (59.5 + 16*n) / a_in\n\n# Offset of camera center from airfoil center\nDx = 12 / a_in\n\n# Offset of scan center from center of airfoil (in water!)\nS = (12 + 30.5*n) / a_in\n\n#Depends on scan direction\nz_sign = 1\n\n\n\n\n\n\n#===============================================================================\n# Calculate angles/distortions and perspective correction functions\n# This shouldn't need to be edited!\n\nPhi = Dx / L\n\nx_x  = cos(Phi)\nx_z  = sin(Phi) * z_sign\nx_xz = cos(2*Phi) / L * z_sign\n\ny_y  = 1\ny_xy = -sin(Phi) / L\ny_yz = cos(Phi) / L * z_sign\n\nz_z  = z_sign\nz_xz = 1 / S * z_sign\n\n# x/y/z scaled (-1:1) for cubic frame\n# def's are used instead of lambda to get closure on the variables.\n\ndef x_func(x, y, z, x_x=x_x, x_z=x_z,   x_xz=x_xz):\n    return x_x*x + x_z*z + x_xz*x*z\n\ndef y_func(x, y, z, y_y=y_y, y_xy=y_xy, y_yz=y_yz):\n    return y_y*y +         y_xy*x*y + y_yz*y*z \n\ndef z_func(x, y, z, z_z=z_z, z_xz=z_xz):\n    return z_z*z +         z_xz*x*z"

if '--example' in sys.argv:
    f = open('example.3dsetup', 'w')
    f.write(DEFAULT_3D_SETUP)
    f.close
    sys.exit()

parser = argparse.ArgumentParser(description="Convert CINE or SPARSE files to S4D, for use in V4D.  A 3dsetup file must be present in the source directory, or one can be specified.")
parser.add_argument('input', metavar='input', type=str, nargs='+',help='input files -- should be .sparse or .cine, or directories (crawl mode)')
parser.add_argument('-c', dest='crawl', action='store_true', default=False, help='crawl directories for files with extensions .sparse or .cine')
parser.add_argument('-s', dest='setup', type=str, default=None, help='3dsetup file, if not specified the first file with extension .3dsetup in the same directory as the source is used.')
parser.add_argument('-t', dest='type', type=str, default='u1', help="data type of output s4d -- valid options are 'u1' (default), 'u2', 'f' or 'd'.  'u1' has gamma correction/clipping applied, and 'f'/'d' are rescaled to (0, 1) based on the 'real_bpp' header field")
parser.add_argument('-o', dest='output', type=str, default='%s.s4d', help="output filename, use %s to original base name replacement [%s.s4d]")
parser.add_argument('-r', dest='range', type=str, default=':', help="range of frames to convert, in pyton slice format [:]")
parser.add_argument('-d', dest='downsample', action='store_true', default=False, help="downsample the image, halfing the resolution in each dimension")
parser.add_argument('--example', dest='example', action='store_true', default=False, help='create an example 3dsetup file')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False, help='over-write existing S4Ds')
args = parser.parse_args()

files = []
for ii in args.input:
    if '*' in ii or '?' in ii: files += glob.glob(ii)
    else: files.append(ii)

def crawl(l, ext=None):
    ext = map(str.lower, ext)
    files = []
    for fn in l:
        if os.path.isdir(fn):
            files += map(lambda x: os.path.join(fn, x), crawl(os.listdir(fn), ext))
        else:
            if ext and os.path.splitext(fn)[1].lower() not in ext:
                continue
            else:
                files.append(fn)
                
    return files

ext = ['.sparse', '.cine']
if args.crawl: files = crawl(files, ext)

if args.type not in ('u1', 'u2', 'f', 'd'):
    print "ERROR: data type must be 'u1', 'u2', 'f' or 'd'"
    sys.exit()

removed = []
files_lower = map(lambda x: os.path.splitext(x)[0] +  os.path.splitext(x)[1].lower(), files)

for fn in files[:]: #Make a copy
    base, ee = os.path.splitext(fn)
    if ee.lower() not in ext:
        removed.append(fn + '  (bad extension)')
        files.remove(fn)
    elif ee.lower() == '.cine' and (base + '.sparse') in files_lower:
        removed.append(fn +  '  (.sparse equivalent in list)')
        files.remove(fn)
        
if removed:
    print 'The following files will be ignored:'
    print '-'*40
    for fn in removed: print '  ' + fn
    print '-'*40
    

__FMT_TIME = [(60, '%06.3f"'), (60, "%02d'"), (24, "%02d hr"), (365, "%3d days"), (None, '%d years')]

def fmt_time(x, max_sections=None):
    sections = []
    for m, fmt in __FMT_TIME:
        rx = x
        if m:
            rx %= m
            x //= m
        else: x = 0
            
        sections.insert(0, fmt % rx)
        if not x: break
    
    sections[0] = sections[0].lstrip('0')
    if sections[0][0] == '.': sections[0] = "0" + sections[0]
    if max_sections: sections = sections[:max_sections]
    return ' '.join(sections)

def eval_vals(s):
    setup = {}
    dummy = {}
    exec("from math import *", dummy)
    exec(s, dummy, setup)
    return setup
    
def try_eval(s):
    while type(s) == str:
        try:
            s = ast.literal_eval(s)
        except:
            return s
    return s
    
def _half_res(x):
    if x.ndim == 0:
        return x
    elif x.ndim == 1:
        m1 = x.shape[0]//2 * 2
        return 0.5 * (x[:m1:2] + x[1:m1:2])
    elif x.ndim == 2:
        m1 = x.shape[0]//2 * 2
        m2 = x.shape[1]//2 * 2
        return 0.25 * (x[:m1:2, :m2:2] + x[1:m1:2, :m2:2] + x[:m1:2, 1:m2:2] + x[1:m1:2, 1:m2:2])
    else:
        raise ValueError("Can't downsample arrays with more than 3 dimensions")
    

def half_res(a):
    #Reduce the resolution by a factor of 2 for an ND array
    #Loops over the first access, so 3D arrays can be handled without using
    #  too much memory.
    return array([
        _half_res(0.5 * (a[2*n].astype('f') + a[2*n+1].astype('f')))  
        for n in range(len(a)//2)
    ], dtype=a.dtype)    
    
    
__REQUIRED_FIELDS = ['cine_depth', 'display_frames', 'bottom_clip', 'u1_top', 'u1_gamma', 'scale', 'rate', 'x_func', 'y_func', 'z_func']
    
    
for fn in files:
    start = time.time()
    
    print fn
    
    base, ee = os.path.splitext(fn)
    ofn = args.output.replace('%s', base)

    if os.path.exists(ofn) and not args.overwrite:
        print ' !! %s exists, skipping' % ofn
        continue
    else: print '  -> %s' % ofn    
    
    if not args.setup:
        try:
            path = os.path.split(fn)[0]
            setups = glob.glob(os.path.join(path, '*.3dsetup'))
            setupfile = sorted(setups)[0]
        except:
            print ' !! no .3dsetup file, either specify with -s or place in source directory'
            continue
    else:
        setupfile = args.setup    
    
    setup_str = open(setupfile, 'rt').read()
    setup = eval_vals(setup_str)
    
    for rf in __REQUIRED_FIELDS:
        if rf not in setup:
            print "ERROR: field '%s' missing from setup file! (%s)" % (rf, setupfile)
            sys.exit()
    
    if ee == '.cine':
        input = cine.Cine(fn)
        bit_depth = input.real_bpp
        bottom_clip = setup['bottom_clip']
        u1_top = setup['u1_top']
    
    else:
        input = cine.Sparse(fn)
        bit_depth = try_eval(input.header['real_bpp'])
        bottom_clip = 0
        u1_top = setup['u1_top'] - try_eval(input.header.get('clip', 0)) #Clip was already applied in sparsing
        
    num_frames = len(input) // setup['cine_depth']
    test_frame = input[0]
    header = {'original_bitdepth':bit_depth, 'original_file':fn, 'bottom_clip':bottom_clip, '3dsetup':setup_str, 'use_3dsetup_perspective':True, 'dtype':args.type, 'frame size': test_frame.shape}
    
    #print header
    
    frames = eval('x[%s]' % args.range, {'x':range(num_frames)})
    
    fmt = '%d' % len(frames)
    fmt = '\r    %%%dd/%s' % (len(fmt), fmt)

    if test_frame.dtype == 'u1':
        input_max = 2**8
    elif test_frame.dtype == 'u2':
        input_max = 2**16
    else:
        raise ValueError('Input data type must be 8 or 16 bit unsigned integers.')

    valmap = clip(arange(input_max), bottom_clip, input_max-1) - bottom_clip
    
    
    if args.type == 'u1':
        valmap = clip(valmap / float(u1_top - bottom_clip), 0, 1)
        valmap = valmap ** (1./setup['u1_gamma'])
        valmap = clip(valmap * 255, 0, 255).astype('u1')
    elif args.type == 'u2':
        valmap = valmap.astype('u2')
    elif args.type in ('f', 'd'):
        valmap = valmap.astype(args.type) / (2**bit_depth - 1)

    volume_shape = (len(setup['display_frames']), ) + test_frame.shape

    output = cine.Sparse4D(ofn, 'w', header)
    print '',
    for i, j in enumerate(frames):
        print fmt % (i+1),
        sys.stdout.flush()

        frame = zeros(volume_shape, dtype=args.type)
                        
        for ik, k in enumerate(setup['display_frames']):
            frame[ik] = valmap[input[k + j * setup['cine_depth']]]
    
        if args.downsample:
            frame = half_res(frame)

        output.append_array(frame)

    output.close()

    print '\r    done in %s' % fmt_time(time.time() - start)
        