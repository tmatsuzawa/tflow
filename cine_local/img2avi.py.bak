#!/usr/bin/env python
import cine
from numpy import *
import os, sys
import argparse
try:
    from PIL import Image
except:
    import Image
import re 

parser = argparse.ArgumentParser(description="Collect image file(s) into an AVI. ")
parser.add_argument('images', metavar='images', type=str, nargs='+', help='input images')
parser.add_argument('-o', dest='output', type=str, default='%s.avi', help='output filename, \n may use %%s for input filename w/o extension and \n %%p for path of folder containing images' )
parser.add_argument('-f', dest='framerate', type=int, default=30, help='frames per second [default: 30]')
parser.add_argument('-q', dest='quality', type=int, default=75, help='JPEG quality (0-100) [default: 75]')
parser.add_argument('-r', dest='rotate', type=int, default=0, help='amount to rotate in counterclockwise direction [default: 0]')
parser.add_argument('-W', dest='width', type=int, default=0, help='width of output, defaults to first image width')
parser.add_argument('-H', dest='height', type=int, default=0, help='height of output, defaults to preserving aspect ratio')
parser.add_argument('--nosort', dest='sort', default=True, action='store_false', help='supress sort of input names')
parser.add_argument('--smart', dest='smart', default=False, action='store_true', help='smart image name parsing; inserts images for skipped frames, etc.')
parser.add_argument('-x', dest='clip_x', type=str, default=':', help='clip of input image in x direction, in python slice notation (applied before scaling)')
parser.add_argument('-y', dest='clip_y', type=str, default=':', help='clip of input image in y direction, in python slice notation (applied before scaling)')
parser.add_argument('-p', dest='preview', default=False, action='store_true', help='apply image transforms and display sample frame (avi is not written)')
parser.add_argument('--frames',dest = 'framelist', default=':', help='list of frames in slice notation [defult: :]')
args = parser.parse_args()

#parser.add_argument('-c', dest='clip', type=float, default=0, help='histogram clip, clip the specified fraction to pure black and pure white, and interpolate the rest; applied before gamma; recommended is 1E-4 - 1E-3 [default: 0]')
#parser.add_argument('-s', dest='hist_skip', type=int, default=10, help='only check every Nth frame when computing histogram clip [default: 5]')
#parser.add_argument('-g', dest='gamma', type=float, default=1., help='gamma of output, assumes input is gamma 1, or: I -> I**(1/gamma); use 2.2 to turn linear image to "natural" for display [default: 1]')

def noneint(s):
    return None if not s.strip() else int(s)

try:
    clip_x = map(noneint, args.clip_x.split(':'))
except:
    print "Clip X should be in format start:end, with negative values permisible for end referenced clips"
    sys.exit()

try:
    clip_y = map(noneint, args.clip_y.split(':'))
except:
    print "Clip Y should be in format start:end, with negative values permisible for end referenced clips"
    sys.exit()

images = list(args.images)
if args.sort: images.sort()

width = args.width
height = args.height

rotate = args.rotate
test_image = Image.open(images[0])
w, h = test_image.size

clip_x = slice(*clip_x).indices(w)[:2]
clip_y = slice(*clip_y).indices(h)[:2]
clip = (clip_x[0], clip_y[0], clip_x[1], clip_y[1])

test_image = test_image.crop(clip)

if rotate: test_image = test_image.rotate(rotate)
w, h = test_image.size

if width <= 0:
    width = w
    
if height <= 0:
    height = int(round(width * float(h) / w))


if not args.preview:
    output = args.output
    if '%s' in args.output: output = output.replace('%s',os.path.splitext(os.path.basename(images[0]))[0])
    if '%p' in args.output:
        output = output.replace('%p','') 
        output = os.path.join(os.path.split(images[0])[0],os.path.basename(output))
    
    output = cine.Avi(output, framerate=args.framerate, quality=args.quality)

if args.smart:
    # this chunk creates the list of files to be put into movie - currently drops all frames that are missing
    # put this whole thing into an if statement in case slice is used to specify stuff
    temp = re.findall(r'\d+', os.path.basename(os.path.split(images[0])[1]))
    if len(temp)>1: print 'what have you done??? \n the filename contains more than two numbers that characterize it \n taking last one....'
    # in case you want to add a query use temp = raw_input('hello') and then swap temp -1 for f
    # create dictionary to obtain file names from kk integers    
    kklist = dict([int(re.findall(r'\d+', os.path.basename(fn))[-1]),images[i]] for i, fn in enumerate(images))
    #slice(args.framelist)
    frame_slice = slice(*map(noneint, args.framelist.split(':')))
    frames = range(*frame_slice.indices(max(kklist.keys())))
    if not args.framelist==':':
        print 'note there are missing frames' + str(list(set(frames) - set(kklist.keys())))
    #ignore missing frames
    kklistf = [kklist[f] for f in frames if f in kklist]
    #kklistf = [kklist.get(f,False) for f in frames if f in kklist]
else:
    kklistf = images
    
for i, fn in enumerate(kklistf):
    #print fn
    
    try:
        img = Image.open(fn)
    except:
        print "!! SKIPPED: %s (python imaging library couldn't open) !!" % fn
        continue

    img = img.crop(clip)

    if rotate: img = img.rotate(rotate)
    
    if (width, height) != img.size:
        if width < img.size[0]:
            img = img.resize((width, height), Image.ANTIALIAS)
        else:
            img = img.resize((width, height), Image.BICUBIC)
            
    if args.preview:
        img.show()
        sys.exit()
    else:
        output.add_frame(img)
    
    del img
    
output.close()    