#!/usr/bin/env python
import cine
import os, sys, time
from math import ceil
import numpy as N
import glob
    
saved_info = [
    ('frame_rate', 'Frame Rate', 'FPS'),
    ('shutter', 'Shutter', 'ms', lambda x: '%.3f'% (int(x) /1E3)),
    ('edr_shutter', 'EDR Shutter', 'ms', lambda x: '%.3f'% (int(x) /1E3)),    
    ('trigger_time', 'Trigger Date', '', cine.T64_S('%a, %b %d, %Y')),
    ('trigger_time', 'Trigger Time', '', cine.T64_S('%I:%M:%S %p')),
    ('post_trigger', 'Post Trigger', 'frames'),
    ('im_width', 'Frame Width'),
    ('im_height', 'Frame Height'),
    ('trigger_time', 'Trigger Time (Unix)', 's after epoch', cine.T64_F_ms),
    ('camera_version', 'Camera Type', '', lambda x: '%.1f' % (float(x) / 10.)),
    ('serial', 'Camera Serial'),
    ('first_image_no', 'First Image', ''),
    ('real_bpp', 'Bit Depth', ''),
]
        
FRAME_CACHE = 10

GiB = 2**30
MiB = 2**20
MAX_SIZE = 4 * GiB - 100 * MiB
#MAX_SIZE = 100 * MiB #For testing

print "Using cine tools revision %s (%s)." % (cine.svn_info.revision, cine.svn_info.date)
files = sys.argv[1:]

try: files.remove('-i')
except: DATA_ONLY = False
else:
    print "Just saving CINE info..."
    DATA_ONLY = True

files_globbed = []
for fn in files: files_globbed += glob.glob(fn)

for fn in files_globbed:

    if DATA_ONLY: print '-- %s --' % fn

    c = cine.Cine(fn)
    f0 = c[0]
    frame_size = f0.size * f0.itemsize

    frames_per_file = MAX_SIZE // frame_size
    num_files = int(ceil(len(c) / float(frames_per_file)))

    otf = open(os.path.splitext(fn)[0] + '.txt', 'w')

    for i in saved_info:
        f = i[0]
        fn_ = i[1]
        fu = i[2] if len(i) > 2 else ''
        ff = i[3] if len(i) > 3 else lambda x: x

        dat = '%20s: %s %s' % (fn_, ff(repr(getattr(c, f))), fu)
        otf.write(dat + '\n')
        if DATA_ONLY: print '    ' + dat

    otf.close()

    #sys.exit()

    if c.serial == 6048:
        print "Serial #6048 detected -- this is the broken camera!\nAutofix enabled.\nIf the image is rotated, this won't work!"
        c.enable_auto_fix()

    if not DATA_ONLY:
        start = 0

        if num_files > 1:
            print "!! File %s over sized (%.2f GB), splitting into multiple output tiffs. !!" % (fn, MAX_SIZE / 1E9)

        for file_num in range(num_files):
            file_ext = ('_%d' % file_num) if (num_files > 1) else ''

            ofn = os.path.splitext(fn)[0] + file_ext + '.tiff'

            output = cine.Tiff(ofn, 'w')

            end = min(start+frames_per_file, len(c))
            n = end - start
            print "Writing %d frames to %s..." % (n, ofn)

            start_time = time.time()
            marked = list(N.arange(0.1, 1, 0.1) * n + start)
            for i in range(start, end, FRAME_CACHE):
                frames = c[i:i+FRAME_CACHE]
                map(output.write_page, frames)
                if marked and i >= marked[0]:
                    marked = filter(lambda x: x > i, marked)
                    print '    %4d/%4d...' % (i-start, n)

            print '    done is %.1f s.' % (time.time() - start_time)
            start = end

