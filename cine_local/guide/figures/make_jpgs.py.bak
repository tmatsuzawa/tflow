#!/usr/bin/env python
import Image
import glob, os
import re
import subprocess

QUALITY = 90

for fn in glob.glob('*dpi.png'):
    ofn = os.path.splitext(fn)[0] + '.jpg'

    print fn, '->', ofn
    
    Image.open(fn).save(ofn, quality=QUALITY)
    
    dpi = int(re.match('.*?([0-9]+)dpi.*?', fn).group(1))
    
    
    try: subprocess.check_call(["exiftool", ofn, '-XResolution=%d' % dpi, '-YResolution=%d' % dpi, '-ResolutionUnit=inches'])
    except: 'Exiftool failed to run -- do you have it installed?  Image DPI not set.'
    
    try: subprocess.check_call(["rm", ofn+"_original"])
    except: 'Failed to delete original jpg.'