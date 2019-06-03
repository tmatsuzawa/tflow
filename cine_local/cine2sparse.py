#!/usr/bin/env python
import cine
import sys, glob, os
from numpy import clip as nclip
from numpy import iinfo
import time

all_fields = cine.HEADER_FIELDS + cine.BITMAP_INFO_FIELDS + cine.SETUP_FIELDS


def convert_file(ifn, ofn=None, clip=0, max_frame=None, overwrite=False):
    if ofn is None:
        ofn = os.path.splitext(ifn)[0] + '.sparse'
    
    if not overwrite and os.path.exists(ofn):
        print "%s exists... skipping. (use --overwrite to override)" % ofn
        return
    
    c = cine.Cine(ifn)
    
    header = {}
    for key, t in all_fields:
        val = getattr(c, key)
        if type(val) == str and '\x00' in val: val = val[:val.index('\x00')]
        #print '%20s: %s' % (key, repr(val))
        header[key] = repr(val)
        
    header['original'] = ifn
    header['clip'] = str(clip)
        
    if (not max_frame) or (max_frame > len(c)): max_frame = len(c)
        
        
    
    p = -1
    
    ifp, ifnn = os.path.split(ifn)
    ofp, ofnn = os.path.split(ofn)
    
    max_len = 70
    
    print ifn
    print '  -> ' + ofn
#    msg = ' --> %s' %(ofn if ofp != ifp else ofnn)
#    if len(msg) > max_len: msg = msg[:max_len-3] + '...'   
#    print msg,
#    sys.stdout.flush()

    s = cine.Sparse(ofn, 'w', header, max_blocks=max_frame) 
    max_val = iinfo(c[0].dtype).max


    gtot = 0
    ctot = 0
    stot = 0
    for i in range(max_frame):
        st = time.time()
        f = c[i]
        gtot += (time.time() - st)
        
        st = time.time()
        f = nclip(f, clip, max_val) - clip
        ctot += (time.time() - st)
        
        st = time.time()
        s.append_array(f)
        stot += (time.time() - st)
        
        pn = i * 100 // max_frame
        if pn != p:
            print '\r  %3d%% (load:%6d s, clip:%6d s, save:%6d s)' % (pn, gtot, ctot, stot),
            sys.stdout.flush()
            #print gtot, ctot, stot
            p = pn
        
    print '\r  done. (load:%6d s, clip:%6d s, save:%6d s)' % (gtot, ctot, stot)
        
    s.close()
    #c.close()
    
def crawl(p, ext=None):
    if os.path.isdir(p):
        fns = []
        
        try:
            for f in os.listdir(p):
                fns += crawl(os.path.join(p, f), ext)
        except:
            print 'Failed to list dir "%s," probably permissions problem. (ignoring)' % p
        
        return fns
    else:
        if (ext is None) or os.path.splitext(p)[1] == ext:
            return [p]
        else:
            return []
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sparse convert a cine file.  Values below clip are set to 0.')
    parser.add_argument('input', metavar='input', type=str, nargs='+',help='input files/directories (crawl mode)')
    parser.add_argument('-c', dest='crawl', action='store_true', default=False, help='crawl directories for files with extension .cine')
    parser.add_argument('-m', dest='clip', type=int, default=80, help='bottom clip for sparse convesion [80]')
    parser.add_argument('-s', dest='size', type=float, default=5, help='minimum size (in GB) to initiate a conversion')    
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=False, help='over-write existing cines')
    args = parser.parse_args()
    
    if args.crawl:
        fns = []
        #print args.input
        for fn in args.input: fns += crawl(fn, ext='.cine')
        #print fns
        #sys.exit()
    else:
        fns = args.input
    
    if len(fns) == 1:
        convert_file(fns[0], clip=args.clip, overwrite=args.overwrite)

    else:
        for fn in fns:
            try:
                fs = os.path.getsize(fn) / 1E9
                if fs < args.size:
                    print "'%s' < %.3fGB, ignoring..." % (fn, args.size)
                else:
                    convert_file(fn, clip=args.clip, overwrite=args.overwrite)
            except:
                print "!!! ERROR CONVERTING '%s' !!!\n   (run with one input to see full error output.)" % fn
    
#if __name__ == '__main__':
#    from pylab import *
#
#    d = 420
#    
#    ifn = glob.glob('*.cine')[0]
#    ofn = os.path.splitext(ifn)[0] + '.sparse'
#    
#    convert_file(glob.glob('*.cine')[0], clip=80)#, max_frame=d*9)
#    
#    s = cine.Sparse(ofn)
#    
#    for i in range(9):
#        subplot(3, 3, i+1)
#        
#        frame = zeros(s[i].shape, dtype='f')
#        for j in range(d): frame += s[j+i*d]
#        print frame.shape
#        
#        imshow(frame)
#        
#    show()