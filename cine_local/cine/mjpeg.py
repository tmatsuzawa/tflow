#!/usr/bin/env python

################################################################################

# Simple writer for MJPEG video in an AVI container
# Author: Dustin Kleckner
# dkleckner@uchicago.edu

################################################################################

try:
    from PIL import Image
except:
    import Image
import numpy as N
import struct
import StringIO
import os, sys

#from decode_test import hex, dwords

def dword(*x):
    if len(x) == 1: x = x[0]
    if hasattr(x, '__iter__') and type(x) is not str:
        return ''.join(dword(y) for y in x)
    else:
        if type(x) is str:
            if len(x)%4 != 0:
                raise ValueError('dword only encodes strings with 4N length!')
            return x
        elif x > 0:
            return struct.pack('I', x)
        else:
            return struct.pack('i', x)

def asimage(x):
    #if isinstance(x, Image.Image): return x
    
    if type(x) == N.ndarray:
        if x.dtype in ('f', 'd'):
            x = (N.clip(x, 0, 1) * 255).astype('u1')
        return Image.fromarray(x)
        
    else:
        return x
        #print x
        #print isinstance(x, Image)
        #raise ValueError('asimage only works with numpy arrays or images')


def jpeg_string(img, **kwargs):
    buf = StringIO.StringIO()
    img.save(buf, format='JPEG', **kwargs)
    s = buf.getvalue()
    buf.close()
    return s


def riff_chunk(*chunks):
    parts = []
    
    for chunk in chunks:
        if type(chunk) == str:
            parts.append(chunk)
            
        elif type(chunk) in (list, tuple) and len(chunk) == 2:
            id, data = chunk
            if type(data) in (list, tuple): data = riff_chunk(*data)
            
            parts += [dword(id, len(data)), data]
            if len(data)%2: parts.append(chr(0)) #Word padding!
            
        else: raise ValueError('can only parse string or 2-tuple chunks')

    return ''.join(parts)
    
class Avi(object):
    HEADER_CLEARANCE = 4096 #blank space in the file left for the header
    
    def __init__(self, fn, framerate=30, strn='Python MJPEG', quality=None, jpeg_args={}):
        self.f = open(fn, 'wb')
        self.open = True
        self.width = None
        self.height = None
        self.framerate = int(framerate)
        self.num_frames = 0
        self.strn = strn
        
        if quality: jpeg_args['quality'] = quality
        
        self.jpeg_args = jpeg_args
        
        self.f.write(riff_chunk(
            ('RIFF', ('AVI ',
                ('JUNK', chr(0) * self.HEADER_CLEARANCE), #Header will go here later.
                ('LIST', 'movi') #Frame chunks go in this list, we'll mod the size later.
            ))
        ))
        
        self.riff_size_addr = 4
        self.header_addr = 12
        self.next_chunk_addr = self.f.tell()

        self.movi_addr = self.f.tell() - 4
        self.movi_size_addr = self.movi_addr - 4 #We need to write the total size of the movi chunk later
        
        self.index = []
        
        
    def close(self):
        if not self.open: return
        
        self.write_header()
        
        self.f.seek(self.movi_size_addr)
        self.f.write(dword(self.next_chunk_addr - self.movi_addr))
        
        self.f.seek(self.next_chunk_addr)
        self.f.write(riff_chunk(('idx1', ''.join(self.index))))
        total_len = self.f.tell() - 8
        
        self.f.seek(self.riff_size_addr)
        self.f.write(dword(total_len))
        self.f.close()
        
        self.open = False
        
        
    def __del__(self):
        self.close()
        
        
    def write_header(self):
        if not self.open: raise ValueError('avi already closed')
        
        #Useful info: http://msdn.microsoft.com/en-us/library/ms779636.aspx
        #More usefull info: http://pvdtools.sourceforge.net/aviformat.txt
        #Note: Some of the header entries (which aren't documented above) are based on an example ImageJ created AVI
        
        chunk = riff_chunk(
            ('LIST', ('hdrl', #header
                ('avih', dword( #avi header
                    int(1E6/self.framerate), 0, 0, 16, self.num_frames, 0, 1,
                    0, self.width, self.height, 0, 0, 0, 0    
                )),
                ('LIST', ('strl',
                    ('strh', dword( #AVISTREAMHEADER chunk (http://msdn.microsoft.com/en-us/library/ms779638.aspx)
                        'vids', 'DIB ', 0, 0, 0, 1, self.framerate, 0,
                        self.num_frames, 0, -1, 0, 0, 0
                    )),
                    ('strf', dword( #BITMAPINFOHEADER chunk (http://msdn.microsoft.com/en-us/library/dd183376(v=vs.85).aspx)
                        40, self.width, self.height, 1 + (24<<16), 'MJPG', #I can't find any documentation for the MJPG compression type, but oh well...
                        self.width*self.height*3, 0, 0, 0, 0
                    )),
                    ('strn', '%-15s' % self.strn[:15] + chr(0))
                ))
            ))
        )
        
        self.f.seek(self.header_addr)
        self.f.write(chunk)
        self.f.write(dword('JUNK', self.HEADER_CLEARANCE - len(chunk)))
        
        
    def add_frame(self, img):
        if not self.open: raise ValueError('avi already closed')

        img = asimage(img)
        
        if self.width is None:
            self.width, self.height = img.size
        elif img.size != (self.width, self.height):
            raise ValueError('appeneded images must have same size as original')

        self.num_frames += 1
        
        jpeg = jpeg_string(img, **self.jpeg_args)
        chunk = riff_chunk(('00dc', jpeg))
#        hex(chunk, end=48)
#        dwords(chunk, end=48)
        
        
        self.index.append(dword('00dc', 16, self.next_chunk_addr - self.movi_addr, chunk[4:8]))
        
        self.f.seek(self.next_chunk_addr)
        self.f.write(chunk)
        self.next_chunk_addr = self.f.tell()


#if __name__ == '__main__':
#    from cine.cine import Cine
#    
#    input = Cine('miro_balloon.cine')
#    output = Avi(os.path.expanduser('~/%s.avi' % sys.argv[0]))
#    
#    for frame in input: output.add_frame(frame)
#    
#    output.close()
    
