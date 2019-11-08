#!/usr/bin/env python
import sys, os, struct, time
import struct
from numpy import array, frombuffer
#import Image
from threading import Lock
#import multiprocessing as M

TIFF_TYPES = {
    1:'B', 2:'s', 3:'H', 4:'I', 5:'L', #TIFF 5.0
    6:'b', 7:'c', 8:'i', 9:'l', 10:'l', 11:'f', 12:'d' #TIFF 6.0
}

TIFF_TAGS = { #Baseline TIFF tags
    262:('PhotometricInterpretation', 3),
    259:('Compression', 3),
    257:('ImageLength', 4),
    256:('ImageHeight', 4),
    296:('ResolutionUnit', 3),
    282:('XResolution', 5),
    283:('YResolution', 5),
    278:('RowsPerStrip', 4),
    273:('StripOffsets', 4),
    279:('StripByteCounts', 4),
    258:('BitsPerSample', 3),
    277:('SamplesPerPixel', 3),
    315:('Artist', 2),
    265:('CellLength', 3),
    264:('CellWidth', 3),
    320:('ColorMap', 3),
    33432:('Copyright', 2),
    306:('DateTime', 2),
    337:('ExtraSamples', 3),
    266:('FillOrder', 3),
    289:('FreeByteCounts', 4),
    288:('FreeOffsets', 4),
    291:('GrayResponseCurve', 3),
    290:('GrayResponseUnit', 3),
    316:('HostComputer', 2),
    270:('ImageDescription', 2),
    271:('Make', 2),
    281:('MaxSampleValue', 3),
    280:('MinSampleValue', 3),
    272:('Model', 2),
    254:('NewSubfileType', 4),
    274:('Orientation', 3),
    284:('PlanarConfiguration', 3),
    305:('Software', 2),
    255:('SubfileType', 3),
    263:('Thresholding', 3),
}

TIFF_TAGS_I = {}
for k, (n, t) in TIFF_TAGS.iteritems():
    TIFF_TAGS_I[n] = (k, t)

DEFAULT_TAGS = {
    'Compression':1,
    'PhotometricInterpretation':1,
    'SamplesPerPixel':1,
    'Software':sys.argv[0]
}

class Tiff(object):
    def __init__(self, filename, read_write='r'):
        self.filename = filename
        self.IFD_offset = []
        self.IFD_raw = []
        self.mtags = []

        if read_write in ('r'):
            self.file = open(filename, 'rb')
            
            self._byteorder = self.file.read(2)
            if self._byteorder == 'II':
                self.little_endian = True
            elif self._byteorder == 'MM':
                self.little_endian = False
            else:
                raise ValueError('First two bytes of "%s" were not byte order indicators -- is this actually a TIFF?' % filename)
                
            self._42 = self.unpack('H')
            if self._42 != 42:
                raise ValueError('"%s" is not a properly formatted TIFF. (bytes 3-4 != 42)' % filename)
            
            IFD_offset = self.unpack('I')
            self.read_IFD(IFD_offset)
            self.tags = self.mtags[0]
                        
            if self.tags.get('Compression', 1) != 1:
                raise ValueError('This library does not support compressed TIFFs.')
                
            self.bpp = self.tags.get('BitsPerSample', 0)
            if self.bpp not in (8, 16):
                raise ValueError('This library only supports 8 or 16 bit grayscale TIFFs.')
                
            self.pages = len(self.IFD_raw)
            
        elif read_write == 'w':
            self.file = open(filename, 'w+b')
            self.little_endian = True if sys.byteorder == 'little' else False
            self._byteorder = 'II' if self.little_endian else 'MM'
            self.file.write(self._byteorder)
            self.pack('H', 42)
            self.pack('I', 8) #This will be the first IFD offset
    
        else:
            raise ValueError("read_write variable should be 'r' or 'w'")
            
    def write_page(self, image, tags={}):
        img = image.tostring()
        
        auto_tags = DEFAULT_TAGS.copy()
        auto_tags['ImageLength'] = image.shape[0]
        auto_tags['ImageHeight'] = image.shape[1]
        auto_tags['StripByteCounts'] = len(img)
        auto_tags['RowsPerStrip'] = auto_tags['ImageLength']
        
        dtype = image.dtype
        
        if dtype in ('uint8', '>u1', '<u1'):
            bpp = 8
        elif dtype in ('uint16', '>u2', '<u2'):
            bpp = 16
        else:
            raise ValueError("Can only write images that are 8 or 16 bits ('uint8' or 'uint16' numpy data type).\nTried to write type '%s'" % dtype)
            
        auto_tags['BitsPerSample'] = bpp
        
        tags = tags.copy()
        tags.update(auto_tags)
        self.write_IFD(tags)
        
        #Get the strip offset from the IFD I just wrote.
        self.file.seek(self.mtags[-1]['StripOffsets'])
        self.file.write(img)
            
    def write_IFD(self, tags):
        #Writes an IFD for a new page.
        #StripOffsets is added automatically
        
        if 273 in tags or 'StripOffsets' in tags:
            raise ValueError('StripOffsets field must not be in IFD tags!\nThis field is automatically created when packing the directory.')
        
        post_values=''
        
        self.file.seek(0, 2)
        offset = self.file.tell() #End of file -- this is where IFD will be added
        
        #Find the location to indicate the offset of our new IFD
        if len(self.mtags):
            offset_offset = self.IFD_offset[-1] + len(self.IFD_raw[-1]) * 12 + 2 
        else:
            offset_offset = 4

        self.file.seek(offset_offset)
        self.pack('I', offset)

        self.file.seek(offset)

        post_offset = offset + 6 + (len(tags) + 1) * 12
        #Offset of end of IFD tag = offset + 2 [count] + 4 [next_IFD] + 12 * nentries [+1 for strip offset]
        
        self.pack('H', len(tags) + 1)
        for tag, value in tags.iteritems():
            num = tag
            if type(value) not in (list, tuple): value = (value, )
            
            if type(num) is str:
                if num in TIFF_TAGS_I:
                    num = TIFF_TAGS_I[num][0]
                else:
                    raise ValueError('Unrecognized TIFF tag: "%s"\nIf this tag is not a baseline tag, you must specify it as (tag number, value(s), type #).' % num)
            
            self.pack('H', num)
            
            if len(value) > 1:
                tiff_type = value[1]
            else:
                try:
                    tiff_type = TIFF_TAGS[num][1]
                except:
                    raise ValueError('Unrecognized tag number %d; You must specify the TIFF type for the value!' % num)
            if tiff_type not in TIFF_TYPES: raise ValueError('Invalid TIFF type (%d)' % tiff_type)
            
            values = value[0] if type(value[0]) in (tuple, list) else (value[0], )
            count = len(values)
            if tiff_type in (5, 10):
                if count % 2:
                    raise ValueError('Tags with fractional types (5, 10) must have an even number of values.')
                tiff_count = count // 2
            else:
                tiff_count = count
                
            self.pack('H', tiff_type)
            self.pack('I', tiff_count)
                
            values = struct.Struct(('<' if self.little_endian else '>') + str(count) + TIFF_TYPES[tiff_type]).pack(*values)
            if len(values) < 4:
                values = values + chr(0) * (4 - len(values))
                    
            if len(values) > 4:
                if len(values) % 2:
                    values + values + chr(0) #Align all values to word boundaries -- this probably isn't really necessary
                self.pack('I', post_offset + len(post_values))
                post_values = post_values + values
            else:
                self.file.write(values)
        
        #Write strip offset tag last, so we know where to put things        
        strip_offset = post_offset + len(post_values)
        self.pack('HHIII', (273, 4, 1, strip_offset, 0))
        
        #Write values longer than 4 bytes to the file
        self.file.write(post_values)
        self.file.flush()
        os.fsync(self.file.fileno())
        self.read_IFD(offset) #Read back IFD I just created -- a little lazy but robust!
            
    def read_IFD(self, offset):
        page = len(self.IFD_offset)
        self.IFD_offset.append(offset)
        self.file.seek(offset)
            
        IFD_num = self.unpack('H')
        self.IFD_raw.append([self.unpack('HHI4s') for i in range(IFD_num)])
        next_IFD = self.unpack('I')
        self.mtags.append({})

        for tag, type, count, value in self.IFD_raw[-1]:
            type = TIFF_TYPES[type]
            if type in (5, 10): count *= 2
            
            if count > 1:
                type = '%d' % count + type
            
            s = struct.Struct(('<' if self.little_endian else '>') + type)
            if s.size <= 4:
                value = self.unpack(type, value)
            else:
                self.file.seek(offset)
                value = self.unpack(type)
            
            self.mtags[page][tag] = value
            
            tag = TIFF_TAGS.get(tag, None)
            if tag is not None:
                self.mtags[page][tag[0]] = value
            
        if next_IFD != 0:
            self.read_IFD(next_IFD)
            
    def tag_list(self, page=0):
        tags = []
        for tag, value in self.mtags[page].iteritems():
            if type(tag) == int:
                tags.append((tag, TIFF_TAGS.get(tag, ('?',))[0], value))
                
        tags.sort()
        return tags
    
    def get_strip(self, strip_num=0, page=0):
        offsets = self.mtags[page]['StripOffsets']
        if type(offsets) not in (list, tuple):
            if strip_num != 0: raise ValueError('Invalid strip number %d in "%s"' % strip_num, self.filename)
            self.file.seek(offsets)
            return self.file.read(self.mtags[page]['StripByteCounts'])
        else:
            if strip_num >= len(offsets): raise ValueError('Invalid strip number %d in "%s"' % strip_num, self.filename)
            self.seek(offsets[strip_num])
            return self.file.read(self.mtags[page]['StripByteCounts'][stip_num])
            
    def get_chunk(self, page=0):
        offsets = self.mtags[page]['StripOffsets']
        num_strips = len(offsets) if type(offsets) in (list, tuple) else 1
        return ''.join(self.get_strip(i, page) for i in range(num_strips))
        
    def __len__(self):
        return len(self.mtags)

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.read_page, range(len(self))[key])

        return self.read_page(key)
        
    def read_page(self, page=0):
        width = self.mtags[page]['ImageLength']
        height = self.mtags[page]['ImageHeight']
        bpp = self.mtags[page]['BitsPerSample']
        
        if self.bpp not in (8, 16):
            raise ValueError('This library only supports 8 or 16 bit grayscale TIFFs.')
        else:
            dtype = ('<' if self.little_endian else '>') + ('u1' if self.bpp == 8 else 'u2')
            
        
        buf = frombuffer(self.get_chunk(page), dtype).reshape((width, height))        
        
        sys_little_endian = True if sys.byteorder == 'little' else False
        if sys_little_endian != self.little_endian:
            #print 'swapping endedness...'
            buf = buf.newbyteorder().byteswap()
            #print buf.dtype
        
        return buf
            
    def unpack(self, format, value=None):
        s = struct.Struct(('<' if self.little_endian else '>') + format)
        if value is None:
            vals = s.unpack(self.file.read(s.size))
        else:
            vals = s.unpack(value[:s.size])
            
        if len(vals) == 1: return vals[0]
        else: return vals
        
    def pack(self, format, values):
        if type(values) not in (list, tuple): values = (values, )
        s = struct.Struct(('<' if self.little_endian else '>') + format)
        self.file.write(s.pack(*values))
        
    def close(self):
        self.file.close()
        
    def __del__(self):
        self.close()
        
    def __iter__(self):
        self._iter_current_frame = -1
        return self
    
    def next(self):
        self._iter_current_frame += 1
        if self._iter_current_frame >= len(self):
            raise StopIteration
        else:
            return self[self._iter_current_frame]
        