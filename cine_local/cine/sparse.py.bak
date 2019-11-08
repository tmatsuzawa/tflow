#!/usr/bin/env python
import struct
#import numpy as N
from numpy import array, count_nonzero, frombuffer, dtype, prod, zeros, clip, arange, linspace, fromstring
import os, sys
import time
import re
import multiprocessing
import glob
import ast

try:
    from scipy import weave
except:
    USE_C_UNPACK = False
else:
    USE_C_UNPACK = True

#USE_C_UNPACK = False

def make_block(a, sparse=None):
    n_dims = len(a.shape)
    
    if sparse is None:
        sparse = True if (count_nonzero(a) / float(a.size) < 0.75) else False
    
    s = a.tostring()
    if sparse: s = sparse_encode(s)    
    
    header = ''.join([
        '%-7s%c' % ('SPARSE' if sparse else 'ARRAY', a.dtype.char),
        struct.pack('3Q', 8 * (n_dims + 4), len(s), n_dims),
        struct.pack('%dQ' % n_dims, *a.shape)
    ])
    
    return header + s

def unpack_block(s):
    desc, header_size, data_size = struct.unpack('8s2Q', s[:24])
    extra_header = s[24:header_size]
    data = s[header_size:header_size+data_size]

    if desc.startswith('ARRAY') or desc.startswith('SPARSE'):
        dtype = desc[-1]
        ndims = struct.unpack('Q', extra_header[:8])[0]
        shape = struct.unpack('%dQ' % ndims, extra_header[8:])
        
        if desc.startswith('SPARSE'):
            return sparse_decode(data, shape, dtype)
        else:
            return fromstring(data, dtype=dtype).reshape(shape)
    else:
        raise ValueError('Unkown block type "%s" in %s' % (desc, self.file_desc))


class TextHeaderedBinary(object):
    post_comments = r"""
#Brief description:
#   first line is identifier string, "THB\n" by default, but should be changed for derived formats.
#   header is text "key: value" pairs with \n line endings
#       values are intepreted with Python's ast.literal_eval -- values that can be evaluated this way appear as strings
#   comments indicated with '#', MUST BEGIN LINE
#   text section ends with ">>>BEGIN BINARY>>>\n"
#   binary section: (pointers referenced from byte after final \n from text section)
#       max blocks <unsigned long> (for clarity, this is the location pointed to by pointer=0)
#       number of blocks <unsigned long> 
#       block pointer 0 <unsigned long> 
#       ...
#       block pointer N
#       [optional blank space if max blocks > number of blocks]
#       block length 0 <unsigned long> (total length does not include this unsigned long)
#       block 0 <block format, below>
#       ...
#       block length N
#       block N
#
#   block format:
#       8 descrtiption bytes:
#           ARRAY[two spaces][data type byte]
#           SPARSE[one space][data type byte]
#           (data type byte is as in numpy.dtype.char -> i.e. one of: bBhHiIlLfd)
#       total header length <unsigned long> (= 8 * (n_dims + 4))
#       data length <unsigned long>
#       number of dimensions <unsigned long>
#       shape[0] <unsigned long>
#       ...
#       shape[n=number of dimensions]
#       binary data (should correspond to data length)  
#           for ARRAY format, the data is raw binary.
#           for SPARSE, zeros are interpretted specially:
#              0x00 [X <unsigned byte>] -> a zero followed by a non zero byte decodes to X repeated zeros
#              0x00 0x00 [X <unsigned short>] -> two zero bytes followed by a unsigned short used to indicate more than 255 repeated zeros.  (Should be treated as little endian.)
"""

    file_id = "THB"
    file_desc = "text headered binary"
    
    def __init__(self, fn, read_write='r', header_dict=None, max_blocks=1024, cache_blocks=False, preload=False):
        if read_write not in ('a', 'r', 'w'):
            raise ValueError("read_write should be one of 'a', 'r', 'w'")
        
        self.fn = fn
        
        self.writeable = False if read_write == 'r' else True
        
        self.cache_blocks = cache_blocks
        self.block_cache = {}
        
        if read_write == 'w':
            self.f = open(fn, 'wb')
            
            self.header = {"creation time (unix)":repr(time.time()), "creation time":time.strftime("%Y %b %d %a %H:%M:%S %Z")}
            if header_dict is not None: self.header.update(header_dict) #Allow override of creation time
            for k in self.header.keys():
            #    self.header[k] = str(self.header[k]).strip().replace('\n', '\\n')
                self.header[k] = repr(self.header[k])
        
            self.f.write(self.file_id + '\n' + '#%s\n#Brief description below.\n\n' % self.file_desc)
            for k in sorted(self.header.keys()):
                self.f.write(k + ': ' + self.header[k] + '\n')
            self.f.write(self.post_comments + "\n>>>BEGIN BINARY>>>\n")
            self.zero_offset = self.f.tell()
            self.write_struct('Q', max_blocks)
            self.f.write('\x00' * 8 * (max_blocks + 1))
            self.max_blocks = max_blocks
            self.block_offsets = []
            
        else:
            self.f = open(fn, 'r+b' if read_write == 'a' else 'rb')
            first_line = self.read_header_line()
            if first_line != self.file_id:
                raise ValueError('First line of "%s" was "%s", should be "%s" for %s' %  (fn, first_line, self.file_id, self.file_desc))
            
            self.header = {}
            while True:
                line = self.read_header_line()
                if not line:
                    raise ValueError('Found EOF in %s before ">>>BEGIN BINARY>>>" -> file is invalid!' % fn)
                
                if line == ">>>BEGIN BINARY>>>":
                    break
                
                if ':' not in line:
                    raise ValueError("Non key:value line in header of '%s' -> file is invalid!" % fn)
                    
                key, value = map(str.strip, line.split(':', 1))
                #print value
                try:
                    value = ast.literal_eval(value)     
#                key, value = map(str.strip, line.split(':', 1))
                except:
                    if value.startswith('(') and value.endswith(')'):
                        value = tuple(map(str.strip, value[1:-1].split(',')))
                
                self.header[key] = value
            
            self.zero_offset = self.f.tell()    
            self.max_blocks, N = self.read_struct('2Q')
            self.block_offsets = list(self.read_struct('%dQ' % N))
            
            if preload:
                for i in range(len(self)):
                    self.block_cache[i] = self.get_raw_block(i)

            
    def append_array(self, a, sparse=None):
        self.append_block(make_block(a, sparse))
        #print header
        

    def append_block(self, block):
        if not self.writeable:
            raise ValueError("This is not a writeable file!")
        
        if len(self.block_offsets) >= self.max_blocks:
            raise RuntimeError("File has reached maximum number of blocks (%d), recreate with higher max_blocks if you need more!" % self.max_blocks) 
        i = len(block)
        self.f.flush()
        self.f.seek(0, os.SEEK_END)
        offset = self.f.tell() - self.zero_offset
        self.write_struct('Q', i)
        self.f.write(block)
        
        self.f.seek(self.zero_offset + 8)
        self.block_offsets.append(offset)
        j = len(self.block_offsets)
        self.write_struct('Q', j)
        self.f.seek(self.zero_offset + 8 + 8 * j)
        self.write_struct('Q', offset)
        
    def get_raw_block(self, block_num):
        if block_num >= len(self.block_offsets):
            raise ValueError("Block number %d does not exist" % block_num)
        self.f.seek(self.zero_offset + self.block_offsets[block_num])
        
        i = self.read_struct('Q')[0]
        block =  self.f.read(i)
        desc, header_size, data_size = struct.unpack('8s2Q', block[:24])
        extra_header = block[24:header_size]
        data = block[header_size:header_size+data_size]

        return (desc, extra_header, data)

    
    def unpack_block(self, desc, extra_header, data):
        if desc.startswith('ARRAY') or desc.startswith('SPARSE'):
            dtype = desc[-1]
            ndims = struct.unpack('Q', extra_header[:8])[0]
            shape = struct.unpack('%dQ' % ndims, extra_header[8:])
            
            if desc.startswith('SPARSE'):
                return sparse_decode(data, shape, dtype)
            else:
                return fromstring(data, dtype=dtype).reshape(shape)
        else:
            raise ValueError('Unkown block type "%s" in %s' % (desc, self.file_desc))
    
                
    def read_block(self, block_num):
        if self.cache_blocks:
            if block_num not in self.block_cache:
                self.block_cache[block_num] = self.get_raw_block(block_num)
            return self.unpack_block(*self.block_cache[block_num])
        else:
            return self.unpack_block(*self.get_raw_block(block_num))
        
        
    def __len__(self):
        return len(self.block_offsets)
        

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.read_block, range(len(self))[key])
        else:
            return self.read_block(key)


    def write_struct(self, fmt, *args):
        self.f.write(struct.pack(fmt, *args))

    def read_struct(self, fmt):
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))
            
    def read_header_line(self):
        #Reads a non-empty/comment only line from the header.
        #Returns None if EOF is encountered
        
        line = "" 
 
        while len(line) == 0 or line[0] == '#':
            l = []
        
            #Because this is a quasi-binary format, I'm a little nervous about line endings...
            while True:
                c = self.f.read(1)
                if c == '\n': break
                elif not c: return None
                else: l.append(c)
            
            line = ''.join(l)
            
            #i = line.find('#')
            #if i >= 0: line = line[:i]
            line = line.strip()
        
        return line
    
    def close(self):
        self.f.flush()
        self.f.close()

#Alias
Sparse = TextHeaderedBinary


class Sparse4D(TextHeaderedBinary):
    file_id = 'S4D'
    file_desc = 'Sparse 4D Array'        
    

pack_re = re.compile('\x00{1,65534}', flags=re.S) #Should be 65535 = 2^16-1, but this causes a regex overflow!
#pack_re = re.compile('\x00{1,255}', flags=re.S) #Debug
unpack_re = re.compile('(\x00\x00..|\x00.)', flags=re.S)
lots_o_zeros = chr(0) * 65536

def unpack_func(m):
    s = m.group(0)    
    if len(s) == 2:
        n = struct.unpack('B', s[1:])[0]
    else:
        n = struct.unpack('H', s[2:])[0]
    return lots_o_zeros[:n]   

def pack_func(m):
    global longest_run
    n = len(m.group(0))
    if (n <= 255):
        return struct.pack('BB', 0, n)
    else:
        return struct.pack('HH', 0, n)


def c_unpack(s, array_size, dt):
    global USE_C_UNPACK

    dt = dtype(dt)
    array_size = prod(array_size) #Shall be 1D, for simplicity
    x = zeros(array_size, dtype=dt)
    item_size = dt.itemsize
    output_buffer_size = int(item_size * array_size)
    input_buffer = fromstring(s, dtype='u1')
    input_buffer_len = len(input_buffer)
    
    code = r'''
        //We're going to treat the output buffer as unsigned chars, no matter what.
        unsigned char *output_buffer = (unsigned char *) x;
        int i = 0, j = 0;

        while (i < input_buffer_len) {
            if (j >= output_buffer_size) {
                return_val = j;
                break;
            }
 
            if (input_buffer[i] == 0) {
                i ++;
                if (input_buffer[i] == 0) {
                    i ++;
                    j += input_buffer[i] + input_buffer[i+1] * 256;
                    i += 2;
                } else {
                    //printf("%d \n", i);
                    j += input_buffer[i];
                    i ++;
                }
            } else {
                output_buffer[j] = input_buffer[i];
                i ++;
                j ++;
            }
        }
        
        if (j != output_buffer_size) {return_val = j;}
        else {return_val = 0;}
        
'''
    c_vars = ['input_buffer', 'input_buffer_len', 'x', 'output_buffer_size']

    return_val = weave.inline(code, c_vars)
    if return_val: print "WARNING: C unpack returned %d, data may be corrupted!" % return_val


    return x
    
def sparse_encode(s):
    return pack_re.sub(pack_func, s)
    
def sparse_decode(s, shape, dtype):     
    global USE_C_UNPACK

    if USE_C_UNPACK:
        try:
            a = c_unpack(s, shape, dtype)
        except:
            print "WARNING: WEAVE INSTALLED, BUT CAN'T COMPILE CODE!"
            USE_C_UNPACK = False
            a = fromstring(unpack_re.sub(unpack_func, s), dtype=dtype)

    else:
        a = fromstring(unpack_re.sub(unpack_func, s), dtype=dtype)
    a.shape = shape
    return a


def if_int(str):
    if not str: return None
    else: return int(str)

def eval_slice(s, N): 
    return range(*slice(*[if_int(x) for x in s.split(':')]).indices(N))
        

if __name__ == '__main__':
    import argparse, cine
    run_test = False

    parser = argparse.ArgumentParser(description='Convert a 4D image to S4D')
    parser.add_argument('input', metavar='input', type=str, nargs='+',help='input files (cine or tiff)')
    parser.add_argument('-r', dest='range', type=str, default=":", help='range of frames to convert, in python slice format [:]')
    parser.add_argument('-d', dest='depth', type=int, default=-1, help='stacks per volume')
    parser.add_argument('-M', dest='max_val', type=int, default=1000, help='max val of rescale (top clip)')
    parser.add_argument('-m', dest='min_val', type=int, default=45, help='min val of rescale (bottom clip)')
    parser.add_argument('-D', dest='displayframes', type=str, default=":", help='range of z frames to save in volume [:]')
#    parser.add_argument('-h', dest='histogram', type=bool, default=False, help='display a histogram and exit')
#    parser.add_argument('-o', dest='output', type=str, default="%s.s4d", help='output filename')
    parser.add_argument('-o', dest='output', type=str, default='%s', help='output filename, may use %%s for input filename w/o extension or %%d for input file number')
    parser.add_argument('-g', dest='gamma', type=float, default=1.0, help='gamma correction [1.0]')
    parser.add_argument('-s', dest='skip', type=int, default=0, help='skip this many frames in the file; used to fix sync offsets')
    
    args = parser.parse_args()
    
    inputs = []
    
    for input in args.input:
        if '*' in input or '?' in input: inputs += glob.glob(input)
        else: inputs.append(input)
    
    for i, input in enumerate(inputs):
        start = time.time()
        
        input_parts = input.split('_')

        base, ext = os.path.splitext(input)
        ext = ext.lower()
        
        output = args.output
        if '%s' in args.output: output = output % base
        elif '%' in args.output: output = output % i
        if not os.path.splitext(output)[1]: output += '.s4d'

        
        if args.depth == -1:
            fpv_part = filter(lambda x: x.lower().endswith('fpv'), input_parts)
            if fpv_part:
                frames_per_volume = int(fpv_part[0][:-3])
            else:
                raise ValueError("frames per volume (depth) must either be specified with '-d' or part of the file name (e.g.: 'XXX_100fpv_YYY')")
        else: frames_per_volume = args.depth
    
        #print frames_per_volume
        
        #display_range = slice(*[if_int(x) for x in args.displayframes.split(':')])
        #DATA = Image4D(args.input, args.depth, brighten=args.brightness, clip=args.clip, series=args.series, offset=args.offset, display_range=display_range)
        #output = args.output if args.output else os.path.splitext(args.input)[0] + '.s4d'
        
        if ext.lower().startswith('.tif'):
            sys.setrecursionlimit(10**5)
            source = cine.Tiff(input)    
        else:
            source = cine.Cine(input)
        
        max_frame = (len(source) - args.skip) // frames_per_volume
        saved_frames = eval_slice(args.range, max_frame)
    
        print '%30s -> %-30s (%d frames)...' % (input, output, len(saved_frames))
        sys.stdout.flush()
    
        #continue
    
        output = Sparse4D(output, 'w', {
            'source':input,
            'frames per volume': frames_per_volume,
            'min clip': args.min_val,
            'max clip': args.max_val,
            'frame size': source[0].shape,
            'gamma': args.gamma,
            'command': ' '.join(sys.argv),
            'skip': args.skip,
        })
        
    
        frame_offsets = array(eval_slice(args.displayframes, frames_per_volume))
    
        bpps = source[0].dtype.itemsize * 8
        m = 1. / (args.max_val - args.min_val)
        valmap = clip(-m * args.min_val + m * arange(2**bpps, dtype='f'), 0, 1)
        valmap = valmap ** (1./args.gamma)
        valmap = clip(valmap * 255, 0, 255).astype('u1')
    
        for frame_num in saved_frames:
            print frame_num,
            sys.stdout.flush()
            #frame = array([source[i] for i in (frame_offsets + frame_num * frames_per_volume + args.skip)], dtype='u4')
            #frame = ((clip(frame, args.min_val, args.max_val) - args.min_val)) * 255 // (args.max_val - args.min_val)            
            #if args.gamma != 1.0:
            #    frame = frame**(1./args.gamma) * 255**(1. - 1./args.gamma)
            #output.append_array(frame.astype('u1'))
            
            frame = array([source[i] for i in (frame_offsets + frame_num * frames_per_volume + args.skip)])
            output.append_array(valmap[frame])
            
    
        print '-> done in %.1fs' % (time.time() - start)
    
#    print frame.shape
    
    #FRAMES = range(*slice(*[if_int(x) for x in args.range.split(':')]).indices(len(DATA)))

    if run_test:        
        test = TextHeaderedBinary('test.thb', 'w', {'test':(1, 2, 3), 'another test':'abc'})
        test.append_array(arange(100)%5)
        test.append_array(linspace(0, 100, 11))
        test.append_array(arange(24).reshape(2, 3, 4))
        test.close()
        
        test = TextHeaderedBinary('test.thb', 'r')
        print test.header
        print test.read_block(0)
        print test.read_block(1)
        print test.read_block(2)