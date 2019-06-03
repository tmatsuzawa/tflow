#!/usr/bin/env python
from numpy import *
import time

#Processing the data in chunks keeps it in the L2 catch of the processor, increasing speed for large arrays by ~50%
CHUNK_SIZE = 6 * 10**5 #Should be divisible by 3, 4 and 5!  This seems to be near-optimal.

def ten2sixteen(a):
    b = zeros(a.size//5*4, dtype='u2')
    
    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2, a3, a4) = [a[j+i:j+CHUNK_SIZE:5].astype('u2') for i in range(5)]
    
        k = j//5 * 4
        k2 = k + CHUNK_SIZE//5 * 4
    
        b[k+0:k2:4] = ((a0 & 0b11111111) << 2) + ((a1 & 0b11000000) >> 6)
        b[k+1:k2:4] = ((a1 & 0b00111111) << 4) + ((a2 & 0b11110000) >> 4)
        b[k+2:k2:4] = ((a2 & 0b00001111) << 6) + ((a3 & 0b11111100) >> 2)
        b[k+3:k2:4] = ((a3 & 0b00000011) << 8) + ((a4 & 0b11111111) >> 0)
    
    return b

def sixteen2ten(b):
    a = zeros(b.size//4*5, dtype='u1')
    
    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1, b2, b3) = [b[j+i:j+CHUNK_SIZE:4] for i in range(4)]
        
        k = j//4 * 5
        k2 = k + CHUNK_SIZE//4 * 5
        
        a[k+0:k2:5] =                              ((b0 & 0b1111111100) >> 2)
        a[k+1:k2:5] = ((b0 & 0b0000000011) << 6) + ((b1 & 0b1111110000) >> 4)
        a[k+2:k2:5] = ((b1 & 0b0000001111) << 4) + ((b2 & 0b1111000000) >> 6)
        a[k+3:k2:5] = ((b2 & 0b0000111111) << 2) + ((b3 & 0b1100000000) >> 8)
        a[k+4:k2:5] = ((b3 & 0b0011111111) << 0)  
    
    return a

def twelve2sixteen(a):
    b = zeros(a.size//3*2, dtype='u2')
    
    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2) = [a[j+i:j+CHUNK_SIZE:3].astype('u2') for i in range(3)]
    
        k = j//3 * 2
        k2 = k + CHUNK_SIZE//3 * 2
    
        b[k+0:k2:2] = ((a0 & 0xFF) << 4) + ((a1 & 0xF0) >> 4)
        b[k+1:k2:2] = ((a1 & 0x0F) << 8) + ((a2 & 0xFF) >> 0)
    
    return b

def sixteen2twelve(b):
    a = zeros(b.size//2*3, dtype='u1')
    
    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1) = [b[j+i:j+CHUNK_SIZE:2] for i in range(2)]
        
        k = j//2 * 3
        k2 = k + CHUNK_SIZE//2 * 3
        
        a[k+0:k2:3] =                       ((b0 & 0xFF0) >> 4)
        a[k+1:k2:3] = ((b0 & 0x00F) << 4) + ((b1 & 0xF00) >> 8)
        a[k+2:k2:3] = ((b1 & 0x0FF) << 0)
        
    return a

if __name__ == '__main__':
    #test = arange(0, 2**10, 2**6, dtype='u2')
    N = CHUNK_SIZE * 10
    
    test = random.randint(0, 2**10, N)

    start = time.time()
    tb = sixteen2ten(test)
    t = time.time() - start
    print '10 bit Encode: %.1f MP/s' % (len(test) / t / 1E6)
    
    start = time.time()
    test2 = ten2sixteen(tb)
    t = time.time() - start
    print '10 bit Decode: %.1f MP/s' % (len(test) / t / 1E6)
    
    
    print '10 bit Match: %s' % (test == test2).all()


    test = random.randint(0, 2**12, N)

    start = time.time()
    tb = sixteen2twelve(test)
    t = time.time() - start
    print '12 bit Encode: %.1f MP/s' % (len(test) / t / 1E6)
    
    start = time.time()
    test2 = twelve2sixteen(tb)
    t = time.time() - start
    print '12 bit Decode: %.1f MP/s' % (len(test) / t / 1E6)
    
    
    print '12 bit Match: %s' % (test == test2).all()

