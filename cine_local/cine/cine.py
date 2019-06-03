#!/usr/bin/env python

################################################################################

# Reader for CINE files produced by Vision Research Phantom Software
# Author: Dustin Kleckner
# dkleckner@uchicago.edu

################################################################################

import sys
import os
import time
import struct
import sparse
# import numpy as N
from numpy import array, frombuffer
# import Image
from threading import Lock
# import multiprocessing as M
import packed

import datetime
import hashlib

FRACTION_MASK = (2 ** 32 - 1)
MAX_INT = 2 ** 32

BYTE = 'B'
WORD = 'H'
INT16 = 'h'
SHORT = 'h'
BOOL = 'i'
DWORD = 'I'
UINT = 'I'
LONG = 'l'
INT = 'l'
FLOAT = 'f'
DOUBLE = 'd'
TIME64 = 'Q'
RECT = '4i'
WBGAIN = '2f'
IMFILTER = '28i'

TAGGED_FIELDS = {
    1000: ('ang_dig_sigs', ''),
    1001: ('image_time_total', TIME64),
    1002: ('image_time_only', TIME64),
    1003: ('exposure_only', DWORD),
    1004: ('range_data', ''),
    1005: ('binsig', ''),
    1006: ('anasig', ''),
    1007: ('undocumented', '')}  # 1007 exists in my files, but is not in documentation I can find

HEADER_FIELDS = [
    ('type', '2s'),
    ('header_size', WORD),
    ('compression', WORD),
    ('version', WORD),
    ('first_movie_image', LONG),
    ('total_image_count', DWORD),
    ('first_image_no', LONG),
    ('image_count', DWORD),
    ('off_image_header', DWORD),
    ('off_setup', DWORD),
    ('off_image_offsets', DWORD),
    ('trigger_time', TIME64),
]

BITMAP_INFO_FIELDS = [
    ('bi_size', DWORD),
    ('bi_width', LONG),
    ('bi_height', LONG),
    ('bi_planes', WORD),
    ('bi_bit_count', WORD),
    ('bi_compression', DWORD),
    ('bi_image_size', DWORD),
    ('bi_x_pels_per_meter', LONG),
    ('bi_y_pels_per_meter', LONG),
    ('bi_clr_used', DWORD),
    ('bi_clr_important', DWORD),
]

SETUP_FIELDS = [
                   ('frame_rate_16', WORD),
                   ('shutter_16', WORD),
                   ('post_trigger_16', WORD),
                   ('frame_delay_16', WORD),
                   ('aspect_ratio', WORD),
                   ('contrast_16', WORD),
                   ('bright_16', WORD),
                   ('rotate_16', BYTE),
                   ('time_annotation', BYTE),
                   ('trig_cine', BYTE),
                   ('trig_frame', BYTE),
                   ('shutter_on', BYTE),
                   ('description_old', '121s'),
                   # Guessed at length... because it isn't documented!  This seems to work.
                   ('mark', '2s'),
                   ('length', WORD),
                   ('binning', WORD),
                   ('sig_option', WORD),
                   ('bin_channels', SHORT),
                   ('samples_per_image', BYTE)] + \
               [('bin_name%d' % i, '11s') for i in range(8)] + [
                   ('ana_option', WORD),
                   ('ana_channels', SHORT),
                   ('res_6', BYTE),
                   ('ana_board', BYTE)] + \
               [('ch_option%d' % i, SHORT) for i in range(8)] + \
               [('ana_gain%d' % i, FLOAT) for i in range(8)] + \
               [('ana_unit%d' % i, '6s') for i in range(8)] + \
               [('ana_name%d' % i, '11s') for i in range(8)] + [
                   ('i_first_image', LONG),
                   ('dw_image_count', DWORD),
                   ('n_q_factor', SHORT),
                   ('w_cine_file_type', WORD)] + \
               [('sz_cine_path%d' % i, '65s') for i in range(4)] + [
                   ('b_mains_freq', WORD),
                   ('b_time_code', BYTE),
                   ('b_priority', BYTE),
                   ('w_leap_sec_dy', DOUBLE),
                   ('d_delay_tc', DOUBLE),
                   ('d_delay_pps', DOUBLE),
                   ('gen_bits', WORD),
                   ('res_1', INT16),  # Manual says INT, but this is clearly wrong!
                   ('res_2', INT16),
                   ('res_3', INT16),
                   ('im_width', WORD),
                   ('im_height', WORD),
                   ('edr_shutter_16', WORD),
                   ('serial', UINT),
                   ('saturation', INT),
                   ('res_5', BYTE),
                   ('auto_exposure', UINT),
                   ('b_flip_h', BOOL),
                   ('b_flip_v', BOOL),
                   ('grid', UINT),
                   ('frame_rate', UINT),
                   ('shutter', UINT),
                   ('edr_shutter', UINT),
                   ('post_trigger', UINT),
                   ('frame_delay', UINT),
                   ('b_enable_color', BOOL),
                   ('camera_version', UINT),
                   ('firmware_version', UINT),
                   ('software_version', UINT),
                   ('recording_time_zone', INT),
                   ('cfa', UINT),
                   ('bright', INT),
                   ('contrast', INT),
                   ('gamma', INT),
                   ('reserved1', UINT),
                   ('auto_exp_level', UINT),
                   ('auto_exp_speed', UINT),
                   ('auto_exp_rect', RECT),
                   ('wb_gain', '8f'),
                   ('rotate', INT),
                   ('wb_view', WBGAIN),
                   ('real_bpp', UINT),
                   ('conv_8_min', UINT),
                   ('conv_8_max', UINT),
                   ('filter_code', INT),
                   ('filter_param', INT),
                   ('uf', IMFILTER),
                   ('black_cal_sver', UINT),
                   ('white_cal_sver', UINT),
                   ('gray_cal_sver', UINT),
                   ('b_stamp_time', BOOL),
                   ('sound_dest', UINT),
                   ('frp_steps', UINT),
               ] + [('frp_img_nr%d' % i, INT) for i in range(16)] + \
               [('frp_rate%d' % i, UINT) for i in range(16)] + \
               [('frp_exp%d' % i, UINT) for i in range(16)] + [
                   ('mc_cnt', INT),
               ] + [('mc_percent%d' % i, FLOAT) for i in range(64)] + [
                   ('ci_calib', UINT),
                   ('calib_width', UINT),
                   ('calib_height', UINT),
                   ('calib_rate', UINT),
                   ('calib_exp', UINT),
                   ('calib_edr', UINT),
                   ('calib_temp', UINT),
               ] + [('header_serial%d' % i, UINT) for i in range(4)] + [
                   ('range_code', UINT),
                   ('range_size', UINT),
                   ('decimation', UINT),
                   ('master_serial', UINT),
                   ('sensor', UINT),
                   ('shutter_ns', UINT),
                   ('edr_shutter_ns', UINT),
                   ('frame_delay_ns', UINT),
                   ('im_pos_xacq', UINT),
                   ('im_pos_yacq', UINT),
                   ('im_width_acq', UINT),
                   ('im_height_acq', UINT),
                   ('description', '4096s')
               ]

T64_F = lambda x: int(x) / 2. ** 32
T64_F_ms = lambda x: '%.3f' % (float(x.rstrip('L')) / 2. ** 32)
T64_S = lambda s: lambda t: time.strftime(s, time.localtime(float(t.rstrip('L')) / 2. ** 32))


def fix_frame(f):
    do = f.dtype
    f = array(f, dtype='u4')
    f[2:300:4, 4::8] = ((f[3:301:4, 4::8] + f[1:299:4, 4::8] + f[2:300:4, 5::8] + f[2:300:4, 3::8])) // 4
    return array(f, dtype=do)


class Cine(object):
    def __init__(self, fn):
        self.f = open(fn, 'rb')
        self.fn = fn

        self.read_header(HEADER_FIELDS)
        self.read_header(BITMAP_INFO_FIELDS, self.off_image_header)
        self.read_header(SETUP_FIELDS, self.off_setup)
        self.image_locations = self.unpack('%dQ' % self.image_count, self.off_image_offsets)
        if type(self.image_locations) not in (list, tuple):
            self.image_locations = [self.image_locations]

        self.width = self.bi_width
        self.height = self.bi_height

        self.file_lock = Lock()  # Allows Cine object to be accessed from multiple threads!

        self._hash = None

    def unpack(self, fs, offset=None):
        if offset is not None:
            self.f.seek(offset)
        s = struct.Struct('<' + fs)
        vals = s.unpack(self.f.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals

    def read_tagged_blocks(self):
        '''
        Reads the tagged block meta-data from the header
        '''

        if not self.off_setup + self.length < self.off_image_offsets:
            return
        next_tag_exists = True
        next_tag_offset = 0
        while next_tag_exists:
            block_size, next_tag_exists = self._read_tag_block(next_tag_offset)
            next_tag_offset += block_size

    def _read_tag_block(self, off_set):
        '''
        Internal helper-function for reading the tagged blocks.
        '''
        self.file_lock.acquire()
        self.f.seek(self.off_setup + self.length + off_set)
        block_size = self.unpack(DWORD)
        b_type = self.unpack(WORD)
        more_tags = self.unpack(WORD)

        if b_type == 1004:
            # docs say to ignore range data
            # it seems to be a poison flag, if see this, give up tag parsing
            self.file_lock.release()
            return block_size, 0

        try:
            d_name, d_type = TAGGED_FIELDS[b_type]

        except KeyError:
            #            print 'unknown type, find an updated version of file spec', b_type
            self.file_lock.release()
            return block_size, more_tags

        if d_type == '':
            #            print "can't deal with  <" + d_name + "> tagged data"
            self.file_lock.release()
            return block_size, more_tags

        s_tmp = struct.Struct('<' + d_type)
        if (block_size - 8) % s_tmp.size != 0:
            #            print 'something is wrong with your data types'
            self.file_lock.release()
            return block_size, more_tags

        d_count = (block_size - 8) // (s_tmp.size)

        data = self.unpack('%d' % d_count + d_type)
        if not isinstance(data, tuple):
            # fix up data due to design choice in self.unpack
            data = (data,)

        # parse time
        if b_type == 1002 or b_type == 1001:
            data = [(datetime.datetime.fromtimestamp(d >> 32), float((FRACTION_MASK & d)) / MAX_INT) for d in data]
        # convert exposure to seconds
        if b_type == 1003:
            data = [float(d) / (MAX_INT) for d in data]

        setattr(self, d_name, data)

        self.file_lock.release()
        return block_size, more_tags

    def read_header(self, fields, offset=0):
        self.f.seek(offset)
        for name, format in fields:
            setattr(self, name, self.unpack(format))

            #       Old method couldn't deal with mult-entry fields properly
            #        names, formats = zip(*fields)
            #        formats = ''.join(formats)
            #        self.__dict__.update(zip(names, self.unpack(formats, offset)))

    def get_frame(self, number):
        self.file_lock.acquire()

        image_start = self.image_locations[number]
        annotation_size = self.unpack(DWORD, image_start)
        annotation = self.unpack('%db' % (annotation_size - 8))
        image_size = self.unpack(DWORD)

        # self.f.seek(image_start + annotation_size-8)
        data_type = 'u1' if self.bi_bit_count in (8, 24) else 'u2'

        # print image_size
        # print self.bi_height, self.bi_width
        actual_bits = image_size * 8 // (self.width * self.height)
        #        print actual_bits
        if actual_bits in (10, 12):
            data_type = 'u1'

        self.f.seek(image_start + annotation_size)

        frame = frombuffer(self.f.read(image_size), data_type)  # original
        # frame = frombuffer(self.f.read(image_size), 'int32')
        #        print len(frame)

        if (actual_bits == 10):
            frame = packed.ten2sixteen(frame)
        elif (actual_bits == 12):
            frame = packed.twelve2sixteen(frame)
        elif (actual_bits % 8):
            raise ValueError(
                'Data should be byte aligned, or 10 or 12 bit packed (appears to be %dbits/pixel?!)' % actual_bits)

        # print len(frame)

        frame = frame.reshape(self.height, self.width)[::-1]

        if actual_bits in (10, 12):
            frame = frame[::-1, :]  # Don't know why it works this way, but it does...
        ##I'm sure I did this for a reason, but I've forgotten what it was... probably something multithreaded (should be fixed by lock)
        #
        # for n in range(3):
        #     try:
        #         self.f.seek(image_start + annotation_size)
        #         frame = frombuffer(self.f.read(image_size), data_type).reshape(self.bi_height, self.bi_width)[::-1]
        #     except:
        #         print 'Failed to read frame from cine file... retrying...'
        #     else:
        #         break
        # else: raise RuntimeError('Failed reading from cine file after three tries.')
        self.file_lock.release()
        if getattr(self, 'auto_fix', False):
            return fix_frame(frame)
        else:
            return frame

    def __len__(self):
        return self.image_count

    len = __len__

    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.get_frame, range(self.image_count)[key])

        return self.get_frame(key)

    def get_time(self, i):
        '''Return the time of frame i in seconds.'''
        # return float(i) / self.frame_rate
        if not hasattr(self, 'image_time_only'):
            self.read_tagged_blocks()

        dt0, f0 = self.image_time_only[0]
        dt1, f1 = self.image_time_only[i]

        return (dt1 - dt0).seconds + f1 - f0

    def get_time_list(self):
        nframes = self.__len__()
        time = []
        for i in range(nframes):
            time.append(self.get_time(i))
        return time

    def get_fps(self):
        return self.frame_rate

    def enable_auto_fix(self):
        self.auto_fix = True

    def __iter__(self):
        self._iter_current_frame = -1
        return self

    def next(self):
        self._iter_current_frame += 1
        if self._iter_current_frame >= self.image_count:
            raise StopIteration
        else:
            return self.get_frame(self._iter_current_frame)

    def close(self):
        self.f.close()

    def __unicode__(self):
        return self.fn

    def __str__(self):
        return unicode(self).encode('utf-8')

    __repr__ = __unicode__

    @property
    def trigger_time_p(self):
        '''Returns the time of the trigger, tuple of (datatime_object, fraction_in_ns)'''
        return datetime.datetime.fromtimestamp(self.trigger_time >> 32), float(FRACTION_MASK & self.trigger_time) / (
        MAX_INT)

    @property
    def hash(self):
        if self._hash is None:
            self._hash_fun()
        return self._hash

    def __hash__(self):
        return int(self.hash, base=16)

    def _hash_fun(self):
        """
        generates the md5 hash of the header of the file.  Here the
        header is defined as everything before the first image starts.

        This includes all of the meta-data (including the plethora of
        time stamps) so this will be unique.
        """
        # get the file lock (so we don't screw up any other reads)
        self.file_lock.acquire()

        self.f.seek(0)
        max_loc = self.image_locations[0]
        md5 = hashlib.md5()

        chunk_size = 128 * md5.block_size
        chunk_count = (max_loc // chunk_size) + 1

        for j in range(chunk_count):
            md5.update(self.f.read(128 * md5.block_size))

        self._hash = md5.hexdigest()

        self.file_lock.release()

    def __eq__(self, other):
        return self.hash == other.hash

    def __ne__(self, other):
        return not self == other
