import subprocess
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import re
import argparse
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''Module with functions for making movies
'''
# Abs path to ffmpeg
path_mod = os.path.abspath(__file__)
moddirpath = os.path.dirname(path_mod)
ffmpeg_path = os.path.join(moddirpath, 'ffmpeg')


def make_movie_noah(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=False,
               save_into_subdir=False, start_number=0, framestep=1):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    Parameters
    ----------
    imgname : str
        path and filename for the images to turn into a movie
    movname : str
        path and filename for output movie
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    imgdir : str or None
        folder to delete if rm_images and save_into_subdir are both True, ie folder containing the images
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None
    """
    subprocess.call(
        ['/Users/stephane/Documents/git/takumi/library/image_processing/ffmpeg',
         '-framerate', str(int(framerate)),
         '-start_number', str(start_number),
         '-i', imgname + '%' + indexsz + 'd.png',
         movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    # Delete the original images
    if rm_images:
        print('Deleting the original images...')
        if save_into_subdir and imgdir is not None:
            print('Deleting folder ' + imgdir)
            subprocess.call(['rm', '-r', imgdir])
        else:
            print('Deleting folder contents ' + imgdir + imgname + '*.png')
            subprocess.call(['rm', '-r', imgdir + imgname + '*.png'])


def make_movie(imgname=None, imgdir=None, movname=None, indexsz='05', framerate=10, rm_images=False,
               save_into_subdir=False, start_number=0, framestep=1, ext='png', option='normal', overwrite=False,
               invert=False, add_commands=[]):
    """Create a movie from a sequence of images using the ffmpeg supplied with ilpm.
    Options allow for deleting folder automatically after making movie.
    Will run './ffmpeg', '-framerate', str(int(framerate)), '-i', imgname + '%' + indexsz + 'd.png', movname + '.mov',
         '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

    ... ffmpeg is not smart enough to recognize a pattern like 0, 50, 100, 150... etc.
        It tries up to an interval of 4. So 0, 3, 6, 9 would work, but this hinders practicality.
        Use the glob feature in that case. i.e. option='glob'

    Parameters
    ----------
    imgname : str
        ... path and filename for the images to turn into a movie
        ... could be a name of directory where images are stored if option is 'glob'
    movname : str
        path and filename for output movie (movie name)
    indexsz : str
        string specifier for the number of indices at the end of each image (ie 'file_000.png' would merit '03')
    framerate : int (float may be allowed)
        The frame rate at which to write the movie
    rm_images : bool
        Remove the images from disk after writing to movie
    save_into_subdir : bool
        The images are saved into a folder which can be deleted after writing to a movie, if rm_images is True and
        imgdir is not None
    option: str
        If "glob", it globs all images with the extention in the directory.
        Therefore, the images does not have to be numbered.
    add_commands: list
        A list to add extra commands for ffmpeg. The list will be added before output name
        i.e. ffmpeg -i images command add_commands movie_name
        exmaple: add_commands=['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']
    """
    # if movie name is not given, name it as same as the name of the img directory
    if movname is None:
        if os.path.isdir(imgname):
            if imgname[-1] == '/':
                movname = imgname[:-1]
            else:
                movname = imgname
        else:
            pdir, filename = os.path.split(imgname)
            movname = pdir


    if not option=='glob':
        command = [ffmpeg_path,
                   '-framerate', str(int(framerate)),
                   '-start_number', str(start_number),
                   '-i', imgname + '%' + indexsz + 'd.' + ext,
                   '-pix_fmt', 'yuv420p',
                   '-vcodec', 'libx264', '-profile:v', 'main', '-crf', '12', '-threads', '0', '-r', '100']
    else:
        # If images are not numbered or not labeled in a sequence, you can use the glob feature.
        # On command line,
        # ffmpeg -r 1
        # -pattern_type glob
        # -i '/Users/stephane/Documents/git/takumi/library/image_processing/images2/*.png'  ## It is CRITICAL to include '' on the command line!!!!!
        # -vcodec libx264 -crf 25  -pix_fmt yuv420p /Users/stephane/Documents/git/takumi/library/image_processing/images2/sample.mp4
        command = [ffmpeg_path,
                 '-pattern_type', 'glob',  # Use glob feature
                 '-framerate', str(int(framerate)),  # framerate
                 '-i', imgname + '/*.' + ext,  # images
                 '-vcodec', 'libx264',  # codec
                 '-crf', '12',  # quality
                 '-pix_fmt', 'yuv420p']
    if overwrite:
        command.append('-y')
    if invert:
        command.append('-vf')
        command.append('negate')
    # check if image has dimensions divisibly by 2 (if not ffmpeg raises an error... why ffmpeg...)
    # ffmpeg raises an error if image has dimension indivisible by 2. Always make sure that this is not the case.
    # image_paths = glob.glob(imgname + '/*.' + ext)
    # img = mpimg.imread(image_paths[0])
    # height, width = img.shape
    # if not (height % 2 == 0 and width % 2 == 0):
    command += ['-vf', ' pad=ceil(iw/2)*2:ceil(ih/2)*2']


    print(command)
    command += add_commands

    command.append(movname + '.mp4')
    subprocess.call(command)

    # Delete the original images
    if rm_images:
        print('Deleting the original images...')
        if not save_into_subdir and imgdir is None:
            imdir = os.path.split(imgname)
        print('Deleting folder ' + imgdir)
        subprocess.call(['rm', '-r', imgdir])


def make_movie_with_time_stamp(imgname, movname, indexsz='05', framerate=10, imgdir=None, rm_images=True,
               save_into_subdir=False, start_number=0, framestep=1, ext='png', option='normal',
                               start=0, timestep=1.0, timelist=None, unit='s', timestamp_loc='br', color='white',
                               font='Arial', fontsize=15, alpha=0):
    def get_std_pos(width, height, loc='br', npartition=10):
        """
        Returns standard positions of an image (top right, center left, bottom center, etc.)
        ... PIL uses a convention where (0, 0) is the upper left corner.
        Parameters
        ----------
        width: int, width of image in px
        height: int, height of image in px
        loc: str, Location of timestamp. Choose from (bl, bc, br, cl, cc, cr, tl, tc, tr)
        npartition: int, (NOT currently used) Can be used to set extra location options.
                    dx, dy below can be used as increments in x and y.

        Returns
        -------
        pos: tuple, coordinate where a text wi

        """
        left, right = 0.025 * width, 0.75 * width
        bottom, top = 0.90 * height, 0.1 * height
        xcenter, ycenter =  width / 2., height / 2.
        dx, dy = width / npartition, height / npartition
        if loc == 'bl':
            pos = (left, bottom)
        elif loc == 'bc':
            pos = (xcenter, bottom)
        elif loc == 'br':
            pos = (right, bottom)
        elif loc == 'cl':
            pos = (left, ycenter)
        elif loc == 'cc':
            pos = (xcenter, ycenter)
        elif loc == 'cr':
            pos = (right, ycenter)
        elif loc == 'tl':
            pos = (left, top)
        elif loc == 'tc':
            pos = (xcenter, top)
        elif loc == 'tr':
            pos = (right, top)
        else:
            print('Location of timestamp was not understood! Choose from (bl, bc, br, cl, cc, cr, tl, tc, tr)')
            raise RuntimeError
        return pos

    def atoi(text):
        'human sorting'
        return int(text) if text.isdigit() else text
    def natural_keys(text):
        '''
        human sorting
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [atoi(c) for c in re.split('(\d+)', text)]

    # Find images
    imgs = glob.glob(imgname + '*.' + ext)
    # print [img[-7:] for img in imgs]

    imgs = sorted(imgs, key=natural_keys)
    # print [img[-7:] for img in imgs]
    if len(imgs)==0:
        print('No images found! Exiting...')
        raise RuntimeError

    # File architecture
    parentdir, imgname= os.path.split(imgs[0])
    tmpdir = os.path.join(parentdir, 'tmp_img')
    # make a tmp dir directory
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)


    # time stamp stuff
    # ... make a time list with a constant timestep unless it is given
    if timelist is None:
        timelist = np.arange(start, start + timestep * len(imgs), timestep)
    # ... font of time stamp
    if os.path.exists('/Library/Fonts/' + font +'.ttf'):
        fnt = ImageFont.truetype('/Library/Fonts/' + font +'.ttf', fontsize)
    elif os.path.exists('/Library/Fonts/' + font +'.ttc'):
        fnt = ImageFont.truetype('/Library/Fonts/' + font +'.ttc', fontsize)
    else:
        print('... Specified was not found under /Library/Fonts/')
        print('... Proceed with /Library/Fonts/Arial.ttf')
        fnt = ImageFont.truetype('/Library/Fonts/' + font + '.ttf')

    for i in range(len(imgs)):
        parentdir, imgname = os.path.split(imgs[i])

        img = Image.open(imgs[i])
        img = img.convert('RGB')
        width, height = img.size
        txt_pos = get_std_pos(width, height, loc=timestamp_loc)
        timestamp = 't={0:0.4f}'.format(timelist[i]) + unit

        draw = ImageDraw.Draw(img)
        try:
            red, blue, green = ImageColor.getrgb(color)
        except ValueError:
            print('... Color string was not understood.')
            print('... Use white as a default')
            red, blue, green = ImageColor.getrgb('w')
        draw.text(txt_pos, timestamp, font=fnt, fill=(red, blue, green, int(255*alpha)))
        outputfile = os.path.join(tmpdir, 'tmp' + '_{0:05d}.'.format(i) + ext)
        img.save(outputfile)

    make_movie(os.path.join(tmpdir, 'tmp_'), movname, indexsz='05', framerate=framerate, rm_images=rm_images,
               save_into_subdir=save_into_subdir, start_number=start_number, framestep=framestep, ext=ext, option=option)

def command_ffmpeg(command):
    """
    command must be a list
    Parameters
    ----------
    command

    Returns
    -------

    """
    command0 = [ffmpeg_path]
    command = command0 + command
    subprocess.call(command)
    print(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make an avi using images in a specified directory using ffmpeg')
    parser.add_argument('-imgdir', '--imgdir', help='Directory where images are stored', type=str, default=None)
    parser.add_argument('-imgtype', '--imgtype', help='Extension of images in the directory. default: png', type=str, default='png')
    parser.add_argument('-framerate', '--framerate', help='Frame rate default: 3', type=int, default=3)
    args = parser.parse_args()

    if args.imgdir is None:
        print('SPECIFY a directory where images are stored! Exiting...')
        sys.exit(1)
    else:
        print('make a movie with glob option using a natural sorting...')
        make_movie(args.imgdir + '/', args.imgdir, framerate=args.framerate, rm_images=False, ext=args.imgtype, option='glob')