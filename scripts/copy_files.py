import argparse
import os, shutil

"""
A simple script to copy a directory using Python Shell Utilities

example: python copy_files.py /path/to/source /path/to/destination
"""

parser = argparse.ArgumentParser(description='Copy a directory to a given destination ')
parser.add_argument("src", help="a abs./relative path to a source directory")
parser.add_argument("dst", help="a abs./relative path to a destination directory")
parser.add_argument("-head", "--header", help="a header of a filename", default="" )
parser.add_argument("-foot", "--footer", help="a footer of a filename", default="")
parser.add_argument("-clip", "--clip", help="Number of letters clipped from the start of a filename", default=0, type=int)
parser.add_argument("-e", "--ext", help="Extension of a file that gets copied", default="")
parser.add_argument("-t", "--test", help="If provided, it copies only the first file", action='store_true')

args = vars(parser.parse_args())
src = args["src"]
dst = args["dst"]

def copy_dir(src="", dst="", header="", footer="", clip=0, ext="", test=False):
    """
    Copies a source directory to a destination directory
    Parameters
    ----------
    src: str, source directory
    dst: str, destination directory

    Returns
    -------
    failed: list, a list of file names that were not copied during the process
    """
    failed = []
    nfiles = 0
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not os.path.exists(src):
        raise argparse.ArgumentError("source does not exist! It must be a directory.")
    else:
        for root, dirs, files in os.walk(src, topdown=False):
            for name in files:
                name_wo_ext, file_ext = os.path.splitext(name)

                src_path = os.path.join(root, name)
                dstfilename = header + os.path.join(root[len(src)+1:], name_wo_ext[clip:]) + footer + file_ext
                dst_path = os.path.join(dst, dstfilename)

                dst_pdir = os.path.dirname(dst_path)
                if not os.path.exists(dst_pdir):
                    os.makedirs(dst_pdir)

                if not os.path.exists(dst_path):
                    if ext == "" or ext == file_ext[1:]:
                        try:
                            shutil.copy(src_path, dst_path)
                        except:
                            failed.append(src_path)
                            print(f"... {src_path} failed")
                else:
                    print(f"... {dst_path} already exists'. Skipping")
                nfiles += 1

                if test:
                    break
            if test:
                break
    print(f"{nfiles - len(failed)} / {nfiles} files were copied.")
    return failed

def main(**args):
    failed = copy_dir(**args)
    # If any of the copies failed, show the filenames in the end
    for i, fyle in enumerate(failed):
        print(i, fyle)

if __name__=="__main__":
    main(**args)
