#!/bin/sh
echo "Installing cine module and copying cine2tiff script to /usr/bin"
echo "(Your password is required to access this directory.)"

python setup.py install
sudo cp cine2tiff.py /usr/bin/cine2tiff
sudo chmod a+x /usr/bin/cine2tiff

sudo cp cine2avi.py /usr/bin/cine2avi
sudo chmod a+x /usr/bin/cine2avi

sudo cp img2avi.py /usr/bin/img2avi
sudo chmod a+x /usr/bin/img2avi

sudo cp cine2sparse.py /usr/bin/cine2sparse
sudo chmod a+x /usr/bin/cine2sparse

sudo cp make_s4d.py /usr/bin/make_s4d
sudo chmod a+x /usr/bin/make_s4d

sudo cp multicine2avi.py /usr/bin/multicine2avi
sudo chmod a+x /usr/bin/multicine2avi

sudo cp cine2mp4.py /usr/bin/cine2mp4
sudo chmod a+x /usr/bin/cine2mp4

sudo cp Helvetica.ttf /usr/bin/Helvetica.ttf
sudo chmod a+r /usr/bin/Helvetica.ttf
