#!/bin/sh
echo "Installing cine module and copying cine2tiff script to /usr/local/bin"
echo '''Make sure you add this to .bash_profile: export PATH="/usr/local/bin:$PATH"'''
echo "(Your password is required to access this directory.)"

python setup.py install
sudo cp cine2tiff.py /usr/local/bin/cine2tiff
sudo chmod a+x /usr/local/bin/cine2tiff

sudo cp cine2avi.py /usr/local/bin/cine2avi
sudo chmod a+x /usr/local/bin/cine2avi

sudo cp img2avi.py /usr/local/bin/img2avi
sudo chmod a+x /usr/local/bin/img2avi

sudo cp cine2sparse.py /usr/local/bin/cine2sparse
sudo chmod a+x /usr/local/bin/cine2sparse

sudo cp make_s4d.py /usr/local/bin/make_s4d
sudo chmod a+x /usr/local/bin/make_s4d

sudo cp multicine2avi.py /usr/local/bin/multicine2avi
sudo chmod a+x /usr/local/bin/multicine2avi

sudo cp cine2mp4.py /usr/local/bin/cine2mp4
sudo chmod a+x /usr/local/bin/cine2mp4

sudo cp Helvetica.ttf /usr/local/bin/Helvetica.ttf
sudo chmod a+r /usr/local/bin/Helvetica.ttf
