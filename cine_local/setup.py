#!/usr/bin/env python
import cine.svn_info
# from distutils.core import setup

from setuptools import setup

print "Installing cine tools revision %s, dated %s." % (cine.svn_info.revision, cine.svn_info.date)

setup(
    name='cine',
    version=cine.svn_info.revision,
    author='Dustin Kleckner',
    author_email='dkleckner@uchicago.edu',
    packages=['cine'],
    )