#!/usr/bin/env python

# SVN Keywords
revision = "$Revision: 14 $"[11:-1].strip()
id = "$Id: svn_info.py 14 2012-04-23 22:26:08Z dustin $"[4:-1].strip()
author = "$Author: dustin $"[9:-1].strip()
date = "$Date: 2012-04-23 17:26:08 -0500 (Mon, 23 Apr 2012) $"[7:-1].strip()
headurl = "$HeadURL: svn://immenseirvine.uchicago.edu/cine/cine/svn_info.py $"[10:-1].strip()


#To add SVN keywords use: svn propset svn:keywords "Revision Id Author Date HeadURL"  svn_info.py
