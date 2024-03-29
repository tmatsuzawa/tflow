#!/usr/bin/env python
import sys, os, glob

FIRST_LINE = '@setlocal enableextensions & python -x "%~f0" %* & goto :EOF\r\n'
LINE_CHECK = '@setlocal enableextensions & python'
OUTPUT_DIR = 'C:\Windows\System32'

fns = []
for x in sys.argv[1:]:
    if '*' in x or '?' in x: fns += glob.glob(x)
    else: fns.append(x)

#print fns

for fn in fns:
    if not os.path.exists(fn):
        print("! '%s' does not exist, ignoring!" % fn)
        continue
    
    ofn = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(fn))[0] + '.cmd')
#    print ofn
    if os.path.exists(ofn):
        print("! '%s' already exists!" % ofn)
        if open(ofn).read(len(LINE_CHECK)) == LINE_CHECK:
            print("  ...but it was apparently generated by this script, and will be replaced.")
        else:
            print("  Delete it and rerun to write this file.")
            continue

    f = open(ofn, 'wt')
    f.write(FIRST_LINE)
    f.write(open(fn, 'r').read())
    f.close()
    print("'%s' -> '%s'" % (fn, ofn))
