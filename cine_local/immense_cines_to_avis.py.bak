#!/usr/bin/env python
import os, sys, time, commands

log_name = os.path.splitext(sys.argv[0])[0] + time.strftime('_log_%Y_%m_%d_%a.txt')
                                                            
print 'Saving log to:', log_name

#basepath = '/Volumes/labshared'

basepath = '/Volumes/labshared/'
time_limit = 12 * 3600
count_limit = 1000
size_limit = 4 * 2**30

count = 0
start = time.time()

log = open(log_name, 'at')

def crawl(path):
    global count, start
    fns = os.listdir(path)
    
    for fn in fns:
        ffn = os.path.join(path, fn)
        bfn, ext = os.path.splitext(fn)
        
        if os.path.isdir(ffn):
            crawl(ffn)
            
        if ext.lower() == '.cine':
            if bfn + '.avi' in fns:
                print '--- %s -> already converted ---' % fn
                
            elif os.path.getsize(ffn) > size_limit:
                print '--- %s -> too big ---' % fn
            
            else:
                if time.time() - start > time_limit:
                    print "Over time limit (%.1f hours)... exiting." % (time_limit / 3600.)
                    sys.exit()
                if count >= count_limit:
                    print "Over count limit (%d)... exiting." % count_limit
                    sys.exit()
                
                count += 1


                cmd = 'cine2avi -c1E-3 "%s"' % ffn
                logstr = '--- ' + time.strftime('%Y/%m/%d %H:%M:%S -> ') + cmd + ' ---'
                print logstr
                log.write(logstr + '\n')
                
                status, output = commands.getstatusoutput(cmd)
                
                logstr = output + '\n --- status: %s ---\n' % status 

                print logstr
                log.write(logstr + '\n')
                
                
            
crawl(basepath)
        
        
