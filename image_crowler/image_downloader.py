import urllib
import urllib2
import sys
import os.path
import socket
import urlparse
import urllib2
from urllib2 import Request
import base64

socket.setdefaulttimeout(10)

dir = sys.argv[2]
if not os.path.exists(dir):
    os.makedirs(dir)

for url in open(sys.argv[1], 'r'):
        url = url.replace('\n', '')
        url = urllib.unquote(url).decode('utf8') 
        name = os.path.basename(url)
        filename = dir +'/' + name
        print 'processing ' + name + ' =>  ' + url
        try:
                if(not os.path.isfile(filename)):
                        filehandle =urllib2.Request(url)
                        #base64string = base64.encodestring('%s:%s' % ('', '')).replace('\n', '')
                        #filehandle.add_header("Authorization", "Basic %s" % base64string)
                        result = urllib2.urlopen(filehandle)
                        fi=open(filename,"w+")
                        fi.write(result.read())
                        fi.close()
        except Exception as inst:
                print inst
                try:
                        os.remove(filename)
                        urllib.urlretrieve (url, filename)
                except Exception as inst:
                        print inst
