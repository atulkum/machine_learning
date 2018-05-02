import sys

prefix = '/Users/atulkumar/ml/yelp/bottleneck/'+ sys.argv[1] +'/'
suffix = '.jpg.txt'

features = open(sys.argv[3],'w')
with open(sys.argv[2]) as f:
        isfirst = True
        for l in f.readlines():
                if isfirst:
                        isfirst = False
                        continue
                m = l.rstrip('\r\n').split(',')
		image_file = prefix + m[0] + suffix
		try:
			with open(image_file) as imf:
				features.write(m[0] + ',' + imf.read() + '\n')
		except:
			print "Unexpected error:", sys.exc_info()
			pass

features.close()
