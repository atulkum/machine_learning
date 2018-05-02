import shutil

freq = {}
fcount = {}
with open('train.csv') as f:
	isfirst = True
	for l in f.readlines():
		if isfirst:
			isfirst = False
			continue
		m = l.rstrip('\r\n').split(',')
		ml =  m[1].split()
		
		fc = len(ml)
		if fc == 0:
			print m[0]
		if fc in fcount:
			fcount[fc] += 1
		else:
			fcount[fc] = 1
		for ll in ml:
			if ll in freq:
				freq[ll] += 1
			else:
				freq[ll] = 1
				

print freq		
print fcount
from collections import Counter

biz = {}
with open('train_photo_to_biz_ids.csv') as f:
        isfirst = True
        for l in f.readlines():
                if isfirst:
                        isfirst = False
                        continue
                m = l.rstrip('\r\n').split(',')
		if m[1] in biz:
			biz[m[1]] += 1
		else:
			biz[m[1]] = 1

	

print Counter(biz.values())
print Counter(biz.values()).most_common()
                


