#! /usr/bin/env python
from scipy.io import wavfile
import numpy as np
import sys
import matplotlib.pyplot as plt



def in_segment(x, seg):
	n_segs = len(seg)
	step = 2 if n_segs == 2 else 2
	for i in range(0, n_segs, step):
		if x >= seg[i] and x <= seg[i+1]:
			return True
	return False


#if __name__ == '__main__':
sample_rate, data = wavfile.read(sys.argv[1])
dur_smp = 1.0/sample_rate
count = 0
start = 0
end = 0
segments = []
s_avg = max(data)*0.1 if max(data) > abs(min(data)) else abs(min(data))*0.1
	
for i in range(0, len(data)):
	if abs(data[i]) < s_avg:
		if count == 0:
			start = i
		count = count + 1
		#print "found  " + str(data[i]) + " --- " + str(start)
	else:
		if (count * dur_smp) > 0.1:
			segments.append(start)
			segments.append(i)
			count = 0
			print 'end'
		else:
			count = 0
print "done seraching"			
data2 = []
data2.append(1)			
for i in range(0, len(data)):
	if not in_segment(i, segments):
		data2.append(data[i])
	#data2.append( 0.5 if in_segment(i, segments) else 0)
write_data = np.asarray(data2)
print np.shape(write_data)
wavfile.write('new_boom.wav', sample_rate, write_data)
plt.subplot(211)
plt.plot(data)
plt.subplot(212)
plt.plot(data2)
plt.show()
