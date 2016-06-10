#! /usr/bin/env python
from scipy.io import wavfile
import numpy as np
import sys
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioFeatureExtraction as af
import csv
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
#to do feature extraction on data without silence

def in_segment(x, seg):
	n_segs = len(seg)
	step = 2 if n_segs == 2 else 2
	for i in range(0, n_segs, step):
		if x >= seg[i] and x <= seg[i+1]:
			return True
	return False

#load model
svm = LinearSVC()
svm = joblib.load('model/good_model.pkl')
write_d = []
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
			#print 'end'
		else:
			count = 0
print "done searching"			
data2 = []
data2.append(1)			
for i in range(0, len(data)):
	if not in_segment(i, segments):
		data2.append(data[i])
	#data2.append( 0.5 if in_segment(i, segments) else 0)
data_no_silence = np.asarray(data2)
print np.shape(data_no_silence)

features = af.stFeatureExtraction(data_no_silence, sample_rate, 0.05*sample_rate, 0.025*sample_rate)
res = svm.predict(features.transpose()).tolist()
male_confidence = (float(res.count(0))/len(res)) * 100
print 'male' if male_confidence > 50 else 'female'
print 'confidence = ' + str(male_confidence) if male_confidence > 50 else str(100 - male_confidence)