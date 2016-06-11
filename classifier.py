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
from utilities import *
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
data_no_silence = no_silence(data, sample_rate)
features = af.stFeatureExtraction(data_no_silence, sample_rate, 0.05*sample_rate, 0.025*sample_rate)
res = svm.predict(features.transpose()).tolist()
male_confidence = (float(res.count(0))/len(res)) * 100
print 'male' if male_confidence > 50 else 'female'
print 'confidence = ' + str(male_confidence) if male_confidence > 50 else str(100 - male_confidence)