#! /usr/bin/env python
'''
FEATURE EXTRACTION CODE TAKEN FROM pyAudioAnalysis LIBRARY
'''
from scipy.io import wavfile
import numpy as np
import sys
import matplotlib.pyplot as plt
from utilities import *
import csv
import os
#from pyAudioAnalysis import audioFeatureExtraction as af
#to do feature extraction on data without silence



def in_segment(x, seg):
	n_segs = len(seg)
	step = 2 if n_segs == 2 else 2
	for i in range(0, n_segs, step):
		if x >= seg[i] and x <= seg[i+1]:
			return True
	return False

feat_count = []
feat_files = []
for feat in range(1, len(sys.argv)): 
	fn = sys.argv[feat].split('/')[-2] #folder name to use as feature file name
	file_counter = 0
	write_d = []
	files_list = os.listdir(sys.argv[feat])
	#out_file = csv.writer(open('features of ' + sys.argv[feat] + '.csv', 'wb'), delimiter=',', quoting=csv.QUOTE_ALL)
	for fil in range(0, len(files_list)):
		sample_rate, data = wavfile.read(sys.argv[feat] + files_list[fil])
		dur_smp = 1.0/sample_rate
		data_no_silence = no_silence(data, sample_rate) #remove all silence part from audio
		
		features = feature_extraction(data_no_silence, sample_rate, 0.05*sample_rate, 0.025*sample_rate)
		#features = af.stFeatureExtraction(data_no_silence, sample_rate, 0.05*sample_rate, 0.025*sample_rate)
		
		
		write_d += (np.transpose(features).tolist())
		print str(np.shape(features)) + " -- " + fn + " -- " + str(file_counter)
		file_counter+=1
		
	feat_count.append(np.shape(write_d)[0])
	print '============================='
	print fn + " feature class => " + str(len(feat_count) - 1) + " -- count = " + str(np.shape(write_d)[0])
	np.savetxt('features-' + fn + '-crook.csv', write_d, delimiter=' , ')
	feat_files += write_d #feat_files will be used to save to a file with all classes
	feat+=1	

#create a class (category) list
class_list = []
for i in range(0, len(feat_count)):
	print i
	for j in range(len(class_list), len(class_list)+feat_count[i]):
		class_list.append(i)

#save class list to file
np.savetxt('feature-classes.csv', class_list, delimiter=' , ')
assert(np.shape(feat_files)[0] == sum(feat_count))
np.savetxt('final-features.csv', feat_files, delimiter=' , ')

