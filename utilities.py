#! /usr/bin/env python
from scipy.io import wavfile
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

def in_segment(x, seg):
	n_segs = len(seg)
	step = 2 if n_segs == 2 else 2
	for i in range(0, n_segs, step):
		if x >= seg[i] and x <= seg[i+1]:
			return True
	return False

def no_silence(data, sample_rate, plot=False):
	dur_smp = 1.0/sample_rate
	count = 0
	start = 0
	end = 0
	segments = []
	aud_segments = []
	s_avg = max(data)*0.05 if max(data) > abs(min(data)) else abs(min(data))*0.05
	data_no_silence = []
	data = list(data)
	
	for i in range(0, len(data)):
		if abs(data[i]) < s_avg:
			if count == 0:
				start = i
			count = count + 1
		else:
			if (count * dur_smp) > 0.1:
				segments.append(start)
				segments.append(i)
				count = 0
			else:
				count = 0
		if(i+1 == len(data)):
			segments.append(start)
			segments.append(i)

	if plot:
		for i in range(0, len(segments)):
			print segments[i]

	if plot:
		data2 = []
		data2.append(1)			
	
	data_no_silence = []
	for i in range(0, len(data)):
		if not in_segment(i, segments):
			data_no_silence.append(data[i])
		if plot:
			data2.append( 0.5 if in_segment(i, segments) else 0)

	if plot:
		plt.subplot(311)
		plt.plot(data, 'b')
		plt.subplot(312)
		plt.plot(data2, 'g')
		plt.subplot(313)
		plt.plot(data_no_silence, 'r')
		plt.show()

	return data_no_silence