#!/usr/bin/env python
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.signal as sig
from util import *



if(len(sys.argv) < 2):
	print "ARGUMENT ERROR!"
	print sys.argv[0] + " [wav_file]"
	exit()

sample_rate, data = wavfile.read(sys.argv[1])
sample_duration = 1./sample_rate
duration = float(len(data)) * sample_duration
time_arr = np.arange(0, duration, sample_duration)
print "Duration = " + str(duration)

features = feature_extraction(data, sample_rate, sample_rate*0.05, sample_rate*0.025)

print "features shape = " + str(np.shape(features))

plt.subplot(711)
plt.plot(data)
plt.xlabel('Original Signal')
plt.subplot(712)
plt.plot(features[0,])
plt.xlabel('ZCR')
plt.subplot(713)
plt.plot(features[1,])
plt.xlabel('Energy')
plt.subplot(714)
plt.plot(features[5,])
plt.xlabel('Spectral Entropy')
plt.subplot(715)
#auto correlation of energy and zcr
f_autocorr = sig.correlate(features[0,], features[1,], mode='full')
half_autocorr = f_autocorr[len(f_autocorr)/2:]
plt.plot(f_autocorr) 
plt.xlabel('Autocorrelation Full')

plt.subplot(716)
plt.plot(half_autocorr) 
plt.xlabel('Autocorrelation half')

#convolution of zcr and energy
f_convolution = sig.convolve(features[0,], features[1,], mode='full')
plt.subplot(717)
plt.plot(f_convolution)

plt.show()