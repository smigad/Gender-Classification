#!/usr/bin/env python
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import numpy as np 
import sys

if len(sys.argv) != 3:
	print "Error command"
	print "		python "+ sys.argv[0]  +" [feature_file] [classes_file]"
	exit()

features = np.loadtxt(sys.argv[1], delimiter=' , ')
classes = np.loadtxt(sys.argv[2], delimiter=' , ')

num_zero = classes.tolist().count(0)
num_one = classes.tolist().count(1)

svm = LinearSVC(verbose=1)
svm.fit(features, classes)
#joblib.dump(svm, 'test/test_model.pkl')
pickle.dump(svm, open("test/model_pickle.pkl", 'w'))
new_zero = svm.predict(features).tolist().count(0)
new_one = svm.predict(features).tolist().count(1)

print "new_zero = " + str(new_zero)
print "new one = " + str(new_one)

print "0 prediction off by ==> " + str(abs(num_zero-new_zero))
print "1 prediction off by ==> " + str(abs(num_one-new_one))
