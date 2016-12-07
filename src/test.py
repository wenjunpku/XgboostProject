import numpy as np
import pandas as pd
from sklearn import svm
import math
fo_test = "../data/test.csv"
fo_train = "../data/train.csv"
data_test  = pd.read_csv(fo_test)
data_train = pd.read_csv(fo_train)
data_list = [data_test, data_train]
data = pd.concat(data_list)

data_train_dummy = pd.get_dummies(data_train)
data_test_dummy = pd.get_dummies(data_test)

def write_pred(filename, res,flag):
	fo = open( filename, 'w' )
	pos = 0
	for r in res:
		if flag == True:
			fo.write(str(40000+pos)+' ')
			fo.write(str(r))
			fo.write('\n')
			pos += 1
			fo.close()

target = 'Score'
IDcol = 'Id'
TRAIN = 32000
predictors = [x for x in data_train_dummy.columns if x not in [target,IDcol]]
data_train_X = data_train_dummy[predictors]
data_train_Y = data_train_dummy[target]
clf = svm.LinearSVR()
clf.fit(data_train_X[:TRAIN],data_train_Y[:TRAIN])
Pre_Y = clf.predict(data_train_X[TRAIN:])
print Pre_Y
print Pre_Y.shape
#write_pred('pred_svr.txt', Pre_Y, False)
score = clf.score(data_train_X[TRAIN:],data_train_Y[TRAIN:])
print score
ans = 0
length = len(Pre_Y)
for y1, y2 in zip(Pre_Y, data_train_Y[TRAIN:]):
	ans += (y1 - y2) * (y1 - y2)
print math.sqrt(ans/length)
