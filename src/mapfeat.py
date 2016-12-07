import numpy as np
import pandas as pd

fo_test = "../data/test.csv"
fo_train = "../data/train.csv"
data_test  = pd.read_csv(fo_test)
data_train = pd.read_csv(fo_train)
data_list = [data_test, data_train]
data = pd.concat(data_list)

ans = 0;
lendict = {}
for i in range(1, 33):
	ans += len(set(data['Col_'+str(i)]))
	lendict['Col_'+str(i)] = len(set(data['Col_'+str(i)]))
print ans
print lendict

cols = data.columns
listdict = {}
for col in cols:
	if lendict.has_key(col):
		print col,  lendict[col]
		listdict[col] = sorted(list(set(data[col])))
		print listdict[col]
print
data_train_dummy = pd.get_dummies(data_train)
cols = data_train_dummy.columns
cols = cols[1:]
print cols
#print data_train_dummy.ix[data_train_dummy.shape[0]-1]
'''
for i in range(10):#data_train_dummy.shape[0]+1
	#print data_train_dummy.ix[i]
	pos = 0;
	for col in cols:
		if lendict.has_key(col):
			print str(int(pos + listdict[col].index(data_train_dummy.ix[i][col])+1)) +':'+'1 ',
			pos += lendict[col]
		else:
			if data_train_dummy.ix[i][col] == 1.0:
				pos += 1
				print str(pos) +':' +'1 ',
			else:
				pos += 1
	print
'''

fo = open("../data/train.txt",'w')
for i in range(data_train_dummy.shape[0]):
	#print data_train_dummy.ix[i]
	pos = 0;
	for col in cols:
		if col == 'Score':
			#print str(int(data_train_dummy.ix[i][col])),
			fo.write(str(int(data_train_dummy.ix[i][col])))
		elif lendict.has_key(col):
			if data_train_dummy.ix[i][col] != 0.0:
				pos += 1
				#print str(pos)+':'+str(data_train_dummy.ix[i][col]),
				fo.write(' '+ str(pos)+':'+str(int(data_train_dummy.ix[i][col])))
			else:
				pos += 1
		else:
			if data_train_dummy.ix[i][col] == 1.0:
				pos += 1
				#print str(pos) +':' +'1 ',
				fo.write(' '+str(pos) +':' +'1')
			else:
				pos += 1
	#print
	fo.write('\n')
fo.close()


data_test_dummy = pd.get_dummies(data_test)
cols = data_test_dummy.columns
cols = cols[1:]
print cols

fo = open("../data/test.txt",'w')
for i in range(data_test_dummy.shape[0]):
	#print data_train_dummy.ix[i]
	pos = 0;
	for col in cols:
		if col == 'Score':
			#print str(int(data_train_dummy.ix[i][col])),
			fo.write(str(int(data_test_dummy.ix[i][col])))
		elif lendict.has_key(col):
			if data_test_dummy.ix[i][col] != 0.0:
				pos += 1
				#print str(pos)+':'+str(data_train_dummy.ix[i][col]),
				fo.write(' '+ str(pos)+':'+str(int(data_test_dummy.ix[i][col])))
			else :
				pos += 1
		else:
			if data_test_dummy.ix[i][col] == 1.0:
				pos += 1
				#print str(pos) +':' +'1 ',
				fo.write(' '+str(pos) +':' +'1')
			else:
				pos += 1
	#print
	fo.write('\n')
fo.close()
