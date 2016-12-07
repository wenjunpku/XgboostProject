#!/usr/bin/python
import sys
import math

if len(sys.argv) < 2:
    print ('Usage:<filename1: data filename> <filename2: result filename >')
    exit(0)

f1 = open( sys.argv[1], 'r' )
f2 = open( sys.argv[2], 'r' )
resTrue = []
resPred = []
for line in f1.readlines():
    resTrue.append(float(line.split()[0]))

for line in f2.readlines():
    resPred.append(float(line))
f2.close()
f1.close()
len1 = len(resTrue)
len2 = len(resPred)
assert(len1 == len2)
tmp = 0;
for y1,y2 in zip(resTrue, resPred):
    tmp += (y1 - y2)*(y1- y2)
print tmp/len1
print math.sqrt(tmp/len1)
