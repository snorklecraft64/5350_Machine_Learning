import sys
sys.path.append('../')
sys.path.append('./')
import numpy
from Perceptron import *
from Basics.Basics import *

dataAttrs = [
             'variance',
             'skewness',
             'curtosis',
             'entropy'
            ]

##return percentage error of weight vector
##inputs:  data: list of Examples to test on
##         w:    weight vector to test
def testWeight(data, w):
  numWrong = 0
  for example in data:
    sum = 0
    for i in range(len(dataAttrs)):
      sum += float(example.getAttrs()[dataAttrs[i]]) * w[i]
    if (example.getLabel() == '1' and sum <= 0) or (example.getLabel() == '0' and sum >= 0):
      numWrong += 1
  return 100 * (numWrong/len(data))

##return percentage error of hypothesis
##inputs: data: list of Examples to test on
##        H:    the hypothesis to test
def testH(data, H):
  numWrong = 0
  for example in data:
    x = []
    for i in range(len(dataAttrs)):
      x.append(float(example.getAttrs()[dataAttrs[i]]))
    pred = H(numpy.array(x))
    if (example.getLabel() == '1' and pred <= 0) or (example.getLabel() == '0' and pred >= 0):
      numWrong += 1
  return 100 * (numWrong/len(data))
  
testData = extractData('./bank-note/test.csv', dataAttrs)

if sys.argv[1] == '1':
  w = stdPercep('./bank-note/train.csv', dataAttrs, 10, '1', '0')
  print(w)
  print(testWeight(testData, w))

if sys.argv[1] == '2':
  WCH = votedPercep('./bank-note/train.csv', dataAttrs, 10, '1', '0')
  weights = WCH[0]
  counts = WCH[1]
  hypothesis = WCH[2]
  for i in range(len(weights)):
    print(weights[i], end=' ')
    print(counts[i])
  print(testH(testData, hypothesis))

if sys.argv[1] == '3':
  w = avgPercep('./bank-note/train.csv', dataAttrs, 10, '1', '0')
  print(w)
  print(testWeight(testData, w))