import sys
sys.path.append('../')
sys.path.append('./')
import numpy
from SVM import *
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

trainData = extractData('./bank-note/train.csv', dataAttrs)
testData = extractData('./bank-note/test.csv', dataAttrs)

#run for part a
if sys.argv[1] == '1':
  gamma = 4
  A = 0.0001
  print('C\ttrain\ttest\tweight vector with bias at end')
  print('100/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 100/873, gamma, 1, a=A)
  print(testWeight(trainData, w), end='\t')
  print(testWeight(testData, w), end='\t')
  print(w)
  print('500/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 500/873, gamma, 1, a=A)
  print(testWeight(trainData, w), end='\t')
  print(testWeight(testData, w), end='\t')
  print(w)
  print('700/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 700/873, gamma, 1, a=A)
  print(testWeight(trainData, w), end='\t')
  print(testWeight(testData, w), end='\t')
  print(w)
  #print('C\ttrain\ttest')
  #
  #print('100/873', end='\t')
  #random.seed(10)
  #trainErrs = []
  #testErrs = []
  #for i in range(100):
  #  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 100/873, 0.01, 1, a=0.001)
  #  trainErrs.append(testWeight(trainData, w))
  #  testErrs.append(testWeight(testData, w))
  #print(sum(trainErrs)/100, end='\t')
  #print(sum(testErrs)/100)
  #
  #print('500/873', end='\t')
  #trainErrs = []
  #testErrs = []
  #for i in range(100):
  #  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 500/873, 0.01, 1, a=0.001)
  #  trainErrs.append(testWeight(trainData, w))
  #  testErrs.append(testWeight(testData, w))
  #print(sum(trainErrs)/100, end='\t')
  #print(sum(testErrs)/100)
  #
  #print('700/873', end='\t')
  #trainErrs = []
  #testErrs = []
  #for i in range(100):
  #  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 700/873, 0.01, 1, a=0.001)
  #  trainErrs.append(testWeight(trainData, w))
  #  testErrs.append(testWeight(testData, w))
  #print(sum(trainErrs)/100, end='\t')
  #print(sum(testErrs)/100)

if sys.argv[1] == '2':
  gamma = 0.001
  print('C\ttrain\ttest\tweight vector with bias at end')
  print('100/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 100/873, gamma, 2)
  print(testWeight(trainData, w), end='\t')
  print(testWeight(testData, w), end='\t')
  print(w)
  print('500/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 500/873, gamma, 2)
  print(testWeight(trainData, w), end='\t')
  print(testWeight(testData, w), end='\t')
  print(w)
  print('700/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 700/873, gamma, 2)
  print(testWeight(trainData, w), end='\t')
  print(testWeight(testData, w), end='\t')
  print(w)
  #print('C\ttrain\ttest\tweight')
  #
  #print('100/873', end='\t')
  #random.seed(10)
  #trainErrs = []
  #testErrs = []
  #for i in range(100):
  #  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 100/873, 0.001, 2)
  #  trainErrs.append(testWeight(trainData, w))
  #  testErrs.append(testWeight(testData, w))
  #print(sum(trainErrs)/100, end='\t')
  #print(sum(testErrs)/100)
  #
  #print('500/873', end='\t')
  #trainErrs = []
  #testErrs = []
  #for i in range(100):
  #  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 500/873, 0.001, 2)
  #  trainErrs.append(testWeight(trainData, w))
  #  testErrs.append(testWeight(testData, w))
  #print(sum(trainErrs)/100, end='\t')
  #print(sum(testErrs)/100)
  #
  #print('700/873', end='\t')
  #trainErrs = []
  #testErrs = []
  #for i in range(100):
  #  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 700/873, 0.001, 2)
  #  trainErrs.append(testWeight(trainData, w))
  #  testErrs.append(testWeight(testData, w))
  #print(sum(trainErrs)/100, end='\t')
  #print(sum(testErrs)/100)