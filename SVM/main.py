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
##         b:    bias
def testWeight(data, w):
  numWrong = 0
  for example in data:
    sum = 0
    for i in range(len(dataAttrs)):
      sum += float(example.getAttrs()[dataAttrs[i]]) * w[i]
    sum += w[len(w)-1]
    if (example.getLabel() == '1' and sum <= 0) or (example.getLabel() == '0' and sum >= 0):
      numWrong += 1
  return 100 * (numWrong/len(data))

trainData = extractData('./bank-note/train.csv', dataAttrs)
testData = extractData('./bank-note/test.csv', dataAttrs)

#run for 2a
if sys.argv[1] == '1':
  gamma = 4
  A = 0.0001
  print('C\ttrain\ttest\tweight vector with bias at end')
  print('100/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 100/873, gamma, 1, a=A)
  print(round(testWeight(trainData, w), 2), end='\t')
  print(round(testWeight(testData, w), 2), end='\t')
  print(w)
  print('500/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 500/873, gamma, 1, a=A)
  print(round(testWeight(trainData, w), 2), end='\t')
  print(round(testWeight(testData, w), 2), end='\t')
  print(w)
  print('700/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 700/873, gamma, 1, a=A)
  print(round(testWeight(trainData, w), 2), end='\t')
  print(round(testWeight(testData, w), 2), end='\t')
  print(w)

#run for 2b
if sys.argv[1] == '2':
  gamma = 0.001
  print('C\ttrain\ttest\tweight vector with bias at end')
  print('100/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 100/873, gamma, 2)
  print(round(testWeight(trainData, w), 2), end='\t')
  print(round(testWeight(testData, w), 2), end='\t')
  print(w)
  print('500/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 500/873, gamma, 2)
  print(round(testWeight(trainData, w), 2), end='\t')
  print(round(testWeight(testData, w), 2), end='\t')
  print(w)
  print('700/873', end='\t')
  w = SVMPrime('./bank-note/train.csv', dataAttrs, 100, '1', '0', 700/873, gamma, 2)
  print(round(testWeight(trainData, w), 2), end='\t')
  print(round(testWeight(testData, w), 2), end='\t')
  print(w)

#run for 3a
if sys.argv[1] == '3':
  result = SVMDual('./bank-note/train.csv', dataAttrs, '1', '0', 100/873)
  print('C = 100/873:')
  w = result[0]
  b = result[1]
  print('w =', w)
  print('b =', b)
  print()
  
  result = SVMDual('./bank-note/train.csv', dataAttrs, '1', '0', 500/873)
  print('C = 500/873:')
  w = result[0]
  b = result[1]
  print('w =', w)
  print('b =', b)
  print()
  
  result = SVMDual('./bank-note/train.csv', dataAttrs, '1', '0', 700/873)
  print('C = 700/873:')
  w = result[0]
  b = result[1]
  print('w =', w)
  print('b =', b)
  print()

#run for 3b and 3c
if sys.argv[1] == '4':
  def testWeightGauss(data, wxFunc, b):
    numWrong = 0
    for example in data:
      #extract numpy array
      arr = []
      for i in range(len(dataAttrs)):
        arr.append(ex.getAttrs()[dataAttrs[i]])
      x = np.array(arr,dtype=float)
      
      wx = wxFunc(x)
      sum = wx + b
      if (example.getLabel() == '1' and sum <= 0) or (example.getLabel() == '0' and sum >= 0):
        numWrong += 1
  
  print('gamma|\tC->')
  print('     v\t100/873\t\t\t500/873\t\t\t700/873')
  print('\ttrain\ttest\tnumVecs\ttrain\ttest\tnumVecs\ttrain\ttest\tnumVecs')
  
  supVecs = []
  
  def printAllC(gamma):
    print(gamma, end='\t')
    result = SVMDualGauss('./bank-note/train.csv', dataAttrs, '1', '0', 100/873, gamma)
    wxFunc = result[0]
    b = result[1]
    print(round(testWeightGauss(trainData, wxFunc, b), 2), end='\t')
    print(round(testWeightGauss(testData, wxFunc, b), 2), end='\t')
    print(len(result[3]), end='\t')
    result = SVMDualGauss('./bank-note/train.csv', dataAttrs, '1', '0', 500/873, gamma)
    wxFunc = result[0]
    b = result[1]
    supVecs.append(result[3])
    print(round(testWeightGauss(trainData, wxFunc, b), 2), end='\t')
    print(round(testWeightGauss(testData, wxFunc, b), 2), end='\t')
    print(len(result[3]), end='\t')
    result = SVMDualGauss('./bank-note/train.csv', dataAttrs, '1', '0', 700/873, gamma)
    wxFunc = result[0]
    b = result[1]
    print(round(testWeightGauss(trainData, wxFunc, b), 2), end='\t')
    print(round(testWeightGauss(testData, wxFunc, b), 2), end='\t')
    print(len(result[3]))
  
  printAllC(0.1)
  printAllC(0.5)
  printAllC(1)
  printAllC(5)
  printAllC(100)
  
  #count overlapped support vectors
  for i in range(4):
    count = 0
    for supVec in supVecs[i]:
      if supVec in supVecs[i+1]:
        count += 1
    if i == 0:
      print('0.1 and 0.5:', count)
    if i == 1:
      print('0.5 and 1:', count)
    if i == 2:
      print('1 and 5:', count)
    if i == 3:
      print('5 and 100:', count)