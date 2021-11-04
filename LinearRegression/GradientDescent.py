import sys
import numpy
import math
import random
from Basics.Basics import *

##runs batch gradient descent on LMS and returns weight vector for given data in file
##the first element of weight vector is the bias, rest is weights on attributes in order given in attrs
##input: trainFile: csv file of training examples
##       attrs:     list of attributes in the order in the file
##       threshold: error threshold to stop when below
##       r:         learning rate
##       p:         bool to print or not
def batch(trainFile, attrs, threshold, r, p=False):
  data = extractData(trainFile, attrs)
  #change examples to vectors, add the bias at the beginning
  attrs.insert(0, 'bias')
  attrData = []
  labelData = []
  for example in data:
    labelData.append(float(example.getLabel()))
    example.getAttrs()['bias'] = 1
    list = []
    for attr in attrs:
      list.append(float(example.getAttrs()[attr]))
    attrData.append(numpy.array(list))

  loss = threshold + 1
  t = 0
  #inital weight vector
  list = []
  for i in range(len(attrs)):
    list.append(0)
  weight = numpy.array(list)
  
  while loss > threshold:
    #find gradient
    g = []
    for j in range(len(weight)):
      sum = 0
      for i in range(len(attrData)):
        sum += (labelData[i] - weight.dot(attrData[i])) * attrData[i][j]
      g.append(-sum)
    
    gradient = numpy.array(g)
    
    newWeight = weight - r * gradient
    
    weight = newWeight
    sum = 0
    for i in range(len(attrData)):
      sum += (labelData[i] - weight.dot(attrData[i])) ** 2
    loss = sum / 2
    if p:
      print(str(t) + '\t' + str(loss))
    
    t += 1
  
  return weight

##find largest r that converges for batch gradient descent
def tuneBatch(trainFile, attrs):
  data = extractData(trainFile, attrs)
  #change examples to vectors, add the bias at the beginning
  attrs.insert(0, 'bias')
  attrData = []
  labelData = []
  for example in data:
    labelData.append(float(example.getLabel()))
    example.getAttrs()['bias'] = 1
    list = []
    for attr in attrs:
      list.append(float(example.getAttrs()[attr]))
    attrData.append(numpy.array(list))
  
  r = 1
  
  for y in range(100):
    t = 0
    decreasing = True
    loss = float('inf')
    converges = False
    #inital weight vector
    list = []
    for i in range(len(attrs)):
      list.append(0)
    weight = numpy.array(list)
    while decreasing:
      #find gradient
      g = []
      for j in range(len(weight)):
        sum = 0
        for i in range(len(attrData)):
          sum += (labelData[i] - weight.dot(attrData[i])) * attrData[i][j]
        g.append(-sum)
      
      gradient = numpy.array(g)
      
      newWeight = weight - r * gradient
      
      if numpy.linalg.norm(newWeight - weight) < 0.000001:
        converges = True
        break
      
      weight = newWeight
      t += 1
      sum = 0
      for i in range(len(attrData)):
        sum += (labelData[i] - weight.dot(attrData[i])) ** 2
      newLoss = sum / 2
      
      if newLoss > loss:
        decreasing = False
      
      loss = newLoss
    
    if converges:
      break
    r = r / 2
  
  return r

##runs stochastic gradient descent on LMS and returns weight vector for given data in file
##the first element of weight vector is the bias, rest is weights on attributes in order given in attrs
##input: trainFile: csv file of training examples
##       attrs:     list of attributes in the order in the file
##       threshold: error threshold to stop when below
##       r:         learning rate
##       p:         bool to print or not
def stochastic(trainFile, attrs, threshold, r, p=False):
  data = extractData(trainFile, attrs)
  #change examples to vectors, add the bias at the beginning
  attrs.insert(0, 'bias')
  attrData = []
  labelData = []
  for example in data:
    labelData.append(float(example.getLabel()))
    example.getAttrs()['bias'] = 1
    list = []
    for attr in attrs:
      list.append(float(example.getAttrs()[attr]))
    attrData.append(numpy.array(list))

  loss = threshold + 1
  t = 0
  #inital weight vector
  list = []
  for i in range(len(attrs)):
    list.append(0)
  weight = numpy.array(list)
  
  while loss > threshold:
    #randomly sample example
    i = random.randint(0, len(attrData)-1)
    list = []
    for j in range(len(weight)):
      list.append(weight[j] + r * (labelData[i] - weight.dot(attrData[i])) * attrData[i][j])
    
    weight = numpy.array(list)
    t += 1
    sum = 0
    for p in range(len(attrData)):
      sum += (labelData[p] - weight.dot(attrData[p])) ** 2
    loss = sum / 2
  
  print(loss)
  
  return weight#

##solve LMS using analysis method
def analyze(trainFile, attrs):
  data = extractData(trainFile, attrs)
  #change examples to matrix, add the bias at the beginning
  attrs.insert(0, 'bias')
  attrData = []
  labelData = []
  for example in data:
    labelData.append(float(example.getLabel()))
    example.getAttrs()['bias'] = 1
    list = []
    for attr in attrs:
      list.append(float(example.getAttrs()[attr]))
    attrData.append(numpy.array(list))
  
  matrixList = []
  for j in range(len(attrs)):
    rowList = []
    for i in range(len(attrData)):
      rowList.append(attrData[i][j])
    matrixList.append(rowList)
  X = numpy.array(matrixList)
  Y = numpy.array(labelData)
  
  #calculate optimal weight
  inv = numpy.linalg.inv(X.dot(X.transpose()))
  XY = X.dot(Y)
  return inv.dot(XY)

##get the loss of a weight vector on the given test file
def getLoss(weight, testFile, attrs):
  data = extractData(testFile, attrs)
  #change examples to vectors, add the bias at the beginning
  attrs.insert(0, 'bias')
  attrData = []
  labelData = []
  for example in data:
    labelData.append(float(example.getLabel()))
    example.getAttrs()['bias'] = 1
    list = []
    for attr in attrs:
      list.append(float(example.getAttrs()[attr]))
    attrData.append(numpy.array(list))

  sum = 0
  for p in range(len(attrData)):
    sum += (labelData[p] - weight.dot(attrData[p])) ** 2
  
  return sum / 2