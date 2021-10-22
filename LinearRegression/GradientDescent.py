import sys
sys.path.append('../')
import numpy
import math
from Basics.Basics import *

##input: trainFile: csv file of training examples
##       attrs:     list of attributes in the order in the file
##       threshold: error threshold to stop when below
##       r:         learning rate
def batch(trainFile, attrs, threshold, r):
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
    t += 1
    sum = 0
    for i in range(len(attrData)):
      sum += (labelData[i] - weight.dot(attrData[i])) ** 2
    loss = sum / 2
    #print(loss)
  
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

#def stochastic():
  

dataAttrs = [
             'Cement',
             'Slag',
             'Fly ash',
             'Water',
             'SP',
             'Coarse Aggr',
             'Fine Aggr',
            ]

weight = batch('./concrete/train.csv', dataAttrs.copy(), 14.981943657085, tuneBatch('./concrete/train.csv', dataAttrs.copy()))
print(weight)