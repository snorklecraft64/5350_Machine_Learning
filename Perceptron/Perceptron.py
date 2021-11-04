import sys
import random
import numpy
from Basics.Basics import *

##run standard perceptron, return weight vector
##uses learning rate of 1
##input:  trainFile:  csv file full of examples to train on
##        attrs:      list of attrs in the order the weight vector will multiply them
##        T:          max epochs
##        posLabel:   label that is considered positive
##        negLabel:   label that is considered negative
##        seed:       (optional) seed for random, default 10
def stdPercep(trainFile, attrs, T, posLabel, negLabel, seed=10):
  #set random seed to have consistant data
  random.seed(seed)
  #extract list of examples
  data = extractData(trainFile, attrs)
  
  #initilize w to 0
  w = []
  for i in range(len(attrs)):
    w.append(0)
  
  for t in range(T):
    random.shuffle(data)
    for example in data:
      #figure out w * x
      sum = 0
      for i in range(len(attrs)):
        sum += float(example.getAttrs()[attrs[i]]) * w[i]
      
      #if the weight vector predicted wrong, update the weight vector
      if (example.getLabel() == posLabel and sum <= 0) or (example.getLabel() == negLabel and sum >= 0):
        for i in range(len(w)):
          #if we have positive label we add
          if example.getLabel() == posLabel:
            w[i] += float(example.getAttrs()[attrs[i]])
          #if we have negative label we subtract
          else:
            w[i] -= float(example.getAttrs()[attrs[i]])
  
  return w

##run voted perceptron, return weight vector, counts, and hypothesis
##uses learning rate of 1
##input:  trainFile:  csv file full of examples to train on
##        attrs:      list of attrs in the order the weight vector will multiply them
##        T:          max epochs
##        posLabel:   label that is considered positive
##        negLabel:   label that is considered negative
##        seed:       (optional) seed for random, default 10
def votedPercep(trainFile, attrs, T, posLabel, negLabel, seed=10):
  #set random seed to have consistant data
  random.seed(seed)
  #extract list of examples
  data = extractData(trainFile, attrs)
  
  #initilize w to 0
  w = []
  for i in range(len(attrs)):
    w.append(0)
  
  weights = [numpy.array(w)]
  m = 0
  
  #initilize C_0
  C = [0]
  
  for t in range(T):
    random.shuffle(data)
    for example in data:
      #figure out w_m * x
      sum = 0
      for i in range(len(attrs)):
        sum += float(example.getAttrs()[attrs[i]]) * weights[m][i]
      
      #if the weight vector predicted wrong, update the weight vector, m, and C
      if (example.getLabel() == posLabel and sum <= 0) or (example.getLabel() == negLabel and sum >= 0):
        newWeight = []
        for i in range(len(w)):
          #if we have positive label we add
          if example.getLabel() == posLabel:
            newWeight.append(weights[m][i] + float(example.getAttrs()[attrs[i]]))
          #if we have negative label we subtract
          else:
            newWeight.append(weights[m][i] - float(example.getAttrs()[attrs[i]]))
        
        weights.append(numpy.array(newWeight))
        m += 1
        C.append(1)
      else:
        C[m] += 1
  
  ##x: numpy array of example
  def hypothesis(x):
    sum = 0
    for k in range(len(weights)):
      sum += C[k] * numpy.sign(weights[k].dot(x))
    return numpy.sign(sum)
  
  return [weights, C, hypothesis]

##run averaged perceptron, return weight vector
##uses learning rate of 1
##input:  trainFile:  csv file full of examples to train on
##        attrs:      list of attrs in the order the weight vector will multiply them
##        T:          max epochs
##        posLabel:   label that is considered positive
##        negLabel:   label that is considered negative
##        seed:       (optional) seed for random, default 10
def avgPercep(trainFile, attrs, T, posLabel, negLabel, seed=10):
  #set random seed to have consistant data
  random.seed(seed)
  #extract list of examples
  data = extractData(trainFile, attrs)
  
  #initilize w to 0
  w = []
  for i in range(len(attrs)):
    w.append(0)
  
  #initilize a to 0
  a = []
  for i in range(len(attrs)):
    a.append(0)
  
  c = 0
  
  for t in range(T):
    random.shuffle(data)
    for example in data:
      #figure out w * x
      sum = 0
      for i in range(len(attrs)):
        sum += float(example.getAttrs()[attrs[i]]) * w[i]
      
      #if the weight vector predicted wrong, update the weight vector
      if (example.getLabel() == posLabel and sum <= 0) or (example.getLabel() == negLabel and sum >= 0):
        for i in range(len(w)):
          #if we have positive label we add
          if example.getLabel() == posLabel:
            w[i] += float(example.getAttrs()[attrs[i]])
          #if we have negative label we subtract
          else:
            w[i] -= float(example.getAttrs()[attrs[i]])
      
      #update a
      for i in range(len(a)):
        a[i] = a[i] + w[i]
      
      c += 1
  
  return numpy.array(a)/c