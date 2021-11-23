import sys
import random
import numpy
from Basics.Basics import *

##run stochasic sub-gradient descent for SVM
##input:  trainFile:  csv file full of examples to train on
##        attrs:      list of attrs in the order the weight vector will multiply them
##        T:          max epochs
##        posLabel:   label that is considered positive
##        negLabel:   label that is considered negative
##        C:          hyperparameter C
##        r_0:        initial learning rate
##        version:    what learning rate schedule to use
##                    1: from part a
##                    2: from part b
##        a:          parameter NEEDED for version 1 (default 1, don't use default lol)
##        seed:       (optional) seed for random, default 10
def SVMPrime(trainFile, attrs, T, posLabel, negLabel, C, r_0, version, a = 1, seed = 10):
  #set random seed to have consistant data
  random.seed(seed)
  #extract list of examples
  data = extractData(trainFile, attrs)
  
  #initilize w to 0
  w = []
  for i in range(len(attrs)):
    w.append(0)
  w.append(0) #bias
  
  updateCount = 0
  prevSlope = float('inf')
  currSlope = float('inf')
  
  r = r_0
  
  for t in range(1,T):
    #schedule learning rate
    if version == 1:
      r = r_0 / (1+(r_0/a)*t)
    else:
      r = r_0 / (1+t)
    
    random.shuffle(data)
    for example in data:
      #figure out wx
      sum = 0
      for i in range(len(attrs)):
        sum += float(example.getAttrs()[attrs[i]]) * w[i]
      
      #figure out y_i
      if example.getLabel() == negLabel:
        y = -1
      else:
        y = 1
      
      ymx = y * sum
      
      ##calculate gradient every 100 updates to see if slope has significantly changed
      #if updateCount % 100 == 0:
      #  #find gradient
      #  gradient = []
      #  for i in range(len(attrs)):
      #    gradient.append(0)
      #  gradient.append(0) #bias
      #  
      #  if ymx <= 1:
      #    for i in range(len(gradient)-1):
      #      gradient[i] = w[i] - C*len(data)*y*float(example.getAttrs()[attrs[i]])
      #    gradient[len(gradient)-1] = 0 - C*len(data)*y
      #  else:
      #    for i in range(len(gradient)-1):
      #      gradient[i] = w[i]
      #    gradient[len(gradient)-1] = 0
      #  
      #  prevSlope = currSlope
      #  currSlope = numpy.linalg.norm(numpy.array(gradient))
      #  #print(currSlope)
      #  if currSlope < 20:
      #    print(currSlope)
      #  
      #  test = currSlope - prevSlope
      #  if test < 0.000001 and test > -0.000001:
      #    print('converges')
      
      if ymx <= 1:
        #update w
        for i in range(len(w)-1):
          w[i] = w[i] - r*w[i] + r*C*len(data)*y*float(example.getAttrs()[attrs[i]])
        w[len(w)-1] = w[len(w)-1] + r*C*len(data)*y
      else:
        print('here')
        #update w_0
        for i in range(len(w)-1):
          w[i] = (1-r)*w[i]
      
      
      
      updateCount += 1
  
  return w