import sys
import math
import random
import numpy as np
import scipy
from Basics.Basics import *
from scipy.optimize import minimize

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
      sum += w[len(w)-1]
      
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
      #  currSlope = np.linalg.norm(np.array(gradient))
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
        #update w_0
        for i in range(len(w)-1):
          w[i] = (1-r)*w[i]
      
      
      
      updateCount += 1
  
  return w

##constraint for DualSVM
def DualCons(a, y):
  sum = 0
  for i in range(len(a)):
    sum += a[i]*y[i]
  return sum

##objective function of the dual SVM
def DualFunc(a, x, y):
  #find left sum
  sum_i = 0
  for i in range(len(x)):
    sum_j = 0
    for j in range(len(x)):
      sum_j += y[i]*y[j]*a[i]*a[j]*np.dot(np.transpose(x[i]), x[j])
    sum_i += sum_j
  leftSum = sum_i/2
  
  #find right sum
  rightSum = 0
  for i in range(len(x)):
    rightSum += a[i]
  
  return leftSum - rightSum

##minimize dual of SVM
##inputs: trainFile:  path to csv file of data
##        attrs:      list of attrs in the order they appear is data
##        posLabel:   label that is considered positive
##        negLabel:   label that is considered negative
##        C:          hyperparameter
##        seed:       (optional) seed for random, default 10
##returns weight vector and bias
def SVMDual(trainFile, attrs, posLabel, negLabel, C, seed=10):
  #set random seed to have consistant data
  random.seed(seed)
  #extract list of examples
  data = extractData(trainFile, attrs)
  
  #change data to numpy array
  x = []
  for ex in data:
    arr = []
    for i in range(len(attrs)):
      arr.append(ex.getAttrs()[attrs[i]])
    x.append(np.array(arr,dtype=float))
  
  #get array of labels
  y = []
  for ex in data:
    if ex.getLabel() == posLabel:
      y.append(1)
    else:
      y.append(-1)
  y = np.array(y,dtype=float)
  
  #constraint for minimization
  cons = {
           'type': 'eq',
           'fun':  DualCons,
           'args': [y]
         }
  
  #build bounds for each variable in a
  bnds = []
  for i in range(len(x)):
    bnds.append((0,C))
  
  #build initial guess of 0s
  a_0 = []
  for i in range(len(x)):
    a_0.append(0)
  
  result = scipy.optimize.minimize(DualFunc, a_0, args=(x,y), method='SLSQP', bounds=bnds, constraints=cons)
  
  if not result.success:
    print('optimization failed!')
  
  a = result.x
  
  #get w
  vecs = []
  for i in range(len(a)):
    vecs.append(a[i]*y[i]*x[i])
  w = vecs[0]
  for i in range(1,len(vecs)):
    w = w + vecs[i]
  
  #get bias by averaging over all j
  sum = 0
  count = 0
  for j in range(len(a)):
    if a[j] > 0 and a[j] < C:
      sum += y[j] - np.dot(np.transpose(w), x[j])
      count += 1
  b = sum/count
  
  return (w,b)

##minimize dual of SVM using gaussian kernel
##inputs: trainFile:  path to csv file of data
##        attrs:    list of attrs in the order they appear is data
##        posLabel: label that is considered positive
##        negLabel: label that is considered negative
##        C:        hyperparameter
##        gamma:    used in gaussian kernel
##        seed:     (optional) seed for random, default 10
##returns function for finding (w,x), b, and support vectors
def SVMDualGauss(trainFile, attrs, posLabel, negLabel, C, gamma, seed=10):
  ##gaussian kernel
  def K(x_i, x_j):
    return math.exp(-(np.linalg.norm(x_i - x_j))**2/gamma)

  ##objective function of the dual SVM with gaussian kernel
  def DualFuncGauss(a, x, y):
    #find left sum
    sum_i = 0
    for i in range(len(x)):
      sum_j = 0
      for j in range(len(x)):
        sum_j += y[i]*y[j]*a[i]*a[j]*K(x[i],x[j])
      sum_i += sum_j
    leftSum = sum_i/2
    
    #find right sum
    rightSum = 0
    for i in range(len(x)):
      rightSum += a[i]
    
    return leftSum - rightSum

  #set random seed to have consistant data
  random.seed(seed)
  #extract list of examples
  data = extractData(trainFile, attrs)
  
  #change data to numpy array
  x = []
  for ex in data:
    arr = []
    for i in range(len(attrs)):
      arr.append(ex.getAttrs()[attrs[i]])
    x.append(np.array(arr,dtype=float))
  
  #get array of labels
  y = []
  for ex in data:
    if ex.getLabel() == posLabel:
      y.append(1)
    else:
      y.append(-1)
  y = np.array(y,dtype=float)
  
  #constraint for minimization
  cons = {
           'type': 'eq',
           'fun':  DualCons,
           'args': [y]
         }
  
  #build bounds for each variable in a
  bnds = []
  for i in range(len(x)):
    bnds.append((0,C))
  
  #build initial guess of 0s
  a_0 = []
  for i in range(len(x)):
    a_0.append(0)
  
  result = scipy.optimize.minimize(DualFuncGauss, a_0, args=(x,y), method='SLSQP', bounds=bnds, constraints=cons)
  
  if not result.success:
    print('optimization failed!')
  
  a = result.x
  
  def findWX(m):
    vecs = []
    for i in range(len(a)):
      vecs.append(a[i]*y[i]*K(x[i], m))
    wx = []
    for i in range(len(vecs)):
      wx = w + vecs[i]
    return wx
  
  #get w
  vecs = []
  for i in range(len(a)):
    vecs.append(a[i]*y[i]*x[i])
  w = vecs[0]
  for i in range(1,len(vecs)):
    w = w + vecs[i]
  
  #get bias by averaging over all j
  sum = 0
  count = 0
  for j in range(len(a)):
    if a[j] > 0 and a[j] < C:
      sum += y[j] - np.dot(np.transpose(w)*x[j])
      count += 1
  b = sum/count
  
  #list support vectors
  supVecs = []
  for i in range(len(a)):
    if not a[i] == 0: #if a support vector
      supVecs.append(i)
  
  return (findWX,b,supVecs)