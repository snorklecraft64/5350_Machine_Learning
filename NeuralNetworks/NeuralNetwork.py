import sys
import math
import random
import numpy as np

class Neuron:
  def __init__(this, value, g):
    #the value of this neuron
    this.__value = value
    #the partial derivative cache
    this.__g = g
    ##dict of int to neuron
    #this.__childs = childs
  
  def getValue(this):
    return this.__value
  
  def getG(this):
    return this.__g
  
  def setValue(this, v):
    this.__value = v
  
  def setG(this, g):
    this.__g = g

class NeuralNetwork:
  ##input:  X:      array of feature vectors, already has bias at beginning of each example
  ##        y:      array of labels
  ##        width:  number of nodes in each hidden layer
  ##        L1W:    (optional) layer 1 weights (dict of tuple to float)
  ##        L2W:    (optional) layer 2 weights (dict of tuple to float)
  ##        L3W:    (optional) layer 3 weights (dict of tuple to float)
  ##                ^if not given, initialized to normal distribution
  ##        zero:   whether or not to initialize weights to 0
  def __init__(this, X, y, width, L1W = None, L2W = None, L3W = None, zero=False):
    this.__X = X
    this.__y = y
    this.__width = width
    if zero:
      #initialize weights with 0
      #weights for layer 1
      if L1W != None:
        this.__L1W = L1W
      else:
        this.__L1W = {}
        for i in range(len(X[0])):
          for j in range(1,width+1):   # +1 because of bias
            this.__L1W[(i,j)] = 0
      
      if L2W != None:
        this.__L2W = L2W
      else:
        this.__L2W = {}
        for i in range(width+1):     # v
          for j in range(1,width+1): # +1 because of bias
            this.__L2W[(i,j)] = 0
      
      if L3W != None:
        this.__L3W = L3W
      else:
        this.__L3W = {}
        for i in range(width+1):
          this.__L3W[(i,0)] = 0
    else:
      #initialize weights according to Guassian distribution
      #weights for layer 1
      if L1W != None:
        this.__L1W = L1W
      else:
        this.__L1W = {}
        for i in range(len(X[0])):
          for j in range(1,width+1):   # +1 because of bias
            this.__L1W[(i,j)] = np.random.normal()
      
      if L2W != None:
        this.__L2W = L2W
      else:
        this.__L2W = {}
        for i in range(width+1):     # v
          for j in range(1,width+1): # +1 because of bias
            this.__L2W[(i,j)] = np.random.normal()
      
      if L3W != None:
        this.__L3W = L3W
      else:
        this.__L3W = {}
        for i in range(width+1):
          this.__L3W[(i,0)] = np.random.normal()
    
    #initialize neurons with None value and g, since we can't know value until example is fed through
    #otherwise known as X
    ##might not be needed
    #this.__L0N = []
    #this.__L0N.append(Neuron(1, None))
    #for i in range(1,len(X[0])+1): # +1 because of bias
    #  this.__L0N.append(Neuron(None, None)
    
    #neurons for layer 1
    this.__L1N = []
    this.__L1N.append(Neuron(1, None))
    for i in range(1,width+1): # +1 because of bias
      this.__L1N.append(Neuron(None, None))
    
    #neurons for layer 2
    this.__L2N = []
    this.__L2N.append(Neuron(1, None))
    for i in range(1,width+1): # +1 because of bias
      this.__L2N.append(Neuron(None, None))
    
    #the single neuron for layer 3, aka y
    this.__YN = Neuron(None, None)
  
  ##sigmoid function
  def sigmoid(x):
    try:
      return 1 / (1 + math.exp(-x))
    except OverflowError:
      if x < 0: #if we got overflow, the value will be extremely close to 0 or 1 depending on if x is negative
        return 0
      return 1
  
  ##feeds given example through, updating the values of the neurons
  ##makes the partial derivatives None, since we no longer know them
  ##input:  ex: example to feed through (make sure has bias at beginning)
  def feedThrough(this, ex):
    #find and store values for layer 1 neurons
    for j in range(1,len(this.__L1N)):
      sum = 0
      for i in range(len(ex)):
        sum += this.__L1W[(i,j)] * ex[i]
      this.__L1N[j].setValue(NeuralNetwork.sigmoid(sum))
      this.__L1N[j].setG(None)
    
    #find and store values for layer 2 neurons
    for j in range(1,len(this.__L2N)):
      sum = 0
      for i in range(len(this.__L1N)):
        sum += this.__L2W[(i,j)] * this.__L1N[i].getValue()
      this.__L2N[j].setValue(NeuralNetwork.sigmoid(sum))
      this.__L2N[j].setG(None)
    
    #find and store value for y
    sum = 0
    for i in range(len(this.__L2N)):
      sum += this.__L3W[(i,0)] * this.__L2N[i].getValue()
    this.__YN.setValue(sum)
    this.__YN.setG(None)
  
  ##returns the derivative of given weight and stores partial derivatives in neurons
  ##input:  index:  index in X,y of example we are processing
  ##        h:      layer of weight
  ##        m:      index of neuron weight comes from
  ##        n:      index of neuron weight goes to
  def findDerivative(this, index, h, m, n):
    #weight points to y
    if h == 3:
      g = 0
      if this.__YN.getG() == None:
        g = this.__YN.getValue() - this.__y[index]
        this.__YN.setG(g)
      else:
        g = this.__YN.getG()
      
      return g * this.__L2N[m].getValue() #y will always have g filled because you need to feedthrough first
    elif h == 2:
      g = 0
      #if not cached
      if this.__L2N[n].getG() == None:
        g = this.__YN.getG() * this.__L3W[(n,0)]
        this.__L2N[n].setG(g)
      else:
        g = this.__L2N[n].getG()
      
      s = 0
      for i in range(len(this.__L1N)):
        s += this.__L2W[(i,n)] * this.__L1N[i].getValue()
      
      return g * NeuralNetwork.sigmoid(s) * (1-NeuralNetwork.sigmoid(s)) * this.__L1N[m].getValue()
    else: #h == 1
      ex = this.__X[index]
      g = 0
      #if not cached
      if this.__L1N[n].getG() == None:
        #for each path
        for i in range(1,len(this.__L2N)):
          s = 0
          for j in range(len(this.__L1N)):
            s += this.__L2W[(j,i)] * this.__L1N[j].getValue()
          
          g += this.__L2N[i].getG() * NeuralNetwork.sigmoid(s) * (1-NeuralNetwork.sigmoid(s)) * this.__L2W[(n,i)]
        this.__L1N[n].setG(g)
      else:
        g = this.__L1N[n].getG()
      
      s = 0
      for i in range(len(ex)):
        s += this.__L1W[(i,n)] * ex[i]
      
      return g * NeuralNetwork.sigmoid(s) * (1-NeuralNetwork.sigmoid(s)) * ex[m]
  
  ##print all partial derivatives of the gradient
  ##input:  index:  index in X,y of example to process
  def printAllDerivatives(this, index):
    ex = this.__X[index]
    
    h = 3
    n = 0
    for m in range(len(this.__L2N)):
      print('h =', h, 'm =', m, 'n =', n, ':', this.findDerivative(index,h,m,n))
    
    h = 2
    for m in range(len(this.__L1N)):
      for n in range(1,len(this.__L2N)):
        print('h =', h, 'm =', m, 'n =', n, ':', this.findDerivative(index,h,m,n))
    
    h = 1
    for m in range(len(ex)):
      for n in range(1,len(this.__L1N)):
        print('h =', h, 'm =', m, 'n =', n, ':', this.findDerivative(index,h,m,n))
  
  ##finds the gradient with respect to the given example index
  ##input:  index:  index in X,y of example we are processing
  ##returns (L1G, L2G, L3G), where LiG is the gradient of layer i (dict of tuple to float)
  def findGradient(this, index):
    h = 3
    n = 0
    L3G = {}
    for m in range(len(this.__L2N)):
      L3G[(m,n)] = this.findDerivative(index, h, m, n)
    
    h = 2
    L2G = {}
    for m in range(len(this.__L1N)):
      for n in range(1,len(this.__L2N)):
        L2G[(m,n)] = this.findDerivative(index, h, m, n)
    
    h = 1
    L1G = {}
    for m in range(len(this.__X[0])):
      for n in range(1,len(this.__L1N)):
        L1G[(m,n)] = this.findDerivative(index, h, m, n)
    
    return L1G, L2G, L3G
  
  ##finds the current loss
  def findLoss(this):
    sum = 0
    for i in range(len(this.__X)):
      this.feedThrough(this.__X[i])
      sum += (this.__YN.getValue() - this.__y[i])**2 / 2
    return sum
  
  ##returns iterator to go randomly go over a sequence
  def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)
  
  ##perform stochastic gradient descent to train
  ##input:  r_0:  initial learning rate
  ##        d:  parameter for learning rate schedule
  ##        T:  number of epochs
  def SGD(this, r_0, d, T):
    for t in range(T):
      r = r_0 / (1+(r_0/d)*t)
      
      #go through X in random order, that way we don't modify X
      for i in NeuralNetwork.randomly(range(len(this.__X))):
        ex = this.__X[i]
        this.feedThrough(ex)
        
        L1G, L2G, L3G = this.findGradient(i)
        
        #update layer 1 weights
        for i in range(len(ex)):
          for j in range(1,this.__width+1):   # +1 because of bias
            this.__L1W[(i,j)] = this.__L1W[(i,j)] - r * L1G[(i,j)]
        
        #update layer 2 weights
        for i in range(this.__width+1):     # v
          for j in range(1,this.__width+1): # +1 because of bias
            this.__L2W[(i,j)] = this.__L2W[(i,j)] - r * L2G[(i,j)]
        
        #update layer 3 weights
        for i in range(this.__width+1):
          this.__L3W[(i,0)] = this.__L3W[(i,0)] - r * L3G[(i,0)]
  
  ##test on given X and y
  def test(this, X, y):
    numWrong = 0
    for i in range(len(X)):
      this.feedThrough(X[i])
      if (this.__YN.getValue() >= 0 and y[i] == -1) or (this.__YN.getValue() <= 0 and y[i] == 1):
        numWrong += 1
    return(100 * (numWrong / len(X)))