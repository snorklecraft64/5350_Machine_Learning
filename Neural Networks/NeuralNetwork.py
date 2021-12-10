import sys
import math

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
  ##input:  X:      numpy array of feature vectors, already has bias at beginning of each example
  ##        y:      numpy array of labels
  ##        width:  number of nodes in each hidden layer
  ##        L1W:    (optional) layer 1 weights (dict of tuple to float)
  ##        L2W:    (optional) layer 2 weights (dict of tuple to float)
  ##        L3W:    (optional) layer 3 weights (dict of tuple to float)
  def __init__(this, X, y, width, L1W = None, L2W = None, L3W = None):
    this.__X = X
    this.__y = y
    this.__width = width
    
    #initialize weights according to Guassian distribution ------------TODO (currently doing all 0)
    #weights for layer 1
    if L1W != None:
      this.__L1W = L1W
    else:
      this.__L1W = {}
      for i in range(len(X[0])):
        for j in range(1,width+1):   # +1 because of bias
          this.__L1W[(i,j)] = 0 #replace with gaussian distribution
    
    if L2W != None:
      this.__L2W = L2W
    else:
      this.__L2W = {}
      for i in range(width+1):     # v
        for j in range(1,width+1): # +1 because of bias
          this.__L2W[(i,j)] = 0 #replace with gaussian distribution
    
    if L3W != None:
      this.__L3W = L3W
    else:
      this.__L3W = {}
      for i in range(1,width+1):
        this.__L3W[(i,0)] = 0 #replace with gaussian distribution
    
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
  
  ##sets all neurons back to default values
  def clearNeurons(this):
    lst = []
    lst.append(Neuron(1,None))
    for i in range(1,this.__width+1):
      lst.append(Neuron(None, None))
    this.L1N = lst
    
    lst = []
    lst.append(Neuron(1,None))
    for i in range(1,this.__width+1):
      lst.append(Neuron(None, None))
    this.L2N = lst
    
    this.__YN = Neuron(None, None)
  
  ##sigmoid function
  def sigmoid(x):
    return 1 / (1 + math.exp(-x))
  
  ##feeds given example through, updating the values of the neurons
  ##input:  i:  index in X,y of example to process
  def feedThrough(this, index):
    ex = this.__X[index]
    #find and store values for layer 1 neurons
    for j in range(1,len(this.__L1N)):
      sum = 0
      for i in range(len(ex)):
        sum += this.__L1W[(i,j)] * ex[i]
      this.__L1N[j].setValue(NeuralNetwork.sigmoid(sum))
    
    #find and store values for layer 2 neurons
    for j in range(1,len(this.__L2N)):
      sum = 0
      for i in range(len(this.__L1N)):
        sum += this.__L2W[(i,j)] * this.__L1N[i].getValue()
      this.__L2N[j].setValue(NeuralNetwork.sigmoid(sum))
    
    #find and store value for y
    sum = 0
    for i in range(len(this.__L2N)):
      sum += this.__L3W[(i,0)] * this.__L2N[i].getValue()
    this.__YN .setValue(sum)
    
    #calculate partial derivative of y
    this.__YN.setG(sum - this.__y[index])
  
  ##returns the derivative of given weight and stores partial derivatives in neurons
  ##input:  index:  index in X,y of example we are processing
  ##        h:      layer of weight
  ##        m:      index of neuron weight comes from
  ##        n:      index of neuron weight goes to
  def findDerivative(this, index, h, m, n):
    #weight points to y
    if h == 3:
      return this.__YN.getG() * this.__L2N[m].getValue() #y will always have g filled because you need to feedthrough first
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