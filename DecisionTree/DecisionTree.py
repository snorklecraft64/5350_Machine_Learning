import sys
import math

##represents a single example in a training data set
class Example:
  def __init__(this, ID, count, attributes, label):
    this.ID = ID
    #how much space this example takes in the training data
    this.count = count
    #the attributes of this example (dictionary where keys are attributes and values are the value of that attribute)
    this.attributes = attributes
    #the label given to this example
    this.label = label
  
  def __str__(this):
    return '[' + str(this.ID) + ', ' + str(this.count) + ', ' + str(this.attributes) + ', ' + str(this.label) + ']'
  
  def __repr__(this):
    return this.__str__()
  
  def __hash__(this):
    return ID

  def __eq__(this, other):
    if this.ID == other.ID:
      return true
    return false

##represents a decision tree, initially untrained
class DecisionTree:
  
  ##input:  attrs:    list of attributes in the order they appear in the data
  ##        attrDict: dictionary of attributes to a list of their possible values
  ##        labels:   list of labels data can have
  def __init__(this, attrs, attrDict, labels):
    #list of training data, an example with ID x will be at trainingData[x]
    this.trainingData = []
    this.attrs = attrs
    this.attrDict = attrDict
    this.labels = labels
    this.tree = [] #:::::::::::::TODO implement tree (maybe take from library)

  ##fills trainingData with the data given in the input file
  def fillData(this, fileName):
    ID = 0
    with open(fileName, 'r') as f:
      for line in f:
        terms = line.strip().split(',')
        
        #process one training example :::::::TODO code for partial counts may go here (or maybe after all full ones have completed (if statement))
        #build the dictionary of attributes
        attributes = {}
        for i in range(len(this.attrs)):
          attributes[this.attrs[i]] = terms[i]
        #the label is the last in data set
        this.trainingData.append(Example(ID, 1, attributes, terms[len(this.attrs)]))
        ID += 1

  ##train this decision tree on examples from the given file
  ##input:  fileName: path to file containing data
  ##        version:  version of information gain to use:
  ##                  'E' = entropy
  ##                  'ME' = majority error
  ##                  'GI' = gini index
  ##        maxDepth: the max depth the tree should reach before stopping
  def train(this, fileName, version, maxDepth):
    this.fillData(fileName)
    
    #create list of all indexes for the initial call to ID3
    examples = []
    for i in range(len(this.trainingData)):
      examples.append(i)
    
    this.tree = this.ID3(examples, this.attrs, version, maxDepth)
  
  ##run the ID3 algorithm on the given set of examples, with the given set of attributes left to consider
  ##input:  examples:   list of rule IDs we are considering
  ##        attributes: list of attributes we are considering
  ##        version:    version of information gain to use:
  ##                    'E' = entropy
  ##                    'ME' = majority error
  ##                    'GI' = gini index
  ##        maxDepth:   the max depth the tree should reach before stopping
  ##returns: root node of the decision tree
  def ID3(this, examples, attributes, version, maxDepth):
    print(this.getInfoGain(examples, 'buying', 'E'))
    print(this.getInfoGain(examples, 'maint', 'E'))
    print(this.getInfoGain(examples, 'doors', 'E'))
    print(this.getInfoGain(examples, 'persons', 'E'))
    print(this.getInfoGain(examples, 'lug_boot', 'E'))
    print(this.getInfoGain(examples, 'safety', 'E'))
    return NotImplementedError
    #if all same label or attributes empty
    
    
    #else
    
  
  ##test this decision tree on examples from the given file
  def test(this, fileName):
    return NotImplementedError
  
  ##input: examples:  list of IDs of examples to calculate on
  ##       attribute: the attribute we are considering splitting by
  ##       version:   version of information gain to use:
  ##                  'E' = entropy
  ##                  'ME' = majority error
  ##                  'GI' = gini index
  def getInfoGain(this, examples, attribute, version):
    _S = 0
    if version == 'E':
      _S = this.getEntropy(examples) #total current entropy
    elif version == 'ME':
      _S = this.getME(examples) #total current majority error
    else:
      _S = this.getGI(examples) #total current gini index
    
    print(attribute + ' ' + str(_S))
    
    #find summation term of information gain
    sum = 0
    #loop through all values the attribute can take
    for v in this.attrDict[attribute]:
      #find S_v
      S_v = []
      for i in examples:
        #if the value in the data matches the value we are looking for
        if this.trainingData[i].attributes[attribute] == v:
          S_v.append(i)
      
      _S_v = 0
      if version == 'E':
        _S_v = this.getEntropy(S_v) #entropy of S_v
      elif version == 'ME':
        _S_v = this.getME(S_v) #majority error of S_v
      else:
        _S_v = this.getGI(S_v) #gini index of S_v
      
      print(v + ' ' + str(_S_v))
      
      sum += (len(S_v)/len(examples)) * _S_v #::::::::TODO will change with fractional counts, must check count
    
    return _S - sum
  
  ##get the entropy of the given set of examples
  def getEntropy(this, examples):
    sum = 0
    for k in this.labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.trainingData[i].label == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      #p_k is the porportion of examples with k as label
      p_k = kSum/len(examples)  #:::::::::::::::TODO will change with fractional counts, must check count
      if p_k > 0:
        sum += p_k * math.log(p_k, 2)
    
    return -sum
  
  ##get the majority error of the given set of examples
  def getME(this, examples):
    #build list of counts of each label
    kList = []
    for k in this.labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.trainingData[i].label == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      kList.append(kSum)
    
    majCount = max(kList)
    #returns |examples without majority label| / |examples|
    return (len(examples) - majCount) / len(examples) #:::::::::::::::TODO will change with fractional counts, must check count
  
  ##get the gini index of the given set of examples
  def getGI(this, examples):
    sum = 0
    for k in this.labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.trainingData[i].label == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      #p_k is the porportion of examples with k as label
      p_k = kSum/len(examples)  #:::::::::::::::TODO will change with fractional counts, must check count
      sum += p_k ** 2
    
    return 1 - sum
