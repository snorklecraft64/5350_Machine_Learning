import sys
import math
import statistics

##represents a single example in a training data set
class Example:
  def __init__(this, ID, count, attributes, label):
    this.__ID = ID
    #how much space this example takes in the training data
    this.__count = count
    #the attributes of this example (dictionary where keys are attributes and values are the value of that attribute)
    this.__attributes = attributes
    #the label given to this example
    this.__label = label
  
  def __str__(this):
    return '[' + str(this.__ID) + ', ' + str(this.__count) + ', ' + str(this.__attributes) + ', ' + str(this.__label) + ']'
  
  def __repr__(this):
    return this.__str__()
  
  def __hash__(this):
    return this.__ID

  def __eq__(this, other):
    if this.__ID == other.__ID:
      return true
    return false
  
  def getID(this):
    return this.__ID
  def getCount(this):
    return this.__count
  def getAttrs(this):
    return this.__attributes
  def getLabel(this):
    return this.__label

##represents a single node in a tree
class Node:
  ##creates node
  ##input:  isLeaf: if true, node will only have a label
  ##                if false, node will have attribute and dict of child nodes
  def __init__(this, attr, label):
      #the label this node has
      this.__label = label
      #dictionary that stores the child nodes of this node
      #the key is the value of the attribute, the value is the child node
      this.__childs = {}
      #the attribute this node splits by
      this.__attr = attr
  
  ##add a given node with given attribute value as a child of this node
  def add_child(this, c_node, a_value):
    this.__childs[a_value] = c_node
  
  ##returns the child node for the specific value given
  def get_child(this, a_value):
    return this.__childs[a_value]
  
  def getLabel(this):
    return this.__label
  
  def getAttr(this):
    return this.__attr

##represents a decision tree, initially untrained
class DecisionTree:
  
  ##input:  attrs:    list of attributes in the order they appear in the data
  ##        attrDict: dictionary of attributes to a list of their possible values
  ##                  an empty list indicates a numerical attribute
  ##        labels:   list of labels data can have
  ##DOES NOT MODIFY attrs, attrDict, OR labels
  def __init__(this, attrs, attrDict, labels):
    #list of training data, an example with ID x will be at trainingData[x]
    this.__trainingData = []
    this.__attrs = attrs.copy()
    this.__attrDict = attrDict.copy()
    this.__labels = labels.copy()
    #dictionary of numeric attributes to their thresholds
    this.__thresholds = {}
    this.__root = None #the root node of the tree

  ##returns a list contatining the examples from the given file
  def __extractData(this, fileName):
    ID = 0
    data = []
    with open(fileName, 'r') as f:
      for line in f:
        terms = line.strip().split(',')
        
        #process one training example :::::::TODO code for partial counts may go here (or maybe after all full ones have completed (if statement))
        #build the dictionary of attributes
        attributes = {}
        for i in range(len(this.__attrs)):
          attributes[this.__attrs[i]] = terms[i]
        #the label is the last in data set
        data.append(Example(ID, 1, attributes, terms[len(this.__attrs)]))
        ID += 1
    
    return data
  
  ##returns the threshold for the given attribute
  def __findThreshold(this, attribute):
    #build list of values to find the median
    values = []
    for ex in this.__trainingData:
      values.append(int(ex.getAttrs()[attribute]))
    
    return statistics.median(values)

  ##train this decision tree on examples from the given file
  ##input:  fileName: path to file containing data
  ##        version:  version of information gain to use:
  ##                  'E' = entropy
  ##                  'ME' = majority error
  ##                  'GI' = gini index
  ##        maxDepth: the max depth the tree should reach before stopping
  def train(this, fileName, version, maxDepth):
    this.__trainingData = this.__extractData(fileName)
    
    #change numeric attributes to binary
    for a in this.__attrs:
      #if a certain attributes value list is empty, it is numerical
      if not this.__attrDict[a]:
        threshold = this.__findThreshold(a)
        #update in the threshold dictionary for use in testing
        this.__thresholds[a] = threshold
        
        #update every example to make this attribute's value either above or below
        this.__attrDict[a] = ['above', 'below']
        for ex in this.__trainingData:
          if int(ex.getAttrs()[a]) >= threshold:
            ex.getAttrs()[a] = 'above'
          else:
            ex.getAttrs()[a] = 'below'
    
    #create list of all indexes for the initial call to ID3
    examples = []
    for i in range(len(this.__trainingData)):
      examples.append(i)
    
    this.__root = this.__ID3(examples, this.__attrs, version, maxDepth, 0)
  
  ##run the ID3 algorithm on the given set of examples, with the given set of attributes left to consider
  ##input:  examples:   list of rule IDs we are considering
  ##        attributes: list of attributes we are considering
  ##        version:    version of information gain to use:
  ##                    'E' = entropy
  ##                    'ME' = majority error
  ##                    'GI' = gini index
  ##        maxDepth:   the max depth the tree should reach before stopping
  ##returns: root node of the decision tree
  def __ID3(this, examples, attributes, version, maxDepth, currDepth):
    #find majority label by counting each label and finding the max
    kList = []
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      kList.append(kSum)
    
    majLabel = this.__labels[kList.index(max(kList))]
    
    #if attributes is empty, make leaf node with majority label
    if not attributes:
      return Node(None, majLabel)
    
    #if the count of the majority label is equal to the size of examples, then we know all examples have the same label
    if max(kList) == len(examples): #::::::::::::TODO will change with fractional counts (len(examples))
      return Node(None, majLabel)
    
    #otherwise, we must calculate info gain
    #find information gains of attributes in order they are given
    infoGains = [] 
    for a in attributes:
      infoGains.append(this.__getInfoGain(examples, a, version))
    
    #use attribute that has max info gain
    attr = attributes[infoGains.index(max(infoGains))]
  
    root = Node(attr, None)
    
    #find S_v for each value the attribute can take and decide to either make a leaf node or recurse accordingly
    for v in this.__attrDict[attr]:
      #find S_v
      S_v = []
      for i in examples:
        #if the value in the data matches the value we are looking for
        if this.__trainingData[i].getAttrs()[attr] == v:
          S_v.append(i)
      
      #if S_v is empty or we have reached the maxDepth, return leaf node with majority label
      if not S_v or currDepth == maxDepth:
        root.add_child(Node(None, majLabel), v)
      #otherwise, we need to recurse on S_v
      else:
        attrsCopy = attributes.copy()
        attrsCopy.remove(attr)
        root.add_child(this.__ID3(S_v, attrsCopy, version, maxDepth, currDepth + 1), v)
    
    return root
  
  ##test this decision tree on examples from the given file
  ##returns the error rate in percentage
  def test(this, fileName):
    testingData = this.__extractData(fileName)
    
    #change numeric attributes to binary
    for a in this.__attrs:
      #if the attribute is in the threshold dictionary, it is numeric
      if a in this.__thresholds.keys():
        #update every example to make this attribute's value either above or below
        for ex in testingData:
          if int(ex.getAttrs()[a]) >= this.__thresholds[a]:
            ex.getAttrs()[a] = 'above'
          else:
            ex.getAttrs()[a] = 'below'
    
    numWrong = 0
    for example in testingData:
      #traverse the tree and predict a label
      exLabel = this.__decide(this.__root, example)
      
      if exLabel != example.getLabel():
        numWrong += 1 #::::::::::::::TODO will change with partial count
    
    return 100 * (numWrong / len(testingData)) #::::::::::::::TODO will change with partial count
  
  ##traverses the tree starting at the node given according to the attributes of the given example
  ##returns the label this example should take according to the decision tree
  def __decide(this, node, example):
    #base case of reaching a leaf node
    if node.getAttr() == None:
      return node.getLabel()
    
    return this.__decide(node.get_child(example.getAttrs()[node.getAttr()]), example)
  
  ##input: examples:  list of IDs of examples to calculate on
  ##       attribute: the attribute we are considering splitting by
  ##       version:   version of information gain to use:
  ##                  'E' = entropy
  ##                  'ME' = majority error
  ##                  'GI' = gini index
  def __getInfoGain(this, examples, attribute, version):
    _S = 0
    if version == 'E':
      _S = this.__getEntropy(examples) #total current entropy
    elif version == 'ME':
      _S = this.__getME(examples) #total current majority error
    else:
      _S = this.__getGI(examples) #total current gini index
    
    #find summation term of information gain
    sum = 0
    #loop through all values the attribute can take
    for v in this.__attrDict[attribute]:
      #find S_v
      S_v = []
      for i in examples:
        #if the value in the data matches the value we are looking for
        if this.__trainingData[i].getAttrs()[attribute] == v:
          S_v.append(i)
      
      _S_v = 0
      if version == 'E':
        _S_v = this.__getEntropy(S_v) #entropy of S_v
      elif version == 'ME':
        _S_v = this.__getME(S_v) #majority error of S_v
      else:
        _S_v = this.__getGI(S_v) #gini index of S_v
      
      sum += (len(S_v)/len(examples)) * _S_v #::::::::TODO will change with fractional counts, must check count
    
    return _S - sum
  
  ##get the entropy of the given set of examples
  def __getEntropy(this, examples):
    #entropy is 0 if there are no examples
    if len(examples) == 0:
      return 0
    
    sum = 0
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      #p_k is the porportion of examples with k as label
      p_k = kSum/len(examples)  #:::::::::::::::TODO will change with fractional counts, must check count
      if p_k > 0:
        sum += p_k * math.log(p_k, 2)
    
    return -sum
  
  ##get the majority error of the given set of examples
  def __getME(this, examples):
    #ME is 0 if there are no examples
    if len(examples) == 0:
      return 0
    
    #build list of counts of each label
    kList = []
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      kList.append(kSum)
    
    majCount = max(kList)
    #returns |examples without majority label| / |examples|
    return (len(examples) - majCount) / len(examples) #:::::::::::::::TODO will change with fractional counts, must check count
  
  ##get the gini index of the given set of examples
  def __getGI(this, examples):
    #GI is 0 if there are no examples
    if len(examples) == 0:
      return 0
    
    sum = 0
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += 1 #:::::::::::::::TODO will change with fractional counts, must check count
      
      #p_k is the porportion of examples with k as label
      p_k = kSum/len(examples)  #:::::::::::::::TODO will change with fractional counts, must check count
      sum += p_k ** 2
    
    return 1 - sum
