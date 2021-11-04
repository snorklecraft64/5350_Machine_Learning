import sys
import math
import statistics
import random
from Basics.Basics import *

##represents a decision tree, initially untrained
class DecisionTree:
  
  ##input:  attrs:    list of attributes in the order they appear in the data
  ##        attrDict: dictionary of attributes to a list of their possible values
  ##                  an empty list indicates a numerical attribute
  ##        labels:   list of labels data can have
  ##DOES NOT MODIFY attrs, attrDict, OR labels
  def __init__(this, attrs, attrDict, labels, unknownVal):
    #list of training data, an example with ID x will be at trainingData[x]
    this.__trainingData = []
    this.__attrs = attrs.copy()
    this.__attrDict = attrDict.copy()
    #dictionary of the majority value of each attribute
    this.__majOfAttr = {}
    this.__unknownVal = unknownVal
    this.__labels = labels.copy()
    #dictionary of numeric attributes to their thresholds
    this.__thresholds = {}
    this.__root = None #the root node of the tree
  
  ##returns the threshold for the given attribute on the given data
  def __findThreshold(this, attribute, data):
    #build list of values to find the median
    values = []
    for ex in data:
      values.append(int(ex.getAttrs()[attribute]))
    
    return statistics.median(values)

  ##train this decision tree on examples from the given file
  ##input:  data:      list of Examples
  ##        version:   version of information gain to use:
  ##                   'E' = entropy
  ##                   'ME' = majority error
  ##                   'GI' = gini index
  ##        maxDepth:  the max depth the tree should reach before stopping
  def train(this, data, version, maxDepth):
    
    #change numeric attributes to binary
    for a in this.__attrs:
      #if a certain attributes value list is empty, it is numerical
      if not this.__attrDict[a]:
        this.__attrDict[a] = ['above', 'below']
        try:
          threshold = this.__findThreshold(a, data)
        except ValueError:
          continue
        #update in the threshold dictionary for use in testing
        this.__thresholds[a] = threshold
        
        #update every example to make this attribute's value either above or below
        for ex in data:
          if int(ex.getAttrs()[a]) >= threshold:
            ex.getAttrs()[a] = 'above'
          else:
            ex.getAttrs()[a] = 'below'
    
    #don't need to change unknown vals if we don't have an unknown val
    if this.__unknownVal != None:
      #split data into set of data with no unknown values, and set with unknown values
      fullData = []
      missingData = []
      for ex in data:
        unknownFound = False
        for a in this.__attrs:
          if ex.getAttrs()[a] == this.__unknownVal:
            unknownFound = True
            #no need to continue after finding an unknown val
            break
        
        if unknownFound:
          missingData.append(ex)
        else:
          fullData.append(ex)
      
      #calculate majority element of each attr in fullData
      for a in this.__attrs:
        vList = []
        for v in this.__attrDict[a]:
          count = 0
          for ex in fullData:
            if ex.getAttrs()[a] == v:
              count += 1
          vList.append(count)
        this.__majOfAttr[a] = this.__attrDict[a][vList.index(max(vList))]
      
      #go thru the missing data and replace any unknown vals with the majority val
      for ex in missingData:
        for a in this.__attrs:
          if ex.getAttrs()[a] == this.__unknownVal:
            ex.getAttrs()[a] = this.__majOfAttr[a]
      
      this.__trainingData = fullData + missingData
    else:
      this.__trainingData = data
    
    #create list of all indexes for the initial call to ID3
    examples = []
    for i in range(len(this.__trainingData)):
      examples.append(i)
    
    this.__root = this.__ID3(examples, this.__attrs, version, maxDepth, 0)
  
  ##train a random decision tree based on given examples
  ##input:  data:    list of Examples
  ##        version: version of information gain to use:
  ##                 'E' = entropy
  ##                 'ME' = majority error
  ##                 'GI' = gini index
  ##        subset:  the size of the subset
  def randTrain(this, data, version, subset):
    #change numeric attributes to binary
    for a in this.__attrs:
      #if a certain attributes value list is empty, it is numerical
      if not this.__attrDict[a]:
        this.__attrDict[a] = ['above', 'below']
        try:
          threshold = this.__findThreshold(a, data)
        except ValueError:
          continue
        #update in the threshold dictionary for use in testing
        this.__thresholds[a] = threshold
        
        #update every example to make this attribute's value either above or below
        for ex in data:
          if int(ex.getAttrs()[a]) >= threshold:
            ex.getAttrs()[a] = 'above'
          else:
            ex.getAttrs()[a] = 'below'
    
    #don't need to change unknown vals if we don't have an unknown val
    if this.__unknownVal != None:
      #split data into set of data with no unknown values, and set with unknown values
      fullData = []
      missingData = []
      for ex in data:
        unknownFound = False
        for a in this.__attrs:
          if ex.getAttrs()[a] == this.__unknownVal:
            unknownFound = True
            #no need to continue after finding an unknown val
            break
        
        if unknownFound:
          missingData.append(ex)
        else:
          fullData.append(ex)
      
      #calculate majority element of each attr in fullData
      for a in this.__attrs:
        vList = []
        for v in this.__attrDict[a]:
          count = 0
          for ex in fullData:
            if ex.getAttrs()[a] == v:
              count += 1
          vList.append(count)
        this.__majOfAttr[a] = this.__attrDict[a][vList.index(max(vList))]
      
      #go thru the missing data and replace any unknown vals with the majority val
      for ex in missingData:
        for a in this.__attrs:
          if ex.getAttrs()[a] == this.__unknownVal:
            ex.getAttrs()[a] = this.__majOfAttr[a]
      
      this.__trainingData = fullData + missingData
    else:
      this.__trainingData = data
    
    #create list of all indexes for the initial call to randLearn
    examples = []
    for i in range(len(this.__trainingData)):
      examples.append(i)
    
    this.__root = this.__randLearn(examples, this.__attrs, version, subset)

  ##run the ID3 algorithm but with randomly selected sample of attributes each node
  ##input:  examples:   list of rule IDs we are considering
  ##        attributes: list of attributes we are considering
  ##        version:    version of information gain to use:
  ##                    'E' = entropy
  ##                    'ME' = majority error
  ##                    'GI' = gini index
  ##        subset:     size of subset of attributes to use
  ##returns: root node of the decision tree
  def __randLearn(this, examples, attributes, version, subset):
    #find majority label by counting each label and finding the max
    kList = []
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += this.__trainingData[i].getWeight() 
      
      kList.append(kSum)
    
    majLabel = this.__labels[kList.index(max(kList))]
    
    #find total weight of all examples
    examplesW = 0
    for i in examples:
      examplesW += this.__trainingData[i].getWeight()
    
    #if attributes is empty OR
    #the count of the majority label is equal to the size of examples (thus all examples have the same label)
    #THEN, make leaf node with majority label
    if not attributes or max(kList) == examplesW:
      return Node(None, majLabel)
    
    #randomly get subset of attributes
    #only need to if subset is less than current left attributes
    attrSubset = []
    if subset < len(attributes):
      attrsCopy = attributes.copy()
      for i in range(subset):
        toAdd = attrsCopy[random.randint(0, len(attrsCopy)-1)]
        attrSubset.append(toAdd)
        attrsCopy.remove(toAdd)
    else:
      attrSubset = attributes
    
    #find information gains of subset of attributes in order they are given
    infoGains = [] 
    for a in attrSubset:
      infoGains.append(this.__getInfoGain(examples, a, version))
    
    #use attribute that has max info gain
    attr = attrSubset[infoGains.index(max(infoGains))]
  
    root = Node(attr, None)
    
    #find S_v for each value the attribute can take and decide to either make a leaf node or recurse accordingly
    for v in this.__attrDict[attr]:
      #find S_v
      S_v = []
      for i in examples:
        #if the value in the data matches the value we are looking for
        if this.__trainingData[i].getAttrs()[attr] == v:
          S_v.append(i)
      
      #if S_v is empty, make leaf node with majority label
      if not S_v:
        root.add_child(Node(None, majLabel), v)
      #otherwise, we need to recurse on S_v
      else:
        attrsCopy = attributes.copy()
        attrsCopy.remove(attr)
        root.add_child(this.__randLearn(S_v, attrsCopy, version, subset), v)
    
    return root
  
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
          kSum += this.__trainingData[i].getWeight() 
      
      kList.append(kSum)
    
    majLabel = this.__labels[kList.index(max(kList))]
    
    #find total weight of all examples
    examplesW = 0
    for i in examples:
      examplesW += this.__trainingData[i].getWeight()
    
    #if attributes is empty OR
    #we have reached the max depth OR
    #the count of the majority label is equal to the size of examples (thus all examples have the same label)
    #THEN, make leaf node with majority label
    if not attributes or currDepth == maxDepth or max(kList) == examplesW:
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
      
      #if S_v is empty, make leaf node with majority label
      if not S_v:
        root.add_child(Node(None, majLabel), v)
      #otherwise, we need to recurse on S_v
      else:
        attrsCopy = attributes.copy()
        attrsCopy.remove(attr)
        root.add_child(this.__ID3(S_v, attrsCopy, version, maxDepth, currDepth + 1), v)
    
    return root
  
  ##test this decision tree on given list of examples
  ##returns the error rate in percentage
  def test(this, data):
    
    if this.__unknownVal != None:
      for ex in data:
        for a in this.__attrs:
          if ex.getAttrs()[a] == this.__unknownVal:
            ex.getAttrs()[a] = this.__majOfAttr[a]
    
    #change numeric attributes to binary
    for a in this.__attrs:
      #if the attribute is in the threshold dictionary, it is numeric
      if a in this.__thresholds.keys():
        try:
          #update every example to make this attribute's value either above or below
          for ex in data:
            if int(ex.getAttrs()[a]) >= this.__thresholds[a]:
              ex.getAttrs()[a] = 'above'
            else:
              ex.getAttrs()[a] = 'below'
        except ValueError:
          break
      
    numWrong = 0
    for example in data:
      #traverse the tree and predict a label
      exLabel = this.__decide(this.__root, example)
      
      if exLabel != example.getLabel():
        numWrong += 1
    
    return 100 * (numWrong / len(data))
  
  ##returns the label this decision tree would give to the given example
  def getDecision(this, example):
    #change numeric attributes to binary
    for a in this.__attrs:
      #if the attribute is in the threshold dictionary, it is numeric
      if a in this.__thresholds.keys():
        #update the example to make this attribute's value either above or below
        try:
            if int(example.getAttrs()[a]) >= this.__thresholds[a]:
              example.getAttrs()[a] = 'above'
            else:
              example.getAttrs()[a] = 'below'
        except ValueError:
            break
    
    return this.__decide(this.__root, example)
  
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
    #find total weight of examples
    examplesW = 0
    for i in examples:
      examplesW += this.__trainingData[i].getWeight()
  
    _S = 0
    if version == 'E':
      _S = this.__getEntropy(examples, examplesW) #total current entropy
    elif version == 'ME':
      _S = this.__getME(examples, examplesW) #total current majority error
    else:
      _S = this.__getGI(examples, examplesW) #total current gini index
    
    #find summation term of information gain
    sum = 0
    #loop through all values the attribute can take
    for v in this.__attrDict[attribute]:
      #find S_v and its total weight
      S_v = []
      S_vW = 0
      for i in examples:
        #if the value in the data matches the value we are looking for
        if this.__trainingData[i].getAttrs()[attribute] == v:
          S_v.append(i)
          S_vW += this.__trainingData[i].getWeight()
      
      _S_v = 0
      if version == 'E':
        _S_v = this.__getEntropy(S_v, S_vW) #entropy of S_v
      elif version == 'ME':
        _S_v = this.__getME(S_v, S_vW) #majority error of S_v
      else:
        _S_v = this.__getGI(S_v, S_vW) #gini index of S_v
      
      sum += (S_vW / examplesW) * _S_v
    
    return _S - sum
  
  ##get the entropy of the given set of examples
  ##totalW: the total weight of all examples
  def __getEntropy(this, examples, totalW):
    #entropy is 0 if there are no examples
    if len(examples) == 0:
      return 0
    
    sum = 0
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += this.__trainingData[i].getWeight()
      
      p_k = kSum / totalW
      if p_k > 0:
        sum += p_k * math.log(p_k, 2)
    
    return -sum
  
  ##get the majority error of the given set of examples
  ##totalW: the total weight of all examples
  def __getME(this, examples, totalW):
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
          kSum += this.__trainingData[i].getWeight()
      
      kList.append(kSum / totalW)
    
    majCount = max(kList)
    #returns |examples without majority label| / |examples|
    return (totalW - majCount)# / totalW
  
  ##get the gini index of the given set of examples
  ##totalW: the total weight of all examples
  def __getGI(this, examples, totalW):
    #GI is 0 if there are no examples
    if len(examples) == 0:
      return 0
    
    sum = 0
    for k in this.__labels:
      #count how many have this label
      kSum = 0
      for i in examples:
        if this.__trainingData[i].getLabel() == k:
          kSum += this.__trainingData[i].getWeight()
      
      p_k = kSum / totalW
      sum += p_k ** 2
    
    return 1 - sum
