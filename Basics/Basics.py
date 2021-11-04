import sys

##represents a single example in a training data set
class Example:
  def __init__(this, ID, weight, attributes, label):
    this.__ID = ID
    #the weight of this example
    this.__weight = weight
    #the attributes of this example (dictionary where keys are attributes and values are the value of that attribute)
    this.__attributes = attributes
    #the label given to this example
    this.__label = label
  
  def __str__(this):
    return '[' + str(this.__ID) + ', ' + str(this.__weight) + ', ' + str(this.__attributes) + ', ' + str(this.__label) + ']'
  
  def __repr__(this):
    return this.__str__()
  
  def __hash__(this):
    return this.__ID

  def __eq__(this, other):
    if this.__ID == other.__ID:
      return true
    return false
  
  def copy(this):
    return Example(this.__ID, this.__weight, this.__attributes.copy(), this.__label)
  
  def getID(this):
    return this.__ID
  def getAttrs(this):
    return this.__attributes
  def getLabel(this):
    return this.__label
  def getWeight(this):
    return this.__weight
  def setWeight(this, weight):
    this.__weight = weight

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

##returns a list of Examples from the given csv file which has attribute values from attrs
##FOR PUBLIC USE
##use this to extract data for use in other functions
##defaults all weights to 1/m
##changes all numeric attributes to 
def extractData(fileName, attrs):
  numlines = sum(1 for line in open(fileName, 'r'))
  
  data = []
  with open(fileName, 'r') as f:
    ID = 0
    for line in f:
      terms = line.strip().split(',')
      
      weight = 1 / numlines
      attributes = {}
      for i in range(len(terms)-1):
        attributes[attrs[i]] = terms[i]
      label = terms[len(terms)-1]
      data.append(Example(ID, weight, attributes, label))
      ID += 1
  
  return data
  
##return percentage error of a hypothesis H on data
def test(data, H):
  numWrong = 0
  for example in data:
    if H(example) != example.getLabel():
      numWrong += 1
  return 100 * numWrong / len(data)