import sys
sys.path.append('../')
import math
import numpy
import time
from DecisionTree.DecisionTree import *
from Basics.Basics import *

##Runs AdaBoost algorithm with decision stumps as classifier
##Only works when exactly two labels exist, first listed is considered positive, second is negative
##input:  trainFile: path to csv file containing training data
##        testFile:  path to csv file containing test data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
##returns: A function that takes an example as input and returns the label given by the algorithm
def AdaBoost(trainFile, testFile, attrs, attrDict, labels, T):
  data = extractData(trainFile, attrs)
  Tdata = extractData(testFile, attrs)
  
  votes = []
  trees = []
  
  for t in range(T+1):
    
    #train the decision stump
    tree = DecisionTree(attrs, attrDict, labels, None)
    tree.train(data, 'E', 1)
    trees.append(tree)
    
    #calculate error
    e = 0
    for example in data:
      if tree.getDecision(example) != example.getLabel():
        e += example.getWeight()
    
    #calculate vote
    vote = (1/2) * math.log((1-e)/e)
    votes.append(vote)
    
    #find values without normalization constant
    D = []
    for example in data:
      if tree.getDecision(example) == example.getLabel():
        D.append(example.getWeight() * math.exp(-vote))
      else:
        D.append(example.getWeight() * math.exp(vote))
    
    z = sum(D)
    
    #update weights in the examples using normalized weights
    for i in range(len(data)):
      data[i].setWeight(D[i] / z)
  
  def hypothesis(example):
    #evaluate sum of all trees
    sum = 0
    for t in range(T+1):
      if trees[t].getDecision(example) == labels[0]:
        sum += votes[t]
      else:
        sum += -(votes[t])
    #return the first label if positive, second if negative
    if numpy.sign(sum) == 1:
      return labels[0]
    return labels[1]
  
  return hypothesis

#return percentage error of H on data
def test(data, H):
  numWrong = 0
  for example in data:
    if H(example) != example.getLabel():
      numWrong += 1
  return 100 * numWrong / len(data)

dataAttrs = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 
               'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
dataDict = {
    'age':        [],
    'job':        ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                   'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
    'marital':    ['married', 'divorced', 'single'],
    'education':  ['unknown', 'secondary', 'primary', 'tertiary'],
    'default':    ['yes', 'no'],
    'balance':    [],
    'housing':    ['yes', 'no'],
    'loan':       ['yes', 'no'],
    'contact':    ['unknown', 'telephone', 'cellular'],
    'day':        [],
    'month':      ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration':   [],
    'campaign':   [],
    'pdays':      [],
    'previous':   [],
    'poutcome':   ['unknown', 'other', 'failure', 'success']
    }
dataLabels = ['yes', 'no']
times1 = []
times2 = []
for t in range(1, 501):
  H = AdaBoost('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, t)
  print(t, end='\t')
  print(test(extractData('./bank/train.csv', dataAttrs), H), end='\t')
  print(test(extractData('./bank/test.csv', dataAttrs), H))