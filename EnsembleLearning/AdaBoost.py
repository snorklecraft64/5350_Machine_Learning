import sys
import math
import numpy
import time
from DecisionTree.DecisionTree import *
from Basics.Basics import *

##Runs AdaBoost algorithm with decision stumps as classifier
##Only works when exactly two labels exist, first listed is considered positive, second is negative
##input:  trainFile: path to csv file containing training data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
##returns: A function that takes an example as input and returns the label given by the algorithm
def AdaBoost(trainFile, attrs, attrDict, labels, T):
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

##Runs adaboost algorithm for every t from 0 to T and prints the train and test error
##prints 1 line per t in the form '<t>\t<train error>\t<test error>'
##input:  trainFile: path to csv file containing training data
##        testFile:  path to csv file containing test data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
def AdaBoostBulk(trainFile, testFile, attrs, attrDict, labels, T):
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
      for j in range(t):
        if trees[j].getDecision(example) == labels[0]:
          sum += votes[j]
        else:
          sum += -(votes[j])
      #return the first label if positive, second if negative
      if numpy.sign(sum) == 1:
        return labels[0]
      return labels[1]
    
    print(t, end='\t')
    print(test(extractData(trainFile, attrs), hypothesis), end='\t')
    print(test(extractData(testFile, attrs), hypothesis))