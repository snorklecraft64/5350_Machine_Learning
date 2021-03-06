import sys
import random
import numpy
import time
from DecisionTree.DecisionTree import *
from Basics.Basics import *

##Runs bagged tree algorithm with T trees
##Only works when exactly two labels exist, first listed is considered positive, second is negative
##input:  trainFile: path to csv file containing training data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
##        T:         amount of iterations
def baggedTrees(trainFile, attrs, attrDict, labels, T):
  data = extractData(trainFile, attrs)
  trees = []
  votes = []
  for t in range(T+1):
    #sample m examples with replacement
    examples = []
    for i in range(len(data)):
      examples.append(data[random.randint(0, len(data)-1)].copy())
    
    #learn tree
    tree = DecisionTree(attrs, attrDict, labels, None)
    tree.train(examples, 'E', len(attrs)+1)
    trees.append(tree)
    
    #calculate error
    e = 0
    for example in data:
      if tree.getDecision(example) != example.getLabel():
        e += example.getWeight()
    
    #calculate vote
    vote = (1/2) * math.log((1-e)/e)
    votes.append(vote)
  
  #vote the resulting trees
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

##Runs bagged trees algorithm for every t from 0 to T and prints the train and test error for each t
##prints 1 line per t in the form '<t>\t<train error>\t<test error>'
##input:  trainFile: path to csv file containing training data
##        testFile:  path to csv file containing test data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
##        T:         total amount of iterations
def baggedTreesBulk(trainFile, testFile, attrs, attrDict, labels, T):
  trainData = extractData(trainFile, attrs)
  testData = extractData(testFile, attrs)
  data = extractData(trainFile, attrs)
  trees = []
  votes = []
  for t in range(T+1):
    #sample m examples with replacement
    examples = []
    for i in range(len(data)):
      examples.append(data[random.randint(0, len(data)-1)].copy())
    
    #learn tree
    tree = DecisionTree(attrs, attrDict, labels, None)
    tree.train(examples, 'E', len(attrs)+1)
    trees.append(tree)
    
    #calculate error
    e = 0
    for example in data:
      if tree.getDecision(example) != example.getLabel():
        e += example.getWeight()
    
    #calculate vote
    vote = (1/2) * math.log((1-e)/e)
    votes.append(vote)
  
    #vote the resulting trees
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
    print(test(trainData, hypothesis), end='\t')
    print(test(testData, hypothesis))