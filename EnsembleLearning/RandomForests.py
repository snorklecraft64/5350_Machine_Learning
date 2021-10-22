import sys
sys.path.append('../')
import random
import numpy
import time
from DecisionTree.DecisionTree import *
from Basics.Basics import *

##Runs random forest algorithm with T trees
##Only works when exactly two labels exist, first listed is considered positive, second is negative
##input:  trainFile: path to csv file containing training data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
##        T:         amount of iterations
def randomForests(trainFile, attrs, attrDict, labels, T, subset):
  data = extractData(trainFile, attrs)
  trees = []
  votes = []
  
  for t in range(T+1):
    #sample m examples with replacement
    examples = []
    for i in range(len(data)):
      examples.append(data[random.randint(0, len(data)-1)].copy())
    
    #learn decision tree
    tree = DecisionTree(attrs, attrDict, labels, None)
    tree.randTrain(data, 'E', subset)
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

##Runs random forest algorithm for every t from 0 to T and prints the train and test error for each t
##prints 1 line per t in the form '<t>\t<train error>\t<test error>'
##input:  trainFile: path to csv file containing training data
##        testFile:  path to csv file containing test data
##        attrs:     list of attributes in the order they appear in the data
##        attrDict:  dictionary of attributes to a list of their possible values
##                   an empty list indicates a numerical attribute
##        labels:    list of labels data can have
##        T:         total amount of iterations
def randomForestsBulk(trainFile, testFile, attrs, attrDict, labels, T, subset):
  trainData = extractData(trainFile, attrs)
  testData = extractData(testFile, attrs)
  data = extractData(trainFile, attrs)
  trees = []
  votes = []
  
  for t in range(1, T+1):
    #sample m examples with replacement
    examples = []
    for i in range(len(data)):
      examples.append(data[random.randint(0, len(data)-1)].copy())
    
    #learn decision tree
    tree = DecisionTree(attrs, attrDict, labels, None)
    tree.randTrain(data, 'E', subset)
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

trainData = extractData('./bank/train.csv', dataAttrs)
testData = extractData('./bank/test.csv', dataAttrs)

#for t in range(1, 101):
t = 500
print('|G| = 2')
H = randomForestsBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, t, 2)
print('|G| = 4')
H = randomForestsBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, t, 4)
print('|G| = 6')
H = randomForestsBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, t, 6)