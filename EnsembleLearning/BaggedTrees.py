import sys
sys.path.append('../')
import random
import numpy
from DecisionTree.DecisionTree import *
from Basics.Basics import *

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

for t in range(1, 501):
  H = baggedTrees('./bank/train.csv', dataAttrs, dataDict, dataLabels, t)

  print(t, end='\t')
  print(test(trainData, H), end='\t')
  print(test(testData, H))