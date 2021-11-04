import sys
sys.path.append('../')
sys.path.append('./')
from DecisionTree import *
from Basics.Basics import *

def testCar():
  dataAttrs = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
  dataDict = {
    'buying':   ['vhigh', 'high', 'med', 'low'],
    'maint':    ['vhigh', 'high', 'med', 'low'],
    'doors':    ['2', '3', '4', '5more'],
    'persons':  ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety':   ['low', 'med', 'high']
    }
  dataLabels = ['unacc', 'acc', 'good', 'vgood']
  
  trainData = extractData('./car/train.csv', dataAttrs)
  testData = extractData('./car/test.csv', dataAttrs)

  print('\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')

  print('Entropy', end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 1)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 2)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 3)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 4)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 5)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 6)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))

  print('Majority Error', end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 1)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 2)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 3)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 4)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 5)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 6)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))

  print('Gini Index', end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 1)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 2)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 3)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 4)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 5)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 6)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))

def testBankNoUnknown():
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
  
  print('\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 1)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 2)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 3)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 4)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 5)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 6)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 1)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 2)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 3)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 4)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 5)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 6)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 1)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 2)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 3)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 4)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 5)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 6)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))
  
  print('\t7\t\t8\t\t9\t\t10\t\t11\t\t12')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 7)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 8)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 9)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 10)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 11)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 12)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 7)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 8)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 9)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 10)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 11)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 12)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 7)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 8)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 9)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 10)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 11)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 12)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))
  
  print('\t13\t\t14\t\t15\t\t16')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 13)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 14)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 15)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train(trainData, 'E', 16)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 13)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 14)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 15)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train(trainData, 'ME', 16)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 13)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 14)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 15)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train(trainData, 'GI', 16)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))

def testBankWithUnknown():
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
  
  print('\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 1)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 2)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 3)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 4)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 5)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 6)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 1)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 2)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 3)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 4)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 5)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 6)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 1)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 2)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 3)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 4)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 5)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 6)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))
  
  print('\t7\t\t8\t\t9\t\t10\t\t11\t\t12')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 7)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 8)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 9)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 10)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 11)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 12)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 7)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 8)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 9)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 10)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 11)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 12)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 7)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 8)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 9)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 10)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 11)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 12)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))
  
  print('\t13\t\t14\t\t15\t\t16')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 13)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 14)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 15)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train(trainData, 'E', 16)
  print(round(ent.test(trainData), 2), end='\t')
  print(round(ent.test(testData), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 13)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 14)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 15)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train(trainData, 'ME', 16)
  print(round(me.test(trainData), 2), end='\t')
  print(round(me.test(testData), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 13)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 14)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 15)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train(trainData, 'GI', 16)
  print(round(gi.test(trainData), 2), end='\t')
  print(round(gi.test(testData), 2))

if sys.argv[1] == '1':
  testCar()
if sys.argv[1] == '2':
  testBankNoUnknown()
if sys.argv[1] == '3':
  testBankWithUnknown()