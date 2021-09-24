import sys
from DecisionTree import *

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

  print('\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')

  print('Entropy', end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./car/train.csv', 'E', 1)
  print(round(ent.test('./car/train.csv'), 2), end='\t')
  print(round(ent.test('./car/test.csv'), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./car/train.csv', 'E', 2)
  print(round(ent.test('./car/train.csv'), 2), end='\t')
  print(round(ent.test('./car/test.csv'), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./car/train.csv', 'E', 3)
  print(round(ent.test('./car/train.csv'), 2), end='\t')
  print(round(ent.test('./car/test.csv'), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./car/train.csv', 'E', 4)
  print(round(ent.test('./car/train.csv'), 2), end='\t')
  print(round(ent.test('./car/test.csv'), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./car/train.csv', 'E', 5)
  print(round(ent.test('./car/train.csv'), 2), end='\t')
  print(round(ent.test('./car/test.csv'), 2), end='\t')

  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./car/train.csv', 'E', 6)
  print(round(ent.test('./car/train.csv'), 2), end='\t')
  print(round(ent.test('./car/test.csv'), 2))

  print('Majority Error', end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./car/train.csv', 'ME', 1)
  print(round(me.test('./car/train.csv'), 2), end='\t')
  print(round(me.test('./car/test.csv'), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./car/train.csv', 'ME', 2)
  print(round(me.test('./car/train.csv'), 2), end='\t')
  print(round(me.test('./car/test.csv'), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./car/train.csv', 'ME', 3)
  print(round(me.test('./car/train.csv'), 2), end='\t')
  print(round(me.test('./car/test.csv'), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./car/train.csv', 'ME', 4)
  print(round(me.test('./car/train.csv'), 2), end='\t')
  print(round(me.test('./car/test.csv'), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./car/train.csv', 'ME', 5)
  print(round(me.test('./car/train.csv'), 2), end='\t')
  print(round(me.test('./car/test.csv'), 2), end='\t')

  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./car/train.csv', 'ME', 6)
  print(round(me.test('./car/train.csv'), 2), end='\t')
  print(round(me.test('./car/test.csv'), 2))

  print('Gini Index', end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./car/train.csv', 'GI', 1)
  print(round(gi.test('./car/train.csv'), 2), end='\t')
  print(round(gi.test('./car/test.csv'), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./car/train.csv', 'GI', 2)
  print(round(gi.test('./car/train.csv'), 2), end='\t')
  print(round(gi.test('./car/test.csv'), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./car/train.csv', 'GI', 3)
  print(round(gi.test('./car/train.csv'), 2), end='\t')
  print(round(gi.test('./car/test.csv'), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./car/train.csv', 'GI', 4)
  print(round(gi.test('./car/train.csv'), 2), end='\t')
  print(round(gi.test('./car/test.csv'), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./car/train.csv', 'GI', 5)
  print(round(gi.test('./car/train.csv'), 2), end='\t')
  print(round(gi.test('./car/test.csv'), 2), end='\t')

  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./car/train.csv', 'GI', 6)
  print(round(gi.test('./car/train.csv'), 2), end='\t')
  print(round(gi.test('./car/test.csv'), 2))

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
  
  print('\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 1)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 2)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 3)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 4)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 5)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 6)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 1)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 2)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 3)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 4)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 5)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 6)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 1)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 2)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 3)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 4)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 5)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 6)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2))
  
  print('\t7\t\t8\t\t9\t\t10\t\t11\t\t12')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 7)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 8)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 9)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 10)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 11)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 12)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 7)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 8)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 9)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 10)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 11)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 12)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 7)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 8)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 9)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 10)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 11)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 12)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2))
  
  print('\t13\t\t14\t\t15\t\t16')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 13)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 14)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 15)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  ent.train('./bank/train.csv', 'E', 16)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 13)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 14)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 15)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  me.train('./bank/train.csv', 'ME', 16)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 13)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 14)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 15)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, None)
  gi.train('./bank/train.csv', 'GI', 16)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2))

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
  
  print('\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 1)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 2)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 3)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 4)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 5)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 6)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 1)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 2)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 3)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 4)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 5)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 6)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 1)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 2)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 3)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 4)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 5)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 6)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2))
  
  print('\t7\t\t8\t\t9\t\t10\t\t11\t\t12')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 7)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 8)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 9)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 10)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 11)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 12)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 7)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 8)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 9)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 10)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 11)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 12)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 7)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 8)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 9)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 10)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 11)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 12)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2))
  
  print('\t13\t\t14\t\t15\t\t16')
  print('\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest\ttrain\ttest')
  
  print('Entropy', end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 13)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 14)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 15)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2), end='\t')
  
  ent = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  ent.train('./bank/train.csv', 'E', 16)
  print(round(ent.test('./bank/train.csv'), 2), end='\t')
  print(round(ent.test('./bank/test.csv'), 2))
  
  print('Majority Error', end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 13)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 14)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 15)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2), end='\t')
  
  me = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  me.train('./bank/train.csv', 'ME', 16)
  print(round(me.test('./bank/train.csv'), 2), end='\t')
  print(round(me.test('./bank/test.csv'), 2))
  
  print('Gini Index', end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 13)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 14)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 15)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2), end='\t')
  
  gi = DecisionTree(dataAttrs, dataDict, dataLabels, 'unknown')
  gi.train('./bank/train.csv', 'GI', 16)
  print(round(gi.test('./bank/train.csv'), 2), end='\t')
  print(round(gi.test('./bank/test.csv'), 2))

testCar()
testBankNoUnknown()
testBankWithUnknown()