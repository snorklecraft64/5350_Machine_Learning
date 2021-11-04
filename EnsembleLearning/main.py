import sys
sys.path.append('../')
sys.path.append('./')
from AdaBoost import *
from BaggedTrees import *
from RandomForests import *

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

if sys.argv[1] == '1':
  AdaBoostBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, 500)

if sys.argv[1] == '2':
  baggedTreesBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, 500)

if sys.argv[1] == '3':
  print('|G| = 2')
  H = randomForestsBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, 500, 2)
  print('|G| = 4')
  H = randomForestsBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, 500, 4)
  print('|G| = 6')
  H = randomForestsBulk('./bank/train.csv', './bank/test.csv', dataAttrs, dataDict, dataLabels, 500, 6)