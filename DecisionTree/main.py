import sys
from DecisionTree import *

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

dTree = DecisionTree(dataAttrs, dataDict, dataLabels)
dTree.train('./car/train.csv', 'E', 6)