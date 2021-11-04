import sys
sys.path.append('../')
sys.path.append('./')
from GradientDescent import *

dataAttrs = [
             'Cement',
             'Slag',
             'Fly ash',
             'Water',
             'SP',
             'Coarse Aggr',
             'Fine Aggr',
            ]

if sys.argv[1] == '4':
  r = tuneBatch('./concrete/train.csv', dataAttrs.copy())
  weight = batch('./concrete/train.csv', dataAttrs.copy(), 14.981943657085, r, p=True)
  print(r)
  print(weight)

if sys.argv[1] == '5':
  print(stochastic('./concrete/train.csv', dataAttrs.copy(), 14.981943657085, 1/(2**18)))

if sys.argv[1] == '6':
  print(analyze('./concrete/train.csv', dataAttrs))