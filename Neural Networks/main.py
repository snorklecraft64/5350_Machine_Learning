import sys
from NeuralNetwork import *

if sys.argv[1] == '1':
  layer1 = {
            (0,1): -1,
            (0,2): 1,
            (1,1): -2,
            (1,2): 2,
            (2,1): -3,
            (2,2): 3
           }
  
  layer2 = {
            (0,1): -1,
            (0,2): 1,
            (1,1): -2,
            (1,2): 2,
            (2,1): -3,
            (2,2): 3
           }
  
  layer3 = {
            (0,0): -1,
            (1,0): 2,
            (2,0): -1.5
           }
  
  X = [[1, 1, 1]]
  
  y = [1]
  
  nn = NeuralNetwork(X, y, 2, layer1, layer2, layer3)
  nn.feedThrough(0)
  nn.printAllDerivatives(0)