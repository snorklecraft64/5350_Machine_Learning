import sys
sys.path.append('../')
sys.path.append('./')
from Basics.Basics import *
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
  nn.feedThrough(X[0])
  nn.printAllDerivatives(0)

if sys.argv[1] == '2':
  trainX, trainy = extractXY('bank-note/train.csv')
  testX, testy = extractXY('bank-note/test.csv')
  r_0 = 0.1
  d = .001
  T = 500
  
  print('width\ttrainError\ttestError')
  nn = NeuralNetwork(trainX, trainy, 5)
  nn.SGD(r_0, d, T)
  print('5', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 10)
  nn.SGD(r_0, d, T)
  print('10', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 25)
  nn.SGD(r_0, d, T)
  print('25', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 50)
  nn.SGD(r_0, d, T)
  print('50', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 100)
  nn.SGD(r_0, d, T)
  print('100', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))

if sys.argv[1] == '3':
  trainX, trainy = extractXY('bank-note/train.csv')
  testX, testy = extractXY('bank-note/test.csv')
  r_0 = 0.1
  d = .001
  T = 500
  
  print('width\ttrainError\ttestError')
  nn = NeuralNetwork(trainX, trainy, 5, zero=True)
  nn.SGD(r_0, d, T)
  print('5', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 10, zero=True)
  nn.SGD(r_0, d, T)
  print('10', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 25, zero=True)
  nn.SGD(r_0, d, T)
  print('25', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 50, zero=True)
  nn.SGD(r_0, d, T)
  print('50', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))
  
  nn = NeuralNetwork(trainX, trainy, 100, zero=True)
  nn.SGD(r_0, d, T)
  print('100', end='\t')
  print(nn.test(trainX, trainy), end='\t')
  print(nn.test(testX, testy))