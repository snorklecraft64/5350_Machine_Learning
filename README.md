# 5350_Machine_Learning
 This is a machine learning library developed by Erik Warling for CS5350/6350 in University of Utah
 
 runHW1.sh
 - Takes a parameter:
   1: get answer to Q2b
   2: get answer to Q3a
   3: get answer to Q3b

 runHW2.sh
 - Takes a parameter:
   1: get answer to Q2a
   2: get answer to Q2b
   3: get answer to Q2d
   4: get answer to Q4a (last line is weight, line before is r, rest is loss at each t)
   5: get answer to Q4b (first line is loss, second is weight)
   6: get answer to Q4c
 
 runHW3.sh
 - Takes a parameter:
   1: get answer to Q2a
   2: get answer to Q2b
   3: get answer to Q2c

 runHW4.sh
 - Takes a parameter:
   1: get answer to Q2a
   2: get answer to Q2b
   3: get answer to Q3a
   4: get answers to Q3b and Q3c

 Decision Trees:
  When creating a new decision tree object, 4 inputs are needed:
   - attrs:      list of strings representing all possible attributes/features of the data
   - attrDict:   a dictionary of attributes to their possible values (empty if the attribute is numerical)
   - labels:     list of possible labels of the data
   - unknownVal: the value that denotes that a particular attribute value is unknown (None if you want to consider the unkown its own value)
 
  To train the decision tree, call the 'train' method on the decision tree object with these inputs:
   - data: list of examples you want to train on
   - version: the version of information gain to use
      ^ 'E' = entropy
      ^ 'ME' = majority error
      ^ 'GI' = gini index
   - maxDepth: the max depth wanted for the tree (use 1 for a stump)

  To test the decision tree, call the 'test' method on the decision tree object with these inputs:
   - data: list of examples you want to test on

 AdaBoost:
  To run the AdaBoost method, need 5 inputs:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the data
  - attrDict:  dictionary of attributes to a list of their possible values, empty list for numerical
  - labels:    list of labels data can have
  - T:         amount of iterations to run
  Returns a function thats takes an example as input and returns the prediction the model would make.

  There is an additional method called AdaBoostBulk which runs AdaBoost and prints the train and test data at each t. Takes an additional argument testFile, which is the path to the csv file to test on.

 Bagged Trees:
  To run the baggedTrees method, need 5 inputs:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the data
  - attrDict:  dictionary of attributes to a list of their possible values, empty list for numerical
  - labels:    list of labels data can have
  - T:         amount of iterations to run
  Returns a function thats takes an example as input and returns the prediction the model would make.
  
  There is an additional method called BaggedTreesBulk which runs baggedTrees and prints the train and test data at each t. Takes an additional argument testFile, which is the path to the csv file to test on.

 Random Forests:
  To run the randomForests method, need 6 inputs:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the data
  - attrDict:  dictionary of attributes to a list of their possible values, empty list for numerical
  - labels:    list of labels data can have
  - T:         amount of iterations to run
  - subset:    the size of subset used when deciding what attribute to split
  Returns a function thats takes an example as input and returns the prediction the model would make.

  There is an additional method called RandomForestsBulk which runs randomForests and prints the train and test data at each t. Takes an additional argument testFile, which is the path to the csv file to test on.

 Batch Gradient Descent:
  To run the batch method, need 4 inputs, with 1 optional input:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the data
  - threshold: error threshold to stop when below
  - r:         learning rate
  - p:         (optional) bool, true when want to print the loss at each t, false if not (false by default)
  Returns the final optimized weight vector
  
  There is an additional method called tuneBatch, which takes in only the trainFile and attrs, and returns a learning rate such that batch graidient descent converges.

 Stochastic Gradient Descent:
  To run the stochastic method, need 4 inputs, with 1 optional input:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the data
  - threshold: error threshold to stop when below
  - r:         learning rate
  - p:         (optional) bool, true when want to print the loss at each t, false if not (false by default)
  Returns the final optimized weight vector

 Analyze Loss:
  The method analyze uses the analysis method for determining the minimum loss weight vector. Takes in 2 inputs:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the data
  Returns the optimal weight vector

 Perceptron:
  Three perceptron methods:
   stdPercep:   run standard perceptron
   votedPercep: run voted perceptron
   avgPercep:   run averaged perceptron

  To run stdPercep method, takes 4 parameters, with 1 optional:
  - trainFile: path to csv file to train on
  - attrs:     list of attributes in the order they appear in the trainFile
  - T:         max epochs
  - posLabel:  the label in the data that should be considered positive
  - negLabel:  the label in the data that should be considered negative
  - seed:      (OPTIONAL) seed for RNG
  returns weight vector as a list

  To run votedPercep method, takes same parameters as stdPercep
  returns a method that takes in a numpy array and returns the prediction as -1 or 1

  To run avgPercep method, takes same parameters as stdPercep
  returns weight vector as numpy array
  
  SVM:
   Three SVM methods:
    SVMPrime:     optimize primal SVM
	SVMDual:      optimize dual SVM
	SVMDualGauss: optimize dual SVM using guassian kernel
   
   To run SVMPrime method, takes 8 parameters, with 2 optional:
   - trainFile: path to csv file to train on
   - attrs:     list of attributes in the order they appear in the trainFile
   - T:         max epochs
   - posLabel:  the label in the data that should be considered positive
   - negLabel:  the label in the data that should be considered negative
   - C:         the C hyperparameter
   - r_0:       the initial learning rate
   - version:   which version of learning rate schedule to use:
                ^ 1: r = r_0 / (1+(r_0/a)*t)
				^ 2: r = r_0 / (1+t)
   - a:         (optional) needed for version 1 learning rate schedule, if using version 2, it does not need to be specified
   - seed:      (optional) seed for RNG
   returns weight vector with the last element being the bias
   
   To run SVMDual method, takes 5 parameters, with 1 optional:
   - trainFile: path to csv file to train on
   - attrs:     list of attributes in the order they appear in the trainFile
   - posLabel:  the label in the data that should be considered positive
   - negLabel:  the label in the data that should be considered negative
   - C:         the C hyperparameter
   - seed:      (optional) seed for RNG
   returns a tuple of 2 elements where the first element is the weight vector, and the second element is the bias
   
   To run SVMDualGauss method, takes 6 parameters, with 1 optional:
   - trainFile: path to csv file to train on
   - attrs:     list of attributes in the order they appear in the trainFile
   - posLabel:  the label in the data that should be considered positive
   - negLabel:  the label in the data that should be considered negative
   - C:         the C hyperparameter
   - gamma:     gamma parameter for the gaussian kernel
   - seed:      (optional) seed for RNG
   returns a tuple of 3 elements where the first element is a function that returns the weight vector times a given example, the second element is the bias, and the third element is a list of the indices of the support vectors