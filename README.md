# 5350_Machine_Learning
 This is a machine learning library developed by Erik Warling for CS5350/6350 in University of Utah
 
 Decision Trees:
 When creating a new decision tree object, 4 inputs are needed:
  - attrs: list of strings representing all possible attributes/features of the data
  - attrDict: a dictionary of attributes to their possible values (empty if the attribute is numerical)
  - labels: list of possible labels of the data
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