# Baseline
Vanilla version of PC (MLP) - an ANN with modularity. 

Features:
  - Strictly based on MLP and the code from http://deeplearning.net/.
  - Circuits are equally divided.
  - Outside dropout rate and sparsity coefficient, other circuit descriptions are similar
  - Weight/Learning rate decay is not included

Main files:
  - execute.py      : experiment configuration file
  - model.py        : PC definition
  - layer.py        : layer defintion  
