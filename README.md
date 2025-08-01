# MLP
A minimalist multilayer perceptron framework

# Features
 - Fully connected feedforward neural network
 - Support for multiple hidden layers with tanh activation
 - Softmax output layer for classification
 - Cross-entropy loss
 - Manual backpropagation
 - Mini-batch gradient descent training
 - Numerical gradient checking utility
 - No external ML libraries — built with pure NumPy
 - Times faster than PyTorch for small projects

# Example usage
```py
import numpy as np
from MLP import *

x = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
W, b = init_weights([2, 3, 7, 5, 2])
W, b = train(x, y, W, b, iters=1000, lr=0.01, batch_size=4)
probs = predict(x, W, b)
print("Predictions:", probs)
gradient_check(x, y, W, b, layer_idx=0, i=0, j=0)
```
