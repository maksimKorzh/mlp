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
# Also imports numpy
from MLP import *

# Inputs should be one-hot encoded
x = np.array([[0,0], [0,1], [1,0], [1,1]])

# Outputs are integers representing output category
y = np.array([0, 1, 1, 0])

# Number of input features
n_inputs = x.shape[1]

# Number of output categories
n_outputs = 2

# Init weights and biases
W, b = init_weights([n_inputs, 8, 4, n_outputs])

# Train the model
W, b = train(x, y, W, b, iters=1000, lr=0.01, batch_size=2)

# Predict results
pred = predict(x, W, b)
print('Predictions:', pred)

# Check if backprop math was correct
gradient_check(x, y, W, b, layer_idx=0, i=0, j=0)
```
