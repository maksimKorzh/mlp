#
# MLP, General Purpose Framework
#
#     Assumes inputs to be
#       one-hot encoded
#

# Packages
import numpy as np

# Same results across different platforms
np.random.seed(1234)

# Weights
def init_weights(layer_sizes):
  if len(layer_sizes) < 2:
    raise ValueError('ERROR: Neural net must have at least input and output layers!')
  weights = []
  biases = []
  n_inputs = layer_sizes[0]
  n_outputs = layer_sizes[-1]
  for i in range(len(layer_sizes)-1):
    x_shape = layer_sizes[i]
    y_shape = layer_sizes[i+1]
    W = np.random.randn(x_shape, y_shape)
    b = np.random.randn(1, y_shape)
    weights.append(W)
    biases.append(b)
    print(f"Layer {i}: W shape {W.shape}, b shape {b.shape}")
  return weights, biases

# Forward pass
def forward(x, y, weights, biases):
  a = x
  layers = []
  activations = [a]
  for i in range(len(weights) - 1):
    l = a @ weights[i] + biases[i]
    layers.append(l)
    a = np.tanh(l)
    activations.append(a)
  o = a @ weights[-1] + biases[-1]
  layers.append(o)
  exp_o = np.exp(o - np.max(o, axis=1, keepdims=True))
  probs = exp_o / np.sum(exp_o, axis=1, keepdims=True)
  activations.append(probs)
  batch_size = x.shape[0]
  clipped_probs = np.clip(probs, 1e-15, 1 - 1e-15)
  correct_logprobs = -np.log(clipped_probs[np.arange(batch_size), y])
  loss = np.sum(correct_logprobs) / batch_size
  return probs, loss, layers, activations

# Backward pass
def backward(x, y, weights, biases, layers, activations):
  batch_size = x.shape[0]
  num_classes = activations[-1].shape[1]
  output_error = activations[-1].copy()
  output_error[np.arange(batch_size), y] -= 1
  output_error /= batch_size
  W_grad = [None] * len(weights)
  b_grad = [None] * len(biases)
  W_grad[-1] = activations[-2].T @ output_error
  b_grad[-1] = np.sum(output_error, axis=0, keepdims=True)
  next_error = output_error @ weights[-1].T
  for i in reversed(range(len(weights) - 1)):
    layer_derivative = 1 - np.tanh(layers[i]) ** 2
    current_error = next_error * layer_derivative
    W_grad[i] = activations[i].T @ current_error
    b_grad[i] = np.sum(current_error, axis=0, keepdims=True)
    if i > 0: next_error = current_error @ weights[i].T
  return W_grad, b_grad

def step(weights, biases, W_grad, b_grad, lr):
  for i in range(len(weights)):
    weights[i] -= lr * W_grad[i]
    biases[i] -= lr * b_grad[i]

def get_batch(X, Y, batch_size):
  batch_size = min(batch_size, X.shape[0])
  indices = np.random.choice(X.shape[0], batch_size, replace=False)
  return X[indices], Y[indices]

def train(x, y, W, b, iters, lr, batch_size):
  for i in range(iters):
    xb, yb = get_batch(x, y, batch_size)
    probs, loss, layers, activations = forward(xb, yb, W, b)
    print(f"Iteration {i+1}, Loss: {loss:.4f}")
    W_grad, b_grad = backward(xb, yb, W, b, layers, activations)
    step(W, b, W_grad, b_grad, lr)
  return W, b

def predict(x, weights, biases):
  a = x
  for i in range(len(weights)-1):
    l = a @ weights[i] + biases[i]
    a = np.tanh(l)
  o = a @ weights[-1] + biases[-1]
  exp_o = np.exp(o - np.max(o, axis=1, keepdims=True))
  probs = exp_o / np.sum(exp_o, axis=1, keepdims=True)
  return np.argmax(probs, axis=1)

def numerical_gradient(x, y, weights, biases, layer_idx, i, j, epsilon=1e-5):
  original_value = weights[layer_idx][i, j]
  weights[layer_idx][i, j] = original_value + epsilon
  _, loss_plus, _, _ = forward(x, y, weights, biases)
  weights[layer_idx][i, j] = original_value - epsilon
  _, loss_minus, _, _ = forward(x, y, weights, biases)
  weights[layer_idx][i, j] = original_value
  grad_approx = (loss_plus - loss_minus) / (2 * epsilon)
  return grad_approx

def gradient_check(x, y, weights, biases, layer_idx=0, i=0, j=0):
  probs, loss, layers, activations = forward(x, y, weights, biases)
  W_grad, b_grad = backward(x, y, weights, biases, layers, activations)
  grad_num = numerical_gradient(x, y, weights, biases, layer_idx, i, j)
  grad_backprop = W_grad[layer_idx][i, j]
  print(f"Numerical grad: {grad_num:.8f}")
  print(f"Backprop grad:  {grad_backprop:.8f}")
  diff = abs(grad_num - grad_backprop)
  print(f"Difference:     {diff:.8e}")
  return diff
