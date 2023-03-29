import numpy as np
from activation import activation

class Layer:
  def __init__(self, neuron: int, function: str, weights: np.array, bias: np.array):
    self.neuron = neuron
    self.weights = weights
    self.bias = bias
    if (function not in activation):
      raise Exception('Activation function not found')
    else:
      self.str_function = function
      if self.str_function == 'sigmoid':
        self.function = lambda x: 1 / (1 + np.exp(-x))
      elif self.str_function == 'relu':
        self.function = lambda x: np.maximum(0, x)
      elif self.str_function == 'linear':
        self.function = lambda x: x
      else:
        self.function = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)

  def forward(self, input: np.array):
    self.output = self.function(np.dot(input, self.weights) + self.bias)
    return self.output
    