import numpy as np
from layer import Layer

class FFNN:
  def __init__ (self, input: np.array, layers: list):
    self.input = input
    self.layers = layers
    self.output = None
  
  def new_layer(self, layer: Layer):
    self.layers.append(layer)

  def forward(self):
    self.output = self.input
    for layer in self.layers:
      self.output = layer.forward(self.output)
    self.predictions()

    return self.output
  
  def predictions(self):
    for i in range(len(self.output)):
      if self.output[i] > 0.5:
        self.output[i] = 1
      else:
        self.output[i] = 0