import numpy as np
from layer import Layer
from graphviz import Digraph


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

    return self.output
  
  def predictions(self):
    if(len(self.output.shape) == 1):
      for i in range(len(self.output)):
        if self.output[i] > 0.5:
          self.output[i] = 1
        else:
          self.output[i] = 0
    else :
      for i in range(len(self.output)):
        for j in range(len(self.output[i])):
          if self.output[i][j] > 0.5:
            self.output[i][j] = 1
          else:
            self.output[i][j] = 0

  def visualize(self):
    dot = Digraph(comment='FFNN')
    dot.attr(rankdir='LR', nodesep='1', ranksep='')
    dot.attr('node', shape='circle', width='0.4', height='0.4')
    
    # Input layer
    for i in range(len(self.input[0])):
        dot.node(f'input{i}', f'input{i}', color='#2ecc71')

    # Hidden layers
    for i in range(len(self.layers) - 1):
        for j in range(self.layers[i+1].neuron):
            dot.node(f'hidden{i}{j}', f'hidden{i}{j}', color='#e67e22')

        if i == 0:
            for j in range(len(self.input[0])):
                for k in range(self.layers[i+1].neuron):
                    weight = self.layers[i].weights[j][k]
                    dot.edge(f'input{j}', f'hidden{i}{k}', label=f'{weight:.2f}', color='#2ecc71')

        else:
            for j in range(self.layers[i].neuron):
                for k in range(self.layers[i+1].neuron):
                    weight = self.layers[i].weights[j][k]
                    dot.edge(f'hidden{i-1}{j}', f'hidden{i}{k}', label=f'{weight:.2f}', color='#e67e22')

    # Output layer
    for i in range(len(self.output[0])):
        dot.node(f'output{i}', f'output{i}', color='#f1c40f')

    for i in range(self.layers[-1].neuron):
        for j in range(len(self.output[0])):
            weight = self.layers[-1].weights[i][j]
            dot.edge(f'hidden{len(self.layers)-2}{i}', f'output{j}', label=f'{weight:.2f}', color='#f1c40f')

    # dot.node_attr.update(fontname='Helvetica', fontsize='14')
    # dot.edge_attr.update(fontname='Helvetica', fontsize='12')

    return dot
