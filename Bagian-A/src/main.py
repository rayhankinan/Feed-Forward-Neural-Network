import numpy as np
import json
from ffnn import FFNN
from layer import Layer
from graphviz import Digraph

model = str(input("Masukkan nama file model (dalam format .json): "))
model = open(f"Bagian-A/models/{model}", "r")
model = json.load(model)

layers = model["case"]["model"]
weights = model["case"]["weights"]
inputArray = model["case"]["input"]

ffnn = FFNN(np.array(inputArray), [])
for i in range (len(layers["layers"])):
  layer = layers["layers"][i]
  weight = weights[i]
  new_layer = Layer(layer["number_of_neurons"], layer["activation_function"], np.array(weight[1:]), np.array(weight[0]))
  ffnn.new_layer(new_layer)

ffnn.forward()


print(ffnn.output)

# Create Graphviz digraph object
dot = ffnn.visualize()


# Save and render the graph
dot.render("ffnn_graph", format="png", cleanup=True)
