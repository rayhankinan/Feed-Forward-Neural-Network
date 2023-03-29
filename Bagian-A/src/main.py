import numpy as np
import json
from ffnn import FFNN
from layer import Layer

model = str(input("Masukkan nama file model (dalam format .json): "))
model = open(f"Bagian-A/models/{model}", "r")
model = json.load(model)

ffnn = FFNN(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), [])
for layer in model["data"]:
  new_layer = Layer(layer["num_neurons"], layer["function"], np.array(layer["weights"]), np.array(layer["bias"]))
  ffnn.new_layer(new_layer)

ffnn.forward()
print(ffnn.output)