from random import random
import json
import fileio

class Neuron:
  _bias = "bias"
  _weights = "weights"
  def __init__(self, input_size):
    self.weights = [random() for x in range(input_size)]
    self.bias = 0.0

  def __getitem__(self, index):
    return self.weights[index]

class Layer:
  def __init__(self, num_inputs, num_neurons):
    self.neurons = [Neuron(num_inputs) for x in range(num_neurons)]

  def __getitem__(self, index):
    return self.neurons[index]

  @classmethod
  def IOLayer(cls, num_inputs):
    return Layer(num_neurons=num_inputs, num_inputs=num_inputs)

class RNN:
  def __init__(self, num_inputs, num_hidden_layers, num_neurons_in_hidden):
    input_layer = Layer.IOLayer(num_inputs)
    output_layer = Layer.IOLayer(num_inputs)

    self.layers = [input_layer]
    for i in range(num_hidden_layers):
      self.layers.append(Layer(num_neurons_in_hidden, num_inputs))
    self.layers.append(output_layer)

  @classmethod
  def LoadRNN(cls, file):
    rnn = RNN(0, 0, 0)
    serialized_rnn = fileio.read_json(file)
    rnn = RNN.deserialize(serialized_rnn)
    return rnn

  def __getitem__(self, index):
    return self.layers[index]

  def write_json(self, file):
    fileio.write_json(self.serialize(), file)
    return

  def serialize(self):
    rnn = []
    for layer in self.layers:
      rnn.append([n.__dict__ for n in layer.neurons])
    return json.dumps(rnn)

  @classmethod
  def deserialize(cls, serialized_rnn):
    rnn = json.loads(serialized_rnn) if isinstance(serialized_rnn, str) else serialized_rnn

    for l in range(0, len(rnn)):
      layer = rnn[l]
      num_inputs = 0
      neurons = []
      for n in range(0, len(layer)):
        neuron_dict = layer[n]
        weights = neuron_dict[Neuron._weights]
        bias = neuron_dict[Neuron._bias]
        neuron = Neuron(len(weights))
        neuron.weights = weights
        neuron.bias = bias
        layer[n] = neuron
        num_inputs = len(weights)
        neurons.append(neuron)
      rnn[l] = Layer(num_inputs, len(layer))
      rnn[l].neurons = neurons

    new_rnn = RNN(0,0,0)
    new_rnn.layers = rnn
    return new_rnn

  
  def load_from_json(self, file):
    serialized_rnn = fileio.read_json(file)
    rnn = RNN.deserialize(serialized_rnn)
    self.layers = rnn
    return self

file = "test.json"

# rnn = RNN(num_inputs=2, num_hidden_layers=1, num_neurons_in_hidden=3)
rnn = RNN.LoadRNN(file)
print(rnn.serialize())
# rnn.write_json(file)
# rnn = RNN(0,0,0)
# rnn.load_from_json(file)
# print(rnn.serialize())
# print(rnn.serialize() == RNN.deserialize(rnn.serialize()).serialize())

# rnn.de(file)
# print(rnn.serialize())
# print(rnn.deserialize(fileio.read_json(file)).serialize())
# print(rnn.serialize() == rnn.load_from_json(file).serialize())
# print(rnn.to_json())
# fileio.tojson(rnn.to_json(), file)
# print(fileio.read_json(file))