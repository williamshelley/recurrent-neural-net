from random import random
import json
import fileio
from activation_functions import sigmoid_logistic, sigmoid_logistic_derivative
from cost_functions import cross_entropy

class Neuron:
  _weights = "weights"
  _output = "output"
  def __init__(self, input_size):
    self.weights = [random() for x in range(input_size)]
    self.output = None
    self.error = 0.0

  def __getitem__(self, index):
    return self.weights[index]

  def __setitem__(self, index, value):
    self.weights[index] = value
    return self.weights[index]

  def __len__(self):
    return len(self.weights)
  
  def activate(self, inputs, activation_fn):
    activation_sum = 0
    if len(inputs) != len(self.weights):
      i_len = str(len(inputs))
      w_len = str(len(self.weights))
      msg = "input length != neuron weights length (" + i_len + " != " + w_len + ")"
      raise Exception(msg)

    for i in range(len(self.weights)):
      activation_sum += self.weights[i] * inputs[i]

    output = activation_fn(activation_sum)
    self.output = output
    return output

  def to_dict(self):
    d = {}
    d[Neuron._weights] = self.weights
    return d


class Layer:
  def __init__(self, size, input_size, bias = 1.0):
    self.neurons = [Neuron(input_size) for _ in range(size)]
    self.errors = [None for _ in range(size)]
    self.size = size
    self.bias = bias

  def __getitem__(self, index):
    return self.neurons[index]

  def __len__(self):
    return len(self.neurons)

  def print(self):
    for neuron in self.neurons:
      print(neuron.weights, neuron.output)

  def __iter__(self):
    return self.neurons.__iter__()

  def activate(self, inputs, activation_fn):
    output = []
    for neuron in self.neurons:
      output.append(neuron.activate(inputs, activation_fn))
    return output

class RNN:
  def __init__(self, input_size, hidden_layer_size, output_size, n_hidden_layers, alpha):
    input_layer = Layer(input_size, input_size)
    self.alpha = alpha
    self.layers = [input_layer]
    prev_layer = self.layers[0]
    for i in range(n_hidden_layers):
      self.layers.append(Layer(hidden_layer_size, prev_layer.size))
      prev_layer = self.layers[-1]
      
    output_layer = Layer(output_size, prev_layer.size)
    self.layers.append(output_layer)

  def __len__(self):
    return len(self.layers)

  def print(self):
    for layer in self.layers:
      print(len(layer))

  def forward_propagate(self, inputs, activation_fn):
    if len(self.layers[0]) != len(inputs):
      raise Exception("Input size does not match input layer")

    new_inputs = [x for x in inputs]
    for layer in self.layers:
      new_inputs = layer.activate(new_inputs, activation_fn)

    return new_inputs

  # L = Lth layer
  # nL = neuron from layer L
  # nl = neuron from layer L - 1
  # wL = weights of neuron nL
  # nL.output = output of neuron nL
  # nl.output = output of neuron nl
  # sigp(_) = derivative of sigmoid function
  # ce = cross entropy cost function
  # cep = derivative of cross entropy function

  # nL.output * sigp(wL * nl.output + L.bias) * cep(nL.output, expected_output)

  def backward_propagate(self, expected_output, derivative_fn, cost_fn):
    current_expected_output = [x for x in expected_output]

    output_layer = self.layers[-1]
    errors = []

    # output layer errors
    for n in range(len(output_layer)):
      errors.append(expected_output[n] - output_layer[n].output)
    prev_error = sum(errors)

    for l in reversed(range(len(self.layers) - 1)):
      layer = self.layers[l]
      errors = []
      for n in range(len(layer)):
        neuron = layer[n]
        error = 0.0
        for w in range(len(neuron)):
          weight = neuron[w]
          error += weight * prev_error
        error /= len(neuron)
        for w in range(len(neuron)):
          neuron[w] += self.alpha * prev_error

        total_weight = 0.0

        for w in range(len(neuron)):
          total_weight += abs(neuron[w])

        for w in range(len(neuron)):
          neuron[w] /= total_weight

        errors.append(error)
      prev_error = sum(errors) / len(errors)

      # print(prev_error, errors)
      # print(errors, new_errors)

      # errors = []
      # errors = new_errors

    # for l in reversed(range(len(self.layers))):
    #   layer = self.layers[l]
    #   errors = [0.0 for _ in range(len(layer))]
    #   for n in range(len(layer)):
    #     neuron = layer[n]
    #     errors[n] = cost_fn(neuron.output, current_expected_output[n])
    #   error = sum(errors) / len(errors)
      
    #   layer = self.layers[l - 1]
    #   for n in range(len(layer)):
    #     neuron = layer[n]
    #     for w in range(len(neuron)):
    #       neuron[w] += self.alpha * errors[n]
          
    #     total_weight = 0.0
    #     for weight in neuron:
    #       total_weight += abs(weight)

    #     for w in range(len(neuron)):
    #       neuron[w] /= total_weight

    #   if l > 0:
    #     # current_expected_output = errors
    #     for n in range(len(self.layers[l-1])):
    #       neuron = self.layers[l-1][n]
    #       current_expected_output.append(error * derivative_fn(neuron.output))
            
    return


  @classmethod
  def LoadRNN(cls, file, alpha):
    serialized_rnn = fileio.read_json(file)
    rnn = RNN.deserialize(serialized_rnn, alpha)
    return rnn

  def __getitem__(self, index):
    return self.layers[index]

  def write_json(self, file):
    fileio.write_json(self.serialize(), file)
    return

  def serialize(self):
    rnn = []
    for layer in self.layers:
      rnn.append([n.to_dict() for n in layer.neurons])
    return json.dumps(rnn)

  @classmethod
  def deserialize(cls, serialized_rnn, alpha):
    rnn = json.loads(serialized_rnn) if isinstance(serialized_rnn, str) else serialized_rnn

    for l in range(0, len(rnn)):
      layer = rnn[l]
      num_inputs = 0
      neurons = []
      for n in range(0, len(layer)):
        neuron_dict = layer[n]
        weights = neuron_dict[Neuron._weights]
        neuron = Neuron(len(weights))
        neuron.weights = weights
        layer[n] = neuron
        num_inputs = len(weights)
        neurons.append(neuron)
      rnn[l] = Layer(num_inputs, len(layer))
      rnn[l].neurons = neurons

    new_rnn = RNN(0,0,0,0,alpha)
    new_rnn.layers = rnn
    return new_rnn

  def load_from_json(self, file):
    serialized_rnn = fileio.read_json(file)
    rnn = RNN.deserialize(serialized_rnn)
    self.layers = rnn
    return self

file = "test.json"

# rnn = RNN(input_size=2, hidden_layer_size=5, output_size=1, n_hidden_layers=3, alpha=0.1)
rnn = RNN.LoadRNN(file, 0.1)

# rnn.forward_propagate([0,0], sigmoid_logistic)
# rnn.backward_propagate([0], sigmoid_logistic_derivative, cross_entropy)

# rnn.print()

for i in range(10000):
  rnn.forward_propagate([0,0], sigmoid_logistic)
  rnn.backward_propagate([0], sigmoid_logistic_derivative, cross_entropy)

  rnn.forward_propagate([0,1], sigmoid_logistic)
  rnn.backward_propagate([0], sigmoid_logistic_derivative, cross_entropy)

  rnn.forward_propagate([1,0], sigmoid_logistic)
  rnn.backward_propagate([0], sigmoid_logistic_derivative, cross_entropy)

  # rnn.forward_propagate([1,1], sigmoid_logistic)
  # rnn.backward_propagate([1], sigmoid_logistic_derivative, cross_entropy)

print([round(x, 5) for x in rnn.forward_propagate([0,0], sigmoid_logistic)]) # 0
print([round(x, 5) for x in rnn.forward_propagate([0,1], sigmoid_logistic)]) # 0
print([round(x, 5) for x in rnn.forward_propagate([1,0], sigmoid_logistic)]) # 0
# print([round(x, 10) for x in rnn.forward_propagate([1,1], sigmoid_logistic)]) # 1
rnn.write_json(file)