from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from data import INPUT_INDEX, OUTPUT_INDEX
from fileio import read_json, write_json
from neuron import Neuron
from tqdm import tqdm

class Network:
  def __init__(self, structure, lossfn, actfn, actfnp) -> None:
    network = []
    L = len(structure) - 1
    for l in range(len(structure)):
      this_layer_num_neurons = structure[l]
      layer = [Neuron() for _ in range(this_layer_num_neurons)]
      for neuron in layer: neuron.layer = l

      network.append(layer)

    for l in range(len(network) - 1):
      this_layer = network[l]
      next_layer = network[l + 1]
      for this_neuron in this_layer:
        for next_neuron in next_layer:
          this_neuron.connect(next_neuron)

    self.network = network
    self.lossfn = lossfn
    self.actfn = actfn
    self.actfnp = actfnp

  def __repr__(self) -> str:
    return str(self) + "\n"

  def __str__(self) -> str:
    result = ""
    for layer in self.network:
      result += "\n" + str(layer) + "\n"
    return result

  def get_total_error(self, dataset):
    error = 0.0
    for data in dataset:
      input = data[INPUT_INDEX]
      expected = data[OUTPUT_INDEX]
      actual = self.forward_propagate(input)
      for i in range(len(actual)):
        error += abs(self.lossfn(a=actual[i], e=expected[i]))
    error /= (1.0 * len(dataset))
    return error

  def serialize(self) -> str:
    return [[x.serialize() for x in layer] for layer in self.network]

  def deserialize(self, serialized_network):
    if serialized_network is None:
      raise Exception("No serialized data avaialble")
    
    deserialized = serialized_network
      
    for l in range(len(deserialized)):
      layer = deserialized[l]
      for n in range(len(layer)):
        neuron = layer[n]
        self.network[l][n].deserialize(neuron)

    return self

  def train(self, dataset, max_epochs, learn_rate, precision):
    for epoch in tqdm(range(max_epochs)):
      for i in range(len(dataset)):
        inputs = dataset[i][INPUT_INDEX]
        expected = dataset[i][OUTPUT_INDEX]
        self.forward_propagate(inputs)
        self.backpropagate(expected, learn_rate)

      total_error = self.get_total_error(dataset)

      if (epoch % (max_epochs // 10)) == 0:
        learn_rate /= 1.05

      if abs(total_error) <= abs(precision):
        return

    return

  def run(self, dataset):
    results = []
    for i in range(len(dataset)):
      if type(dataset[i]) == list:
        inputs = dataset[i][0]
      elif type(dataset[i]) == dict:
        inputs = dataset[i]["input"]
      results.append(self.forward_propagate(inputs))
    return results

  def build_file_name(self, prefix, filetype=".json"):
    structure = "x".join([str(len(l)) for l in self.network])
    return prefix + "-" + structure + filetype

  def write_to_file(self, prefix):
    return write_json(self.serialize(), file=self.build_file_name(prefix=prefix))

  def load_from_file(self, prefix):
    network_file = self.build_file_name(prefix)
    serialized_network = read_json(network_file)
    if serialized_network is not None:
      self.deserialize(serialized_network)
    else:
      print(network_file + " not found, creating file...")
    return

  def forward_propagate(self, inputs):
    if len(inputs) != len(self.network[0]):
      raise Exception("Input length does not match input layer")
  
    # set all neuron outputs to 0.0
    for layer in self.network:
      for neuron in layer:
        neuron.actsum = 0.0
        neuron.output = 0.0

    # set input layer neuron outputs to the inputs
    for i in range(len(inputs)):
      self.network[0][i].output = inputs[i]

    # loop through all but the output layer propagating input
    for l in range(len(self.network) - 1):
      layer = self.network[l]
      next_layer = self.network[l + 1]
  
      # if it's not the last hidden layer multiply weight by output
      for neuron in layer:
        for weight in neuron.weights:
          weight.output_neuron.actsum += weight.value * neuron.output

      # apply activation function to all neuron outputs
      for neuron in next_layer:
        neuron.actsum += neuron.bias
        neuron.output = self.actfn(neuron.actsum)

    return [x.output for x in self.network[-1]]

  # network, expected output, loss function, derivative of activation function
  def backpropagate(self, expected, learn_rate):
    output_layer = self.network[-1]

    for i in range(len(output_layer)):
      neuron = output_layer[i]
      gradient = self.lossfn(a=neuron.output, e=expected[i]) * self.actfnp(neuron.actsum)
      neuron.gradient = gradient
      self.backpropagate_hidden(neuron, learn_rate)

  # L is the same layer that source neuron is in
  def backpropagate_hidden(self, source, learn_rate):
    if source is None or source.layer < 1:
      return
    
    layer = self.network[source.layer - 1]
    for neuron in layer:
      new_gradient = 0.0
      for weight in neuron.weights:
        if weight.output_neuron == source:
          gradient_wrt_w = weight.value * source.gradient * self.actfnp(neuron.actsum)
          new_gradient += gradient_wrt_w
          weight_delta = learn_rate * source.gradient * neuron.output
          weight.value += weight_delta
          bias_delta = -learn_rate * source.gradient
          neuron.bias += bias_delta
      neuron.gradient = new_gradient
      self.backpropagate_hidden(neuron, learn_rate)
