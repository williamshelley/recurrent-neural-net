from neuron import Neuron
from json import dumps, loads

class Network:
  def __init__(self, structure) -> None:
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

  def __repr__(self) -> str:
    return str(self) + "\n"

  def __str__(self) -> str:
    result = ""
    for layer in self.network:
      result += "\n" + str(layer) + "\n"
    return result

  def serialize(self) -> str:
    return dumps([[x.serialize() for x in layer] for layer in self.network])

  def deserialize(self, serialized_network):
    if serialized_network is None:
      raise Exception("No serialized data avaialble")
    
    deserialized = serialized_network
    if type(serialized_network) is str:
      deserialized = loads(serialized_network)
      
    for l in range(len(deserialized)):
      layer = deserialized[l]
      for n in range(len(layer)):
        neuron = layer[n]
        self.network[l][n].deserialize(neuron)

    return self

  def forward_propagate(self, inputs, actfn):
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
        neuron.output = actfn(neuron.actsum)

    return [x.output for x in self.network[-1]]

  # network, expected output, loss function, derivative of activation function
  def backpropagate(self, expected, learn_rate, lossfn, actfnp):
    output_layer = self.network[-1]
    L = len(self.network) - 1

    for i in range(len(output_layer)):
      neuron = output_layer[i]
      # gradient = (expected[i] - neuron.output) * actfnp(neuron.actsum)
      gradient = lossfn(a=neuron.output, e=expected[i]) * actfnp(neuron.actsum)
      neuron.gradient = gradient
      self.backpropagate_hidden(neuron, learn_rate, actfnp)

  # L is the same layer that source neuron is in
  def backpropagate_hidden(self, source, learn_rate, actfnp):
    if source is None or source.layer < 1:
      return
    
    layer = self.network[source.layer - 1]
    for neuron in layer:
      new_gradient = 0.0
      for weight in neuron.weights:
        if weight.output_neuron == source:
          gradient_wrt_w = weight.value * source.gradient * actfnp(neuron.actsum)
          new_gradient += gradient_wrt_w
          weight_delta = learn_rate * source.gradient * neuron.output
          weight.value += weight_delta
          bias_delta = -learn_rate * source.gradient
          neuron.bias += bias_delta
      neuron.gradient = new_gradient
      self.backpropagate_hidden(neuron, learn_rate, actfnp)