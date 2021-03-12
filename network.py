from neuron import Neuron

class Network:
  def __init__(self, structure) -> None:
    network = []
    for l in range(len(structure)):
      this_layer_num_neurons = structure[l]
      layer = [Neuron() for _ in range(this_layer_num_neurons)]
      network.append(layer)

    for l in range(len(network) - 1):
      this_layer = network[l]
      next_layer = network[l + 1]
      for this_neuron in this_layer:
        for next_neuron in next_layer:
          this_neuron.connect(next_neuron)

    self.network = network

  def __repr__(self) -> str:
    result = ""
    for layer in self.network:
      result += "\n" + str(layer) + "\n"
    result += "\n"
    return result

  def forward_propagate(self, inputs, actfn):
    if len(inputs) != len(self.network[0]):
      raise Exception("Input length does not match input layer")
  
    # set all neuron outputs to 0.0
    for layer in self.network:
      for neuron in layer:
        neuron.output = 0.0

    # set input layer neuron outputs to the inputs
    for i in range(len(inputs)):
      self.network[0][i].output = inputs[i]

    # loop through all but the output layer propagating input
    for l in range(len(self.network) - 1):
      layer = self.network[l]
      next_layer = self.network[l + 1]
  
      # if it's not the last hidden layer multiply weight by output
      if l < len(self.network) - 2:
        for neuron in layer:
          for weight in neuron.weights:
            weight.output_neuron.output += weight.value * neuron.output
      else:
        for neuron in layer:
          for weight in neuron.weights:
            weight.output_neuron.output += neuron.output

      # apply activation function to all neuron outputs
      for neuron in next_layer:
        neuron.output = actfn(neuron.output + neuron.bias)

    return [x.output for x in self.network[-1]]

  # network, expected output, loss function, derivative of activation function
  def backpropagate(self, expected, learning_rate, lossfn, actfnp):
    output_layer = self.network[-1]
    L = len(self.network) - 1
    for out_n in range(len(output_layer)):
      output_neuron = output_layer[out_n]
      error = lossfn(actual=output_neuron.output, expected=expected[out_n])
      output_neuron.gradient = error * actfnp(output_neuron.output)
      self.backpropagate_hidden(output_neuron, L, learning_rate, actfnp)
    return

  def backpropagate_hidden(self, source_neuron, l, learning_rate, actfnp):
    if source_neuron is None or l < 0:
      return
    
    bias_delta = -learning_rate * source_neuron.gradient
    source_neuron.bias += bias_delta

    for neuron in self.network[l - 1]:
      error_gradient = source_neuron.gradient
      for weight in neuron.weights:
        if weight.output_neuron == source_neuron:
          delta = learning_rate * error_gradient * neuron.output
          
          weight.value += delta

          new_gradient = weight.value * error_gradient * actfnp(neuron.output)

          neuron.gradient = new_gradient
          self.backpropagate_hidden(neuron, l-1, learning_rate, actfnp)
    return