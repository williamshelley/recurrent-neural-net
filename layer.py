from activation_functions import sigmoid_logistic
from neuron import Neuron
import json

class Layer:
  neurons_str = "neurons"
  next_layer_str = "next_layer"
  input_layer_id = "input_layer"
  output_layer_id = "output_layer"

  def __init__(self, n_inputs=2, n_neurons=1, activation_fn=sigmoid_logistic, id=1, next_layer=None):
    super().__init__()
    self.id = id
    self.activation_fn = activation_fn
    self.neurons = [Neuron(self, self.activation_fn) for x in range(0, n_inputs)]
    self.next_layer = next_layer
    self.initialize_neuron_weights()

  def process_input(self, inputs):
    outputs = [0 for x in range(0, len(inputs))]
    for i in range(0, len(inputs)):
      result = self.neurons[i].activate(inputs=inputs, bias=0)
      if result:
        outputs[i] += result
      else:
        outputs[i] = inputs[i]
    return outputs

  def get_neurons_as_weights(self):
    return [neuron.weights for neuron in self.neurons]

  def set_neurons_with_weights(self, weights):
    if (len(self.neurons)) != len(weights):
      self.neurons = [Neuron(self, self.activation_fn, weights[i]) for i in range(0, len(weights))]
    else:
      for i in range(len(weights)):
        self.neurons[i].set_weights(weights[i])
    return

  def initialize_neuron_weights(self):
    for neuron in self.neurons:
      neuron.initialize_weights()

  # if a layer loops back around to itself, this will exit
  def activate_all(self):
    current = self
    
    while current:
      print("\nlayer " + str(current.id))

      for neuron in current.neurons:
        neuron.activate()

      current = current.next_layer

      if current == self:
        return

    return
      
  def serialize(self):
    serialized_neurons = [x.serialize() for x in self.neurons]
    serialized_layer = {}
    serialized_layer[Layer.neurons_str] = serialized_neurons
    serialized_layer[Layer.next_layer_str] = self.next_layer.id if self.next_layer else None
    return serialized_layer

  @staticmethod
  def deserialize(id, json_layer):
    weights = [x for x in json_layer[Layer.neurons_str]]
    next_layer_id = json_layer[Layer.next_layer_str]

    layer = Layer()
    layer.set_neurons_with_weights(weights)
    layer.id = id
    layer.next_layer = Layer(id=next_layer_id)

    return layer

  def __getitem__(self, key):
    return neurons[key] if 0 <= key < len(neurons) else None