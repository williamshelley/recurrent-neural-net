from activation_functions import default_fn
from neuron import Neuron
import json

class Layer:
  neurons_str = "neurons"
  next_layer_str = "next_layer"

  def __init__(self, n_neurons=1, activation_fn=default_fn, id=0, next_layer=None):
    super().__init__()
    self.id = id
    self.activation_fn = activation_fn
    self.neurons = [Neuron(self, self.activation_fn) for x in range(0, n_neurons)]
    self.next_layer = next_layer
    self.next_layer_id = next_layer.id if next_layer else None
    self.initialize_neuron_weights()

  def set_next_layer(self, next_layer):
    if next_layer:
      self.next_layer = next_layer
      self.next_layer_id = next_layer.id

  def set_next_layer_id(self, next_layer_id):
    self.next_layer_id = next_layer_id

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
    return json.dumps({ self.id: serialized_layer })

  @staticmethod
  def deserialize(json_layer):
    parsed = json.loads(json_layer)
    keys = list(parsed.keys())
    id = keys[0]
    layer_json = parsed[id]
    weights = [eval(x) for x in layer_json[Layer.neurons_str]]
    next_layer_id = layer_json[Layer.next_layer_str]

    layer = Layer()
    layer.set_neurons_with_weights(weights)
    layer.set_next_layer_id(next_layer_id)

    return layer

  def __getitem__(self, key):
    return neurons[key] if 0 <= key < len(neurons) else None