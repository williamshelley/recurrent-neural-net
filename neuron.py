import json

class Neuron(object):
  placeholder_weight = 0.0

  def __init__(self, layer, activation_fn, weights=None):
    super().__init__()
    self.layer = layer
    self.activation_fn = activation_fn
    self.weights = weights
    
  def initialize_weights(self):
    if self.weights is None and self.layer.next_layer:
      next_layer_neurons = self.layer.next_layer.neurons
      n_neurons = len(next_layer_neurons)
      self.weights = [Neuron.placeholder_weight for x in range(0, n_neurons)]

  def set_weights(self, new_weights):
    self.weights = new_weights
    return

  def activate(self):
    self.activation_fn(self)
    return

  def serialize(self):
    return str(self.weights)

  @staticmethod
  def deserialize(json_neuron):
    return