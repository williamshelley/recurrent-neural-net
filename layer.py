from activation_functions import default_fn
from neuron import Neuron

class Layer:
  def __init__(self, n_neurons, activation_fn = default_fn, next_layer = None):
    super().__init__()
    self.neurons = [Neuron(self, activation_fn) for x in range(0, n_neurons)]
    self.next_layer = next_layer

  def activate_all(self):
    for neuron in self.neurons:
      neuron.activate()