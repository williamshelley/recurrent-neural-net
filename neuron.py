import json
import sys
from random import random

class Neuron(object):
  def __init__(self, layer, activation_fn, weights=None):
    super().__init__()
    self.layer = layer
    self.activation_fn = activation_fn
    self.weights = weights
    
  def initialize_weights(self):
    if self.weights is None and self.layer.next_layer:
      next_layer_neurons = self.layer.next_layer.neurons
      n_neurons = len(next_layer_neurons)
      self.weights = [random() for x in range(0, n_neurons)]

  def set_weights(self, new_weights):
    self.weights = new_weights
    return

  def activate(self, inputs, bias=0):
    activation_sum = bias

    if self.weights is None:
      return None

    if len(inputs) != len(self.weights):
      raise Exception("Input array length does not match neuron weights")
    else:
      for i in range(0, len(inputs)):
        activation_sum += inputs[i] * self.weights[i]

    return self.activation_fn(activation_sum)

  def serialize(self):
    return self.weights

  @staticmethod
  def deserialize(json_neuron):
    return