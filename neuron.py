from weight import Weight
from random import random

class Neuron:
  def __init__(self) -> None:
    self.output = None
    self.weights = []
    self.bias = 1.0

  def connect(self, next_neuron) -> None:
    new_weight = Weight(input_neuron=self, output_neuron=next_neuron)
    self.weights.append(new_weight)
    return new_weight

  def is_output(self) -> bool:
    return not self.weights

  def __repr__(self) -> str:
    return str(self)
  
  def __str__(self) -> str:
    return "Out: " + str(self.output) + ", W: " + str(self.weights)

  @classmethod
  def BiasNeuron(cls):
    neuron = Neuron()
    neuron.output = random()