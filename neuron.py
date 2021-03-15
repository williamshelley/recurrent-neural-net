from weight import Weight
from random import random

class Neuron:
  def __init__(self) -> None:
    self.output = None
    self.actsum = None
    self.weights = []
    self.bias = random()
    self.gradient = None
    self.layer = None

  def connect(self, next_neuron) -> None:
    new_weight = Weight(input_neuron=self, output_neuron=next_neuron)
    self.weights.append(new_weight)
    return new_weight

  def is_output(self) -> bool:
    return not self.weights

  def __repr__(self) -> str:
    return str(self)
  
  def __str__(self) -> str:
    return "Out: " + str(self.output) + ", S: " + str(self.actsum) + ", W: " + str(self.weights)