from weight import Weight
from random import random
from json import dumps, loads

class Neuron:
  WEIGHTS = "weights"
  BIAS = "bias"
  
  def __init__(self) -> None:
    self.output = None
    self.actsum = None
    self.weights = []
    self.bias = random()
    self.gradient = None
    self.layer = None

  def serialize(self) -> str:
    serialized_weights = dumps([w.value for w in self.weights])
    return { Neuron.WEIGHTS: serialized_weights, Neuron.BIAS: self.bias }

  def deserialize(self, serialized_neuron):
    weights = loads(serialized_neuron[Neuron.WEIGHTS])
    if len(weights) != len(self.weights):
      raise Exception("serialized weights != neuron weights")

    for w in range(len(weights)):
      self.weights[w].value = weights[w]

    self.bias = serialized_neuron[Neuron.BIAS]
    return self

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