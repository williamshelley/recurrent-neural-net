from random import random

class Weight:
  def __init__(self, value=None, input_neuron=None, output_neuron=None) -> None:
    self.value = value if value is not None else random()
    self.input_neuron = input_neuron
    self.output_neuron = output_neuron
  
  def __repr__(self) -> str:
    return str(self.value)