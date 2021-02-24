class Neuron:
  placeholder_weight = 0.0

  def __init__(self, layer, activation_fn):
    super().__init__()
    self.layer = layer
    self.activation_fn = activation_fn
    self.weights = None

  def activate(self):
    if self.weights is None:
      self.weights = [Neuron.placeholder_weight for x in range(len(self.layer.neurons))]
    self.activation_fn(self)
