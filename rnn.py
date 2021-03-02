from layer import Layer
import json
import fileio

class RNN:
  hidden_layer_neuron_count = 3
  n_hidden_str = "n_hidden"
  n_inputs_str = "n_inputs"

  def __init__(self, n_inputs = 2, n_hidden = 1):
    super().__init__()
    self.input_layer = Layer(n_inputs=n_inputs, n_neurons=RNN.hidden_layer_neuron_count, id=Layer.input_layer_id)
    self.input_layer.next_layer = Layer(n_inputs=n_inputs, n_neurons=RNN.hidden_layer_neuron_count, id=1)
    current_layer = self.input_layer.next_layer
    for i in range(1, n_hidden):
      hidden_layer = Layer(n_inputs=n_inputs, n_neurons=RNN.hidden_layer_neuron_count, id=current_layer.id+1)
      current_layer.next_layer = hidden_layer
      current_layer.initialize_neuron_weights()
      current_layer = current_layer.next_layer

    self.output_layer = Layer(n_inputs=n_inputs, n_neurons=RNN.hidden_layer_neuron_count)
    self.output_layer.id = Layer.output_layer_id
    current_layer.next_layer = self.output_layer
    current_layer.initialize_neuron_weights()

    self.n_hidden = n_hidden
    self.n_inputs = n_inputs

  def forward_propogate(self, inputs):
    if len(inputs) != self.n_inputs:
      raise Exception("Input array length does not equal RNN input length")
    
    current_layer = self.input_layer
    outputs = inputs
    while current_layer:
      outputs = current_layer.process_input(outputs)
      current_layer = current_layer.next_layer

    return outputs

  def transfer_derivative(self, output):
    return output * (1.0 - output)

  # used for neurons in output layer
  def backpropagate_error(self, output, expected_output):
    return (expected_output - output) * self.transfer_derivative(output)

  def serialize(self):
    data = {}
    current_layer = self.input_layer
    visited = set()

    data[RNN.n_hidden_str] = self.n_hidden
    data[RNN.n_inputs_str] = self.n_inputs

    while current_layer and current_layer.id not in visited:
      visited.add(current_layer.id)
      data[current_layer.id] = current_layer.serialize()
      current_layer = current_layer.next_layer

    return json.dumps(data)

  def tofile(self, file):
    fileio.tojson(self.serialize(), file)
    return

  @classmethod
  def load_from_file(cls, file):
    data = json.loads(fileio.parse_json(file))

    if Layer.input_layer_id not in data or Layer.output_layer_id not in data:
      raise Exception("No input or output layer in json file")

    rnn = RNN(n_inputs=data[RNN.n_inputs_str], n_hidden=data[RNN.n_hidden_str])
    visited = set()

    json_input_layer = data[Layer.input_layer_id]
    rnn.input_layer = Layer.deserialize(Layer.input_layer_id, json_input_layer)

    current_layer = rnn.input_layer
    next_layer_id = data[Layer.input_layer_id][Layer.next_layer_str]

    while next_layer_id and next_layer_id not in visited:
      if str(next_layer_id) in data:
        json_next_layer = data[str(next_layer_id)]
        current_layer.next_layer = Layer.deserialize(next_layer_id, json_next_layer)
        
        current_layer = current_layer.next_layer
        next_layer_id = current_layer.next_layer.id if current_layer.next_layer else None
      else:
        raise Exception("Layer " + str(next_layer_id) + " not in data ")

    if current_layer.id == Layer.output_layer_id:
      rnn.output_layer = current_layer
    else:
      raise Exception("Last layer is not output layer on load from file")

    return rnn