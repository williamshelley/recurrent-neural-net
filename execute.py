from rnn import RNN
import fileio
import json

# layer_2 = Layer(n_neurons=3, id=2)
# layer_1 = Layer(n_neurons=5, id=1, next_layer=layer_2)

json_file = "data.json"
# dl1 = Layer.deserialize(fileio.parse_json(json_file)) # deserialized layer 1
# print([neuron.weights for neuron in dl1.neurons])

basic_rnn = RNN.load_from_file(json_file)


# basic_rnn = RNN(n_inputs=2, n_hidden=5)
print(basic_rnn.forward_propogate([0,0]))


basic_rnn.tofile(json_file)