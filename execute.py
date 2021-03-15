from network import Network
from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from tqdm import tqdm
from fileio import write_json, read_json

structure = [2,4,1]
network = Network(structure)
epochs = 100000
learn_rate = 0.1
network_file = "network.json"


t1 = [0,0]
t2 = [0,1]
t3 = [1,0]
t4 = [1,1]

zero = [0]
one = [1]


logical_and = [[t1, zero], [t2, zero], [t3, zero], [t4, one]]
logical_or = [[t1, zero], [t2, one], [t3, one], [t4, one]]
exclusive_or = [[t1, zero], [t2, one], [t3, one], [t4, zero]]

dataset = logical_and

serialized_network = read_json(network_file)
network.deserialize(serialized_network)

for _ in tqdm(range(epochs)):
  for i in range(len(dataset)):
    inputs = dataset[i][0]
    expected = dataset[i][1]

    output = network.forward_propagate(inputs, sigmoid)
    network.backpropagate(expected, learn_rate, mse, sigmoidp)
    print("output:", output)
    print(network)

write_json(network.serialize(), network_file)

print(network.forward_propagate(t1, sigmoid))
print(network.forward_propagate(t2, sigmoid))
print(network.forward_propagate(t3, sigmoid))
print(network.forward_propagate(t4, sigmoid))