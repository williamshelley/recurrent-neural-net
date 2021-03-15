from network import Network
from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from tqdm import tqdm

structure = [2,5,1]
network = Network(structure)
reps = 100000
alpha = 0.1


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
# dataset = logical_or
# dataset = exclusive_or

for _ in tqdm(range(reps)):
  for i in range(len(dataset)):
    inputs = dataset[i][0]
    expected = dataset[i][1]

    output = network.forward_propagate(inputs, sigmoid)
    network.backpropagate(expected, alpha, mse, sigmoidp)
    # print("output:", output)
    # print(network)

# print(network)


print(network.forward_propagate(t1, sigmoid))
print(network.forward_propagate(t2, sigmoid))
print(network.forward_propagate(t3, sigmoid))
print(network.forward_propagate(t4, sigmoid))