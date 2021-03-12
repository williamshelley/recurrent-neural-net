from network import Network
from activation_functions import sigmoid, sigmoidp
from cost_functions import mse

# structure = [2,2,1]
structure = [2,1,1]
network = Network(structure)
reps = 100000
alpha = 0.1


t1 = ([0,0])
t2 = ([0,1])
t3 = ([1,0])
t4 = ([1,1])

zero = [0]
one = [1]


logical_and = [[t1, zero], [t2, zero], [t3, zero], [t4, one]]
dataset = logical_and
# dataset = [[t1, one], [t2, zero]]
# dataset = [[t1, zero]]
# dataset = [[t1, one]]

# print(mat_vector_multiply([[1,2,3], [4,5,6]], [2,2,2]))
# print(mat_scalar_op([[1,2,3], [4,5,6]], 2, lambda x,y: x * y))

# print(network)
for i in range(reps):
  for i in range(len(dataset)):
    inputs = dataset[i][0]
    expected = dataset[i][1]

    output = network.forward_propagate(inputs, sigmoid)
    network.backpropagate(expected, alpha, mse, sigmoidp)
    # print(network)

# print(network)
print(network.forward_propagate(t1, sigmoid))
print(network.forward_propagate(t2, sigmoid))
print(network.forward_propagate(t3, sigmoid))
print(network.forward_propagate(t4, sigmoid))