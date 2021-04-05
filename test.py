from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from network import Network

w1 = [[0.2, 0.3, 0.1, 0.4], [0.05, 0.15, 0.45, 0.34]]
w2 = [[0.1], [0.01], [0.3], [0.4]]
weights = [w1, w2]

network = Network([2,4,1], lossfn=mse, actfn=sigmoid, actfnp=sigmoidp)
for l in range(len(network.network)):
  layer = network.network[l]
  for n in range(len(layer)):
    neuron = layer[n]
    for w in range(len(neuron.weights)):
      neuron.weights[w].value = weights[l][n][w]
    neuron.bias = 1.0

result = network.forward_propagate([0.5, 0.4])
network.backpropagate([0.3], 0.1)
print(network)
