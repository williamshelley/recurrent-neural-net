from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from network import Network
from data import print_summary

ZERO,ONE,T1,T2,T3,T4 = [0],[1],[0,0],[0,1],[1,0],[1,1]
AND_DATA = ([[T1, ZERO], [T2, ZERO], [T3, ZERO], [T4, ONE]], "AND")
OR_DATA = ([[T1, ZERO], [T2, ONE], [T3, ONE], [T4, ONE]], "OR")
XOR_DATA = ([[T1, ZERO], [T2, ONE], [T3, ONE], [T4, ZERO]], "XOR")

EPOCHS = 100000
LEARN_RATE = 0.1
ERR_PRECISION = 0.001

structure = [2,5,1]
network = Network(structure, lossfn=mse, actfn=sigmoid, actfnp=sigmoidp)

dataset, prefix = AND_DATA
# dataset, prefix = OR_DATA
# dataset, prefix = XOR_DATA

network.load_from_file(prefix=prefix)
network_file = network.build_file_name(prefix)

network.train(dataset, EPOCHS, LEARN_RATE, ERR_PRECISION)

results = network.run(dataset)
print("results:",results)

network.write_to_file(prefix=prefix)

total_network_err = network.get_total_error(dataset)

print_summary(dataset, results, total_network_err, network_file)
