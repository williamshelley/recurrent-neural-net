from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from data import print_summary
from network import Network
import requests

EPOCHS = 10000
LEARN_RATE = 0.05
ERR_PRECISION = 0.001

def iris_to_lst(x):
  x = x[len("Iris-"):]
  if x == "setosa":
    return [1,0,0]
  elif x == "versicolor":
    return [0,1,0]
  elif x == "virginica":
    return [0,0,1]
  else:
    raise Exception("Error converting iris result to boolean list")

def iris_to_data(row):
  inputs = [float(x) for x in row[:-1]]
  outputs = iris_to_lst(row[-1])
  return [inputs, outputs]

def get_iris_data():
  iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
  req = requests.get(iris_url)
  data = req.text.strip().split("\n")
  return [x.split(",") for x in data]

dataset = [iris_to_data(x) for x in get_iris_data()]

structure = [4, 20, 3]
network = Network(structure, lossfn=mse, actfn=sigmoid, actfnp=sigmoidp)

prefix = "iris-network"
network_file = network.build_file_name(prefix)

network.train(dataset, EPOCHS, LEARN_RATE, ERR_PRECISION)

network.load_from_file(prefix=prefix)
results = network.run(dataset)
network_total_error = network.get_total_error(dataset)
print_summary(dataset, results, network_total_error, network_file)

network.write_to_file(prefix)