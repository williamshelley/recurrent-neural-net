from network import Network
from data import Data, T1, T2, T3, T4
from activation_functions import sigmoid, sigmoidp
from cost_functions import mse
from tqdm import tqdm
from fileio import write_json, read_json
import requests


EPOCHS = 10000
LEARN_RATE = 0.1
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



# use iris training --> 3 outputs, [1,0,0]=setosa, [0,1,0]=versicolor, [0,0,1]=virginica

# recognizing numbers dataset

# structure = [2,5,1]
structure = [4,10,3]
network = Network(structure)
# network.deserialize(read_json("test.json"))
iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
req = requests.get(iris_url)
data = req.text.strip().split("\n")
data = [x.split(",") for x in data]
dataset = [iris_to_data(x) for x in data]
outfile = "iris-network.json"

# dataset, outfile = Data.AND()
# dataset, outfile = Data.OR()
# dataset, outfile = Data.XOR()

dic = {}
for i in range(len(dataset)):
  expected = str(dataset[i][1])
  if expected in dic:
    dic[expected] += 1
  else:
    dic[expected] = 1
print(dic)


serialized_network = read_json(outfile)
network.deserialize(serialized_network)

network.train(dataset, mse, sigmoid, sigmoidp, EPOCHS, LEARN_RATE, ERR_PRECISION)
write_json(network.serialize(), outfile)

print(network.get_total_error(dataset, mse, sigmoid, sigmoidp))
results = {}
errors = []
for i in range(len(dataset)):
  result = network.forward_propagate(dataset[i][0], sigmoid)
  rounded = [round(x) for x in result]
  result_str = str(rounded)
  if rounded != dataset[i][1]:
    errors.append([i, rounded, dataset[i][1]])
  if result_str in results:
    results[result_str] += 1
  else:
    results[result_str] = 1

print(results)
print(errors)

# print(network.forward_propagate(dataset[0][0], sigmoid)) # [1,0,0]
# print(network.forward_propagate(dataset[len(dataset) // 2][0], sigmoid)) # [0,1,0]
# print(network.forward_propagate(dataset[-1][0], sigmoid)) # [0,0,1]

# print(network.forward_propagate(T1, sigmoid)) # 0,0
# print(network.forward_propagate(T2, sigmoid)) # 0,1
# print(network.forward_propagate(T3, sigmoid)) # 1,0
# print(network.forward_propagate(T4, sigmoid)) # 1,1