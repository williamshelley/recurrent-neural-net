from os import error
from random import random
import math

# neuron is a list of weights
# hidden layer is a list of neurons with an additional bias neuron
# input layer is a list of neurons
# output layer is a list of neurons

def sigmoid(activation_sum):
  return 1.0 / (1.0 + math.exp(-activation_sum))

def sigmoidp(output):
  return output * (1.0 - output)

def mse(actual, expected):
  e = expected - actual
  return e * e

# number of inputs, outputs, hidden layers, and neurons per hidden layer
def init_network(insize, outsize, n_hidden, n_neurons):
  layers = []

  input_layer = [[random() for _ in range(insize)] for _ in range(insize)]
  output_layer = [[random() for _ in range(outsize)] for _ in range(outsize)]

  layers.append(input_layer)
  for _ in range(n_hidden):
    new_layer = [[random() for _ in range(insize)] for _ in range(n_neurons)]
    layers.append(new_layer)
  layers.append(output_layer)

  biases = [1.0 for _ in range(n_hidden)]

  return (layers, biases)

def transpose(matrix):
  transposed = [[None for _ in matrix] for _ in matrix[0]]
  for r in range(len(matrix)):
    for c in range(len(matrix[r])):
      transposed[c][r] = matrix[r][c]
  return transposed

  # print("\noriginal:")
  # for row in weights_matrix:
  #   print(row)

  # print("\ntransposed:")
  # for row in transposed:
  #   print(row)

# neural net
def print_network(network):
  for l in range(len(network)):
    layer = network[l]
    print("\nLayer", l + 1)
    for n in range(len(layer)):
      neuron = layer[n]
      print("Neuron", n+1, ":", neuron)
  return


def mv_multiply(matrix, vector):
  result = [0.0 for _ in range(len(matrix))]
  for r in range(len(matrix)):
    row = matrix[r]
    if len(row) != len(vector):
      raise Exception("Matrix and vector have incompatible dimensions (matrix row len=" + str(len(row)) + ", vector len=" + str(len(vector)))

    for w in range(len(row)):
      result[r] += row[w] * vector[w]
  return result

# neural net, list of inputs, and activation function
def forward_propagate(network, inputs, actfn):
  layers,biases = network
  al_lst = [] # activations (all a^l), same structure as network
  zl_lst = [] # intermediate value to activation (all z^l), same structure as network
  last_al_lst = [x for x in inputs] # inputs for first layer, will become outputs of l-1

  for l in range(len(layers)):
    layer = layers[l]
      

    zl = [] # layer's intermediate activations
    al = [] # layer's activations
    for n in range(len(layer)):
      neuron = layer[n]

      zl_n = 0.0

      # if current layer is output layer, don't apply its weights (just sum previous layer's outputs --> output layer's inputs)
      if l == len(layers) - 1:
        for i in range(len(inputs)):
          zl_n += inputs[i]
        zl_lst.append([zl_n])
        al_n = actfn(zl_n)
        al_lst.append([al_n])
        return (al_lst, zl_lst)

      for last_output in last_al_lst:
        for w in range(len(neuron)):
          zl_n += last_output * neuron[w]

      if l > 0 and l < len(layers) - 1:
        zl_n += biases[l - 1] * 1.0

      zl.append(zl_n) # add intermediate activation to layer's zl
      al_n = actfn(zl_n) # calculate activation for neuron n
      al.append(al_n) # add activation to layer's al

    zl_lst.append(zl)
    al_lst.append(al)
    last_al_lst = al # use previous layer's activations as inputs to next layer
  
  return (al_lst,zl_lst)


# first calculate the error gradients for the output layer
# iterate over gradients
# backprop for each gradient through entire network
# only update weights w.r.t. 
def backpropagate(network, network_output, expected_output, lossfn, derivative_actfn):
  layers,biases = network
  al_lst,zl_lst = network_output
  L = len(layers) - 1
  gl_lst = [None for _ in layers] # list of list of errors for each neuron (network -> layer -> neuron errors)
  for l in reversed(range(len(layers))):
    layer = layers[l]

    gradients = [] # errors for each layer's neurons

    if l == L: # output layer
      if len(layer) != len(expected_output):
        raise Exception("Network output layer and expected output length are unequal")

      for n in range(len(layer)):
        
        loss_wrt_output = lossfn(actual=al_lst[l][n], expected=expected_output[n])
        error_gradient_wrt_output = loss_wrt_output * derivative_actfn(al_lst[l][n])
        gradients.append(error_gradient_wrt_output)
        
        # calculate gradient
      gl_lst[l] = gradients
      
    else: # hidden layers
      # last_layer = layers[l + 1]
      current_layer_outputs = al_lst[l]
      last_layer_gradients = gl_lst[l + 1]
      current_layer = layers[l]
      # transposed_weights = transpose(last_layer)
      # print(len(last_layer), transposed_weights)
      alpha = 0.1

      gradients = []
      for n in range(len(current_layer_outputs)):
        neuron_output = current_layer_outputs[n]
        neuron = current_layer[n]
        print(len(neuron), len(last_layer_gradients))
        for w in range(len(neuron)):
          delta = last_layer_gradients[w] * neuron_output * alpha
          neuron[w] += delta
          err_gradient = neuron[w] * last_layer_gradients[w] * derivative_actfn(neuron_output)
          gradients.append(err_gradient)
      gl_lst[l] = gradients

        # output gradient * corresponding weight from this neuron



      # for hn in last_hidden_layer:
      #   alpha = 0.1
      #   delta = error_gradient_wrt_output * hn.output * alpha
      #   for w in hn:


      # layers[l][n][]
      # print(transposed_weights, gl_lst[l+1])
      
      # gl = mv_multiply(transposed_weights, gl_lst[l + 1])
      # gl_lst[l] = gl
      # print(gl_lst)

  print(gl_lst)
  return

# orig = [[1,2,3], [4,5,6]]
# transposed = transpose(orig)
# vector = [2,2,2]
# print(mv_multiply(orig, vector))
  
network = init_network(2,1,1,2)
network_output = forward_propagate(network, [0,0], sigmoid)
backpropagate(network, network_output, [0], mse, sigmoidp)
# print(network[0])

# neural net, actual outputs, expected outputs, derivative of logistic fn, loss fn
# def backward_propagate(network, outputs, expected, logistic_p, lossfn):
#   output_errors = []
#   output_layer = network[-1]
#   for n in range(len(output_layer)):
#     # loss_wrt_output = lossfn(actual=outputs[-1][n], expected=expected[n])
#     # error = loss_wrt_output * logistic_p(outputs[-1][n])
#     error = (outputs[-1][n] - expected[n])

#     output_errors.append(error)
#   errors = [x for x in output_errors]
#   network_errors = [output_errors]
  
#   # should each weight have an error associated with it?
#   # or should only each neuron have an error associated with it?
#   for l in reversed(range(len(network) - 1)):
#     new_errors = []
#     layer = network[l]
#     for n in range(len(layer)):
#       neuron = layer[n]
#       error = 0.0
#       for e in errors:
#         for w in range(len(neuron)):
#           weight = neuron[w]
#           # what needs to be multiplied?
#           error = weight * e * logistic_p(outputs[l][n])
#       new_errors.append(error)
#     network_errors.append(new_errors)
#     errors = new_errors
#   network_errors.reverse()
#   print(network_errors)

#   # g^l_j = error in the jth neuron in lth layer
#   #       = dC / dz^l_j
#   # Four fundamental equations:
#   #   z^L_j is the weighted input vector (weights of neuron j times the input)
#   #   Equation for error in output layer: g^L_j = (dC / da^L_j) * logistic_p(z^L_j)
#   #     The error for neuron j at layer L (output layer) is equal to 
#   #     the (partial derivative of the Cost function w.r.t. the activation of neuron j at
#   #     the output layer L) multiplied by the logistic function's derivative on the 
#   #     weighted input of neuron j at output layer L
#   #   First term measures the rate of change in the cost
#   #   Second term measures the rate of change of the activation function at z^L_j

#   return network_errors

# def update_weights(network, biases, errors, outputs, alpha, lossp):
#   for l in range(len(network)):
#     layer = network[l]
#     layer_errors = errors[l]
#     for n in range(len(layer)):
#       neuron = layer[n]
#       if l > 0 and l < len(network) - 1:
#         biases[l - 1] -= alpha * layer_errors[n]

#       for w in range(len(neuron)):
#         network[l][n][w] -= alpha * layer_errors[n] * outputs[l][n]

#   return

# # y-hat is actual output, y is expected output
# # def cross_entropy(actual, expected):
# #   return -(expected * math.log(actual) + (1.0 - expected) * math.log(1.0 - actual))

# # def cross_entropy_p(actual, expected):
# #   return -(expected / actual) + ((1 - expected) / (1 - actual))

# def train_network(network, biases, training_time, dataset, training_rate):
#   for _ in range(training_time):
#     for data in dataset:
#       inputs, expected = data
#       outputs = forward_propagate(network, biases, inputs, sigmoid)
#       errors = backward_propagate(network, outputs, expected, sigmoidp, mse)
#       update_weights(network, biases, errors, outputs, training_rate, sigmoidp)

# # indices: 0->inputs, 1->expected output
# t1 = ([0,0], [0])
# t2 = ([0,1], [0])
# t3 = ([1,0], [0])
# t4 = ([1,1], [1])
# dataset = [t1, t2, t3, t4]
# reps = 10000
# training_rate = 0.01

# # network,biases = init_network(insize=2,outsize=1,n_hidden=2,n_neurons=3)
# # print_network(network=network)
# # print("-----------------------")

# # outputs = forward_propagate(network=network,biases=biases,inputs=t1[0],actfn=sigmoid)
# # network_output = outputs[-1]
# # print(network_output)

# # train_network(network, biases, reps, dataset, training_rate)

# # outputs = forward_propagate(network=network,biases=biases,inputs=t1[0],actfn=sigmoid)
# # network_output = outputs[-1]
# # print(network_output)

# # outputs = forward_propagate(network=network,biases=biases,inputs=t2[0],actfn=sigmoid)
# # network_output = outputs[-1]
# # print(network_output)

# # outputs = forward_propagate(network=network,biases=biases,inputs=t3[0],actfn=sigmoid)
# # network_output = outputs[-1]
# # print(network_output)

# # outputs = forward_propagate(network=network,biases=biases,inputs=t4[0],actfn=sigmoid)
# # network_output = outputs[-1]
# # print(network_output)

# # print_network(network=network)
# # print(biases)