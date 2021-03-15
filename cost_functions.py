from math import  log
from matrix_ops import vec_vec_op, vec_scalar_op

# y is expected output
# y-hat is actual output


def cross_entropy(output, expected_output):
  if expected_output == 1:
    return -log(output)
  else:
    return -log(1 - output)
  # return -(expected * math.log(actual) + (1.0 - expected) * math.log(1.0 - actual))

def mse(actual, expected):
  if type(actual) != list or type(expected) != list:
    raise Exception("Parameters need to be lists")

  if len(actual) != len(expected):
    raise Exception("Actual results must be of same size as expected results")

  summed_vector = vec_vec_op(actual, expected, lambda y_hat, y: (y_hat - y)**2)
  return vec_scalar_op(summed_vector, len(actual), lambda x,y: x / y)

  # e = actual - expected
  # return e * e
  # return e

# def mse_gradient(actual, expected, neuron, learning_rate):
#   if type(actual) != list or type(expected) != list:
#     raise Exception("Parameters need to be lists")

#   if len(actual) != len(expected):
#     raise Exception("Actual results must be of same size as expected results")

#   N = len(actual)
#   for w in range(len(neuron.weights)):
#     weight_deriv = 0
#     bias_deriv = 0
#     weight = neuron.weights[w]
#     bias = neuron.bias
#     for i in range(N):
#       weight_deriv += -2 * actual[i] * (actual[i] - (weight * expected[i] + bias))
#       bias_deriv += -2 * (actual[i] - (weight * expected[i] + bias))

#     neuron.weights[w] = weight - (weight_deriv / N) * learning_rate
#     neuron.bias = neuron.bias - (bias_deriv / N) * learning_rate

#   return