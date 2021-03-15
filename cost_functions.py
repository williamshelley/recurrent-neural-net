from math import log

# a=actual, e=expected (both are float values)
def cross_entropy(a, e):
  # if expected_output == 1:
  #   return -log(output)
  # else:
  #   return -log(1 - output)
  return -(e * log(a) + (1.0 - e) * log(1.0 - a))

def mse(a, e):
  return e - a