from math import exp

def sigmoid(activation_sum):
  return 1.0 / (1.0 + exp(-activation_sum))

def sigmoidp(output):
  return output * (1.0 - output)