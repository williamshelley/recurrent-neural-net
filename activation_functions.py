import math

def sigmoid_logistic(activation_sum):
  return 1 / (1 + math.exp(-activation_sum))