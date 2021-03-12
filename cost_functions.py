from math import  log

# y is expected output
# y-hat is actual output


def cross_entropy(output, expected_output):
  if expected_output == 1:
    return -log(output)
  else:
    return -log(1 - output)
  # return -(expected * math.log(actual) + (1.0 - expected) * math.log(1.0 - actual))

def mse(actual, expected):
  e = actual - expected
  # return e * e
  return e