INPUT_INDEX = 0
OUTPUT_INDEX = 1
INPUT_KEY = "input"
OUTPUT_KEY = "output"

def format_error(dataset_idx, actual_val, expected_val):
  return {"index": dataset_idx, "actual": actual_val, "expected": expected_val}

def islist(data):
  return type(data) == list

def isdict(data):
  return type(data) == dict

def get_expected(data):
  if islist(data): 
    return data[OUTPUT_INDEX]
  elif isdict(data): 
    return data[OUTPUT_KEY]
  else:
    raise Exception("Unsupported expected output type (" + str(type(data)) + ")")

def get_inputs(data):
  if islist(data):
    return data[INPUT_INDEX]
  elif isdict(data):
    return data[INPUT_KEY]
  else:
    raise Exception("Unsupported input type (" + str(type(data)) + ")")

def round_results(results):
  return [[round(x) for x in data] for data in results]

def count_rounded_binary(dataset, results):
  actual = {}
  expected = {}
  errors = []
  rounded = round_results(results)
  for i in range(len(dataset)):
    data = dataset[i]
    expected_val = str(get_expected(data))
    actual_val = str(rounded[i])

    if expected_val in expected:
      expected[expected_val] += 1
    else:
      expected[expected_val] = 1

    if actual_val != expected_val:
      errors.append(format_error(i, actual_val, expected_val))

    if actual_val in actual:
      actual[actual_val] += 1
    else:
      actual[actual_val] = 1

  return (actual, expected, errors)

def dec_to_percent_str(decimal):
  percent = decimal * 100
  return str(round(percent, 2)) + "%"

def print_summary(dataset, results, total_network_err, network_file):
  print("network:",network_file)
  actual,expected,errors = count_rounded_binary(dataset, results)
  print("actual:",actual)
  print("expected:",expected)
  print("errors:",errors)

  print("Number of Errors:", len(errors))
  print("Error Percentage in Results:", dec_to_percent_str(len(errors) / len(dataset)))
  print("Total Network Error Percentage:", dec_to_percent_str(total_network_err))
  return