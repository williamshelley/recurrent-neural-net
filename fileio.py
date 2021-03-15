import json

def write_json(serialized_data, file):
  try:
    with open(file, 'w') as outfile:
      return json.dump(serialized_data, outfile)
  except IOError:
    print(file + " could not be written to")

def read_json(file):
  try:
    with open(file) as json_file:
      data = json.load(json_file)
      return json.loads(data)
  except IOError:
    print(file + " was not accessible")
    return None