import json

def tojson(serialized_data, file):
  with open(file, 'w') as outfile:
    json.dump(serialized_data, outfile)
  return

def parse_json(file):
  with open(file) as json_file:
    data = json.load(json_file)
    return data
  return None

def write_json(serialized_data, file):
  with open(file, 'w') as outfile:
    return json.dump(serialized_data, outfile)

def read_json(file):
  with open(file) as json_file:
    data = json.load(json_file)
    return json.loads(data)