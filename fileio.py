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