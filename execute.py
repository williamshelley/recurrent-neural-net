from layer import Layer
import json

layer_2 = Layer(n_neurons=3, id=2)
layer_1 = Layer(n_neurons=5, id=1, next_layer=layer_2)
# layer_2.set_next_layer(layer_1)
layer_1.set_neurons_with_weights([1,2,3,4,5])
serialized_layer_1 = layer_1.serialize()
# deserialized_layer1 = Layer.deserialize(layer_1.serialize())
# deserialized_layer1.set_neurons_with_weights([3,2,1])

with open('data.json') as json_file:
    data = json.load(json_file)
    print(data)
    deserialized_layer_1 = Layer.deserialize(data)
    print([x.weights for x in deserialized_layer_1.neurons])
    print(deserialized_layer_1.next_layer_id)

# with open('data.json', 'w') as outfile:
#     json.dump(serialized_layer_1, outfile)



# print([x.weights for x in deserialized_layer1.neurons])

layer_1.activate_all()
