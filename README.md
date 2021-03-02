
# Recurrent Neural Net

- Input
- Weights
- Activation Function
- Output


- Need to store weights
- Choose a type of activation function

#
| Artificial Neural Net | Deep Neural Net     |
| :---:                 | :---:               |
| 1 hidden layer        | many hidden layers  |
#

inputs are numeric data points

activation function maps a network's inputs and outputs

non-linear activation functions can:
- allow backpropagation
- allow stacking of layers

types of non-linear activation functions:
- sigmoid/logistic
- tanh/hyperbolic tangent
- ```relu (rectified linear unit)```
- ```leaky relu```
- ```parametric relu```
- softmax
- swish

Neurons are connected to neurons of next layer with weights

Data Sets:
- california ervine


implement backpropagation with sigmoid function
(1 hidden layer)
(exclusive or will not work)
(and, or, nor, xor)

input,input,expected output
0,0,0
0,1,0
1,0,0
1,1,1


Todo:
Create a RNN class that can:
- take whole json file and turn it into a sequence of layers
  - connecting layer ids with layer classes --> right now, only can store layer ids of next_layer

Add in backpropagation
Add in sigmoid activation function instead of default_fn



RNN = Array of layers
Layer = Array of neurons
Neuron = Dictionary w/ ```weights``` and ```bias``` keys

RNN = [
  Input Layer = [
    ...
  ],
  Hidden Layer = [
    Neuron = { weights: [w1, w2, ..., wn], bias: b },
    ...
  ],
  Output Layer = [
    ...
  ]
]