inputs = [1.0,2.0,3.0,2.5]

weights = [[0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases =[2.0,3.0, 0.5]

#output of current layer
layer_outputs = []

#For each neuron
for neuron_wights, neuron_bias in zip(weights, biases):
    # Iterated neuron output save
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_wights):
        # Multiply this input by associated weight
        # and add to the neuron’s output variable
        neuron_output+= n_input*weight
    # Add bias
    neuron_output+=neuron_bias
    # putting the neuron's result to the layer's output
    layer_outputs.append(neuron_output)

print(layer_outputs)