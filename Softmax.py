import numpy as np

layer_outputs = [4.8, 1.21, 2.385]
E = 2.71828182846

exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
# print(exp_values)

norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value/norm_base)
# print(norm_values)

# With Numpy
exp_values = np.exp(layer_outputs)
print(exp_values)

norm_values = exp_values / np.sum(exp_values)
print(norm_values)

# With batch inputs
layer_outputs = np.array([[4.8, 1.21, 2.385],
                          [8.9, -1.81, 0.2],
                          [1.41, 1.051, 0.026]])

print('Sum axis 1(row wise sum), but keep the same dimensions as input:')
print(np.sum(layer_outputs, axis=1, keepdims=True))
