import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        # weight and biases initialization
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # we are doing (inputs x neurons) instead (neurons x inputs) 
                                                                  # so that don't have to calculate transpose every time
        self.biases = np.zeros((1, n_neurons))
        
    
    def forward(self, inputs):
        # Calculate output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

dense1.forward(X)

print(dense1.output[:5])




