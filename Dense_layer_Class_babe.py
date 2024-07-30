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
    

class Acivation_ReLU:
    def __init__(self) -> None:
        pass
    
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def __init__(self) -> None:
        pass
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # To prevent neuron exploit we are subtracting the max
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

#Common Loss class
class Loss:
   
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):

    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) 

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        #Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods






X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3)

activation1 = Acivation_ReLU()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()

# First through neuron
dense1.forward(X)
# Then through activation function
activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)

print('loss:', loss)


