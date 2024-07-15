import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data
import matplotlib.pyplot as plt

nnfs.init()

# X, y = spiral_data(samples=100, classes=3)
X, y = sine_data(samples=100)
plt.scatter(X[:, 0], y[:,0])
plt.show()
