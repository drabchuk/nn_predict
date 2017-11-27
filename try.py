from neuralnet import NeuralNet
import numpy as np

nn = NeuralNet([2, 4, 1])
x = np.array([[1, 2]]).T
print(x.shape)
o = nn.forward(x)
