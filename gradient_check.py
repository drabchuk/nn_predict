from neuralnet import NeuralNet
import numpy as np


def loss(a, y):
    return np.mean(np.square(a - y)) / 2.0

topology = [3, 5, 4, 1]
nn = NeuralNet(topology)
train_set = np.array(
    [[1, 2, 3], [-1, -2, -2.5], [-3, 2, 4], [5, 2, -1]]
).T
labels = np.array([[1, 0, 0, 1]]).T
print(train_set.shape)
print(labels.shape)
a = nn.forward(train_set[:, 0])
delta_out = a - labels[0, 0]
backprop_grad = nn.backprop(delta_out)
print(backprop_grad)

num_grad = []
for i in range(len(topology) - 1):
    t = np.zeros((topology[i] + 1, topology[i + 1]), dtype=float)
    num_grad.append(t)

for i in range(len(topology) - 1):
    for j in range(topology[i] + 1):
        for k in range(topology[i + 1]):
            a1 = nn.forward(train_set[:, 0])
            l1 = loss(a1, labels[0])
            nn.theta[i][j, k] += 0.001
            a2 = nn.forward(train_set[:, 0])
            l2 = loss(a2, labels[0])
            num_grad[i][j, k] = (l2 - l1) / 0.001
            nn.theta[i][j, k] -= 0.001

print(num_grad)

