import numpy as np
import random
from datetime import datetime
import time


class NeuralNet:

    def __init__(self, topology, theta=None):
        self.topology = topology
        self.depth = len(topology)
        self.theta = []
        self.theta_grad = []
        self.__a = []
        if theta is None:
            for i in range(len(topology) - 1):
                e_init = np.sqrt(6.0 / (topology[i] + topology[i + 1]))
                np.random.seed()
                ran = np.random.normal(0, e_init, (topology[i] + 1, topology[i + 1]))
                self.theta.append(ran)
                self.theta_grad.append(np.zeros((topology[i] + 1, topology[i + 1])))
                self.__a.append(np.zeros((topology[i] + 1, 1)))
            self.__a.append(np.zeros((topology[-1], 1)))
        else:
            self.theta = theta
            for i in range(len(topology) - 1):
                self.theta_grad.append(np.zeros((topology[i] + 1, topology[i + 1])))
                self.__a.append(np.zeros((topology[i] + 1, 1)))
            self.__a.append(np.zeros((topology[-1], 1)))

    @staticmethod
    def __activation(z):
        return NeuralNet.__sigmoid(z)

    @staticmethod
    def __relu(z):
        return np.maximum(z, 0.0)

    @staticmethod
    def __relu_grad(z):
        return 1.0 if z > 0.0 else 0.0

    @staticmethod
    def __sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def __sigmoid_grad(z):
        sigma = NeuralNet.__sigmoid(z)
        return sigma * (1.0 - sigma)

    def forward(self, x):
        self.__a[0] = x.reshape(self.topology[0], 1)
        for i in range(len(self.topology) - 1):
            a = np.vstack((np.ones((1, 1)), self.__a[i]))
            z = np.dot(self.theta[i].T, a)
            self.__a[i + 1] = NeuralNet.__activation(z)
        return self.__a[-1]

    def backprop(self, delta_out):
        d = delta_out * self.__a[self.depth - 1] * (1.0 - self.__a[self.depth - 1])
        for i in range(self.depth - 2, -1, -1):
            a = np.vstack((np.ones((1, 1)), self.__a[i]))
            self.theta_grad[i] = np.dot(a, d.T)
            d = np.dot(self.theta[i], d)
            d = d[1:, :] * self.__a[i] * (1.0 - self.__a[i])
        return self.theta_grad

    def save(self, file_name):
        f = open(file_name, 'w')
        depth = len(self.topology)
        f.write(str(depth))
        f.write('\n')
        for i in range(depth):
            f.write(str(self.topology[i]))
            f.write('\n')
        for i in range(depth - 1):
            for j in range(self.topology[i] + 1):
                for k in range(self.topology[i + 1]):
                    f.write(str(self.theta[i][j, k]) + ' ')
                f.write('\n')
        f.close()

    def norm_l2(self):
        acc = 0
        for i in range(len(self.topology) - 1):
            tL2 = np.mean(self.theta[i] ** 2)
            acc += tL2
        return acc
