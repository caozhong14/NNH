#!/home/caozhong/anaconda2/bin/python
# --*--      coding: utf-8      --*--
# ***********************************
# @version : python 2.7.13
# @File    : BP2.py
# @Author  : caozhong
# @Email   : caoz10@foxmail.com 
# @Software: PyCharm
# @Time    : 3/16/17 11:54 PM
# ***********************************



import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
random.seed(0)

# calculate a random number where: a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# sigmoid funtion
def sigmoid(x):
    return 1/(1+math.exp(-x))
    #return math.tanh(x)

# derivative of sigmoid function
def dsigmoid(y):
    return y*(1 - y)
    #return 1.0 - x**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # create weights
        self.wi = np.zeros((self.ni, self.nh))
        #self.wi = np.array([[3, 7, 1, 0, 7, 7, 2],
        #                    [-7, -3, 0, 1, 0, 0, 2],
        #                    [-1, 0, -1, -1, -4, 5, 2],
        #                    ])
        self.wo = np.zeros((self.nh, self.no))
        #self.wo = np.array([[-3], [-2], [-4], [-2], [-4], [-3], [-4], [-3]])

        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-2, 2)

        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-20, 20)

        # last change in weights for momentum
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))
#        self.ci = self.wi
#        self.co = self.wo

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]

    # back propagate
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calcute error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        # N: learning rate
        # M: momentum factor
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5*(targets[k]-self.ao[k])**2
        return error

    # test
    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]), p[1])

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=10000, N=0.1, M=0.4):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error += self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)

def generate(n):
    pat = []
    for i in range(n):
        x1 = rand(-2, 2)
        x2 = rand(-2, 2)
        y = 10.0/(4*x1**2 - 2.1*x1**4 + 1/3*x1**6 + x1*x2 - 4*x2**2 + 4*x2**4 + 2)/11.0
        pat.append([[x1, x2], [y]])
    return pat

def curve():
    fig = plt.figure()
    ax = Axes3D(fig)
    x1 = np.arange(-3, 3, 0.1)
    x2 = np.arange(-3, 3, 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    y = 10.0/(4*x1**2 - 2.1*x1**4 + 1/3*x1**6 + x1*x2 - 4*x2**2 + 4*x2**4 + 2)/11.0
    ax.plot_surface(x1, x2, y)

def demo():
    pat = generate(200)

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 7, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)


if __name__ == '__main__':
    demo()

    curve()
    plt.show()
