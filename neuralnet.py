import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))
def sigmoid_d(y):
    return y * (1 - y)

class NeuralNet:

    def __init__(self, n_in, n_mid, n_out):
        self.yi = np.zeros(n_in)
        self.yj = np.zeros(n_mid)
        self.yk = np.zeros(n_out)
        self.weight_mid = 2*np.random.random_sample((n_in,  n_mid)) - 1
        self.weight_out = 2*np.random.random_sample((n_mid, n_out)) - 1
        self.delta_mid = np.zeros((n_in,  n_mid))
        self.delta_out = np.zeros((n_mid, n_out))
        
        self.teacher = np.zeros((n_out, n_out))
        for l in range(n_out):
            self.teacher[l][l] = 1

    def set_param(self, eta_, alpha_):
        self.eta = eta_
        self.alpha = alpha_

    def weight_update(self, v):
        middle_vec = (v*sigmoid_d(self.yk))
        out_vec    = (self.weight_out * middle_vec).dot(np.ones(20)) * sigmoid_d(self.yj)
        delta__out = np.array( np.matrix(self.yj).T * middle_vec)
        delta__mid = np.array( np.matrix(self.yi).T * out_vec)
        
        self.delta_mid = self.eta * delta__mid + self.alpha * self.delta_mid
        self.delta_out = self.eta * delta__out + self.alpha * self.delta_out
        
        self.weight_mid += self.delta_mid
        self.weight_out += self.delta_out
        
    def forward(self, data):
        self.yi = data
        self.yj = sigmoid(self.yi @ self.weight_mid)
        self.yk = sigmoid(self.yj @ self.weight_out)
        return self.yk

    def evaluate(self, dataset):
        s = 0
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                t = self.teacher[i] - self.forward(dataset[i][j])
                s += np.sum((t * t))/len(t)
        s /= dataset.shape[0]
        return s

    def learn(self, dataset):
        for i in range(dataset.shape[0]):
            for j in range(dataset.shape[1]):
                t = self.teacher[i] - self.forward(dataset[i][j])
                self.weight_update(t)

    def predict(self, data):
        y = self.forward(data)
        return np.argmax(y)
    def test(self, dataset):
        ans_list = []
        for i in range(dataset.shape[0]):
            count = 0
            for j in range(dataset.shape[1]):
                if(self.predict(dataset[i][j]) == i):
                    count += 100
            ans_list.append(count / dataset.shape[1])
        ans_list.append(np.array(ans_list).sum() / dataset.shape[0])
        return ans_list

