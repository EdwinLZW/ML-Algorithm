import numpy as np


class bpNeuralNetwork(object):
    def __init__(self):
        # 权值初始化,学利率初始化
        self.V = np.random.random((3, 4)) * 2 - 1
        self.W = np.random.random((4, 1)) * 2 - 1
        self.lr = 0.1

    # 激活函数以及激活函数的求导函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def desigmoid(self, h):
        return h * (1 - h)

    def updata(self, Xdata, Ydata):
        # 隐藏层输出
        L1 = self.sigmoid(np.dot(Xdata, self.V))
        # 输出层输出
        L2 = self.sigmoid(np.dot(L1, self.W))

        # 权值调整规则，delta学习规则，与单层感知器一样
        '''
        Delta学习规:
        Δ𝑊𝑙 = −𝜂𝜕𝐸/𝜕𝑊(𝑙)= 𝜂 𝑋(𝑙).𝑇*𝛿(𝑙)
        𝛿(𝐿) = (𝑡 − 𝑦) *𝑓′(𝑋(𝐿)𝑊(𝐿))          输出层的delta
        𝛿(𝑙) = 𝛿(𝑙+1) 𝑊(𝑙+1).𝑇*𝑓′(𝑋(𝑙)𝑊(𝑙))    反向delta 
        '''
        Ydata = Ydata.reshape((4, 1))
        L2_delta = -(L2 - Ydata) * self.desigmoid(L2)
        L1_delta = L2_delta.dot(self.W.T) * self.desigmoid(L1)

        W_C = self.lr * L1.T.dot(L2_delta)
        V_C = self.lr * Xdata.T.dot(L1_delta)

        self.W = self.W + W_C
        self.V = self.V + V_C

    def bp_neural_network(self, xdata, ydata):
        for i in range(20000):
            self.updata(xdata, ydata)
        L1 = self.sigmoid(np.dot(Xdata, self.V))
        L2 = self.sigmoid(np.dot(L1, self.W))
        for i in range(L2.shape[0]):
            if L2[i] >= 0.5:
                L2[i] = 1
            else:
                L2[i] = 0
        print(L2)
        return self.W, self.V


if __name__ == '__main__':
    # 加载数据
    x_data = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
    Xdata = np.concatenate((np.ones((4, 1)), x_data), axis=1)
    Ydata = np.array([0, 1, 1, 0])
    # Ydata = Ydata.reshape((4, 1))
    # print(Ydata)
    bp_NN = bpNeuralNetwork()
    W, V = bp_NN.bp_neural_network(Xdata, Ydata)
    # 预测
    pre = np.array([1, 2, 2])
    L1 = bp_NN.sigmoid(np.dot(pre, V))
    L2 = bp_NN.sigmoid(np.dot(L1, W))
    print(L2)