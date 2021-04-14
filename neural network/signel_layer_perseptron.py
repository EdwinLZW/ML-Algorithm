import numpy as np
import matplotlib.pyplot as plt
'''
感知器的学习规则：
    学习信号等于神经元期望输出与实际输出之差，即 r=t-y,
其中t为Ydata,y为激活函数输出值，且其权值调整公式为：
∆𝑤𝑖 = 𝜂(𝑡 − 𝑦)𝑥𝑖，η表示学习率，t表示正确的标签，y为激活函
数输出值。𝑤𝑖 = 𝑤𝑖 + ∆𝑤𝑖，E = 0.5*((t-y)**2),梯度为：
(𝑡 − 𝑦)𝑥𝑖，sign()函数在-1到1之间的梯度为1
'''


class signelLayerPerseptron(object):
    def __init__(self, Xdata, Ydata, xor=False):
        # 权值初始化，3行一列，生成一组服从“-1~1”均匀分布的随机样本值。
        self.W = (np.random.random((6, 1))-0.5) * 2
        # 设置学习率
        self.lr = 0.1
        # 输出
        self.out = None

        self.Xdata = Xdata
        self.Ydata = Ydata
        # 加偏置项，axis=1表示对应行的数组进行拼接
        self.x_data = np.concatenate((np.ones((4, 1)), Xdata), axis=1)
        self.xor = xor

    def update(self):
        out = np.sign(np.dot(self.x_data, self.W))
        wc = self.lr * self.x_data.T.dot(self.Ydata - out)
        self.W = self.W + wc

    def drow(self, x1, y1, x2, y2):
        plt.scatter(x1, y1, c='r')
        plt.scatter(x2, y2, c='y')
        if not self.xor:
            plt.plot(self.Xdata, self.Xdata * (-self.W[1] / self.W[2]) + (-self.W[0] / self.W[2]), 'r')
        plt.show()

    def signel_layer_prepeseptron(self):
        for i in range(100):
            self.update()
            out = np.sign(np.dot(self.x_data, self.W))
            if (out == self.Ydata).all():
                break

    def predict(self, pre):
        plt.plot(pre[0], pre[1], 'b*')
        print(np.sign(self.W[0] + self.W[1] * pre[0] + self.W[2] * pre[1]))

    # 解决异或问题,增加非线性变量
    def solve_XOR_problem(self,  x1, y1, x2, y2):
        for i in range(1000):
            self.update()
        xd = np.linspace(-1, 2)
        plt.plot(xd, self.cal(xd, 1), 'r')
        plt.plot(xd, self.cal(xd, 2), 'r')
        self.drow(x1, y1, x2, y2)

    def cal(self, xdata, root):
        a = self.W[5]
        b = self.W[2]+xdata*self.W[4]
        c = self.W[0]+xdata*self.W[1] + xdata*xdata*self.W[3]
        if root == 1:
            return (-b + np.sqrt(b*b - 4*a*c))/(2*a)
        if root == 2:
            return (-b - np.sqrt(b*b - 4*a*c))/(2*a)


if __name__ == '__main__':
    # 创建数据,按分母布局
    Xdata = np.array([[0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1],
                      [1, 0, 1, 0, 0],
                      [1, 1, 1, 1, 1]])
    Ydata = np.array([[-1],
                      [1],
                      [1],
                      [-1]])
    # 正样本
    x1 = [0, 1]
    y1 = [1, 0]
    # 负样本
    x2 = [0, 1]
    y2 = [0, 1]
    slp = signelLayerPerseptron(Xdata, Ydata, xor=True)
    slp.solve_XOR_problem(x1, y1, x2, y2)

    '''
    slp.signel_layer_prepeseptron()
    pre = np.array([0, 5])
    slp.predict(pre)
    slp.drow(x1, y1, x2, y2)
    '''



