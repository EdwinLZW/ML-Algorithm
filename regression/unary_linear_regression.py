import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


class unaryLinearRegression(object):
    def __init__(self):
        self.learn_rate = 0.0001    # 学习率
        self.slope = 0              # 斜率
        self.intercept = 0          # 截距
        self.epochs = 100           # 迭代次数
        self.num = 0
        self.x_data = None
        self.y_data = None

    def load_data(self, filename):
        '''
        加载数据
        :param filename:
        :return:
        '''
        data = np.genfromtxt(filename, delimiter=",")
        self.x_data = data[:, 0]
        self.y_data = data[:, 1]
        self.num = len(self.x_data)
        # plt.scatter(self.x_data, self.y_data)
        # plt.show()

    def Lost_function(self):
        '''
        损失函数/代价函数，最小二乘法
        :return:
        '''
        error_sum_squares = 0
        # 最小误差平方和
        for i in range(0, self.num):
            error_sum_squares += (self.y_data[i]-(self.intercept+self.slope*self.x_data[i]))**2
        return error_sum_squares/float(self.num)/2

    def Lsm(self):
        '''
        梯度下降法迭代求斜率和截距，作回归
        :return:
        '''
        for i in range(self.epochs):
            intercept_grad = 0
            slope_grad = 0
            for j in range(0, self.num):
                intercept_grad += (1/self.num) *((self.slope*self.x_data[j]+self.intercept)-self.y_data[j])
                slope_grad += (1/self.num) *((self.slope*self.x_data[j]+self.intercept)-self.y_data[j])*self.x_data[j]
            self.intercept = self.intercept-(self.learn_rate*intercept_grad)
            self.slope = self.slope - (self.learn_rate*slope_grad)
        plt.plot(regression.x_data, regression.y_data, 'b.')
        plt.plot(regression.x_data, regression.slope * regression.x_data + regression.intercept, 'r')
        plt.show()

    def regressionWithSklearn(self):
        '''
        用sklearn作回归
        :return:
        '''
        Xdata = self.x_data[:, np.newaxis]
        Ydata = self.y_data[:, np.newaxis]
        model = LinearRegression()
        model.fit(Xdata, Ydata)
        plt.plot(Xdata, Ydata, 'b.')
        plt.plot(Xdata, model.predict(Xdata), 'r')
        plt.show()


if __name__ == '__main__':
    fileNmae = "data.csv"
    regression = unaryLinearRegression()
    regression.load_data(fileNmae)
    regression.Lsm()
    regression.regressionWithSklearn()