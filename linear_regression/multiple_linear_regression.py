import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D


class multipleLinearRegression(object):
    def __init__(self):
        self.learn_rate = 0.0001    # 学习率
        self.theta0 = 0
        self.theta1 = 0
        self.theta2 = 0
        self.epochs = 1000          # 迭代次数
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
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1]
        self.num = len(self.y_data)

    def Lost_function(self):
        '''
        损失函数/代价函数，最小二乘法
        :return:
        '''
        error_sum_squares = 0
        # 最小误差平方和
        for i in range(0, self.num):
            error_sum_squares += (self.y_data[i] - (self.theta0 + self.theta1 * self.x_data[i, 0]+self.theta2 * self.x_data[i, 1])) ** 2
        return error_sum_squares / float(self.num) / 2

    def Lsm(self):
        '''
        梯度下降法迭代作回归
        :return:
        '''
        for i in range(self.epochs):
            theta0_grad = 0
            theta1_grad = 0
            theta2_grad = 0
            for j in range(0, self.num):
                theta0_grad += (1 / self.num) * ((self.theta0 + self.theta1 * self.x_data[j, 0]+self.theta2 * self.x_data[j, 1]) - self.y_data[j])
                theta1_grad += (1 / self.num) * ((self.theta0 + self.theta1 * self.x_data[j, 0]+self.theta2 * self.x_data[j, 1]) - self.y_data[j]) * \
                              self.x_data[j, 0]
                theta2_grad += (1 / self.num) * (
                            (self.theta0 + self.theta1 * self.x_data[j, 0] + self.theta2 * self.x_data[j, 1]) -
                            self.y_data[j]) * \
                               self.x_data[j, 1]
            self.theta0 = self.theta0 - (self.learn_rate * theta0_grad)
            self.theta1 = self.theta1 - (self.learn_rate * theta1_grad)
            self.theta2 = self.theta2 - (self.learn_rate * theta2_grad)
        self.draw(self.theta0, self.theta1, self.theta2)

    def regressionWithSklearn(self):
        '''
        用sklearn作回归
        :return:
        '''
        model = LinearRegression()
        model.fit(self.x_data, self.y_data)
        # 系数
        print("coef:", model.coef_)
        # 截距
        print("intercept:", model.intercept_)
        x_test = [[172, 4]]
        predict = model.predict(x_test)
        print("predict:", predict)
        self.draw(model.intercept_, model.coef_[0], model.coef_[1])

    # noinspection PyUnresolvedReferences
    def draw(self, t0, t1, t2):
        '''
        :param t0:
        :param t1:
        :param t2:
        :return:
        '''
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(self.x_data[:, 0], self.x_data[:, 1], self.y_data, c='r',
                   marker='o', s=100)
        x0 = self.x_data[:, 0]
        x1 = self.x_data[:, 1]
        # 生成网络矩阵,类似离散中的复合矩阵
        x0, x1 = np.meshgrid(x0, x1)
        z = t0 + t1*x0 + t2*x1
        # 画图
        ax.plot_surface(x0, x1, z)
        ax.set_xlabel('miles')
        ax.set_ylabel('Num of deliveries')
        ax.set_zlabel('Time')

        plt.show()


if __name__ == '__main__':
        fileNmae = "Delivery.csv"
        regression = multipleLinearRegression()
        regression.load_data(fileNmae)
        regression.Lsm()
        regression.regressionWithSklearn()