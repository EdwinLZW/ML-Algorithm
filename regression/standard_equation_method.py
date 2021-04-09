import numpy as np
import matplotlib.pyplot as plt


class standardEquationMethod(object):
    def __init__(self):
        pass

    def load_data(self, filename):
        '''
        加载数据
        :param filename:
        :return:
        '''
        data = np.genfromtxt(filename, delimiter=",")
        self.x_data = data[:, 0, np.newaxis]
        self.y_data = data[:, 1, np.newaxis]
        self.num = len(self.x_data)
        # plt.scatter(self.x_data, self.y_data)
        # plt.show()

    # 标准方程法
    def standard_equation_method(self):
        # 给样本添加偏置项
        Xdata = np.concatenate((np.ones((100, 1)), self.x_data), axis=1)

        xMat = np.mat(Xdata)
        yMat = np.mat(self.y_data)
        xTx = xMat.T * xMat  # 矩阵乘法
        # 计算矩阵的值,如果值为0，说明该矩阵没有逆矩阵
        if np.linalg.det(xTx) == 0.0:
            print("This matrix cannot do inverse")
            return
        # xTx.I为xTx的逆矩阵
        ws = xTx.I * xMat.T * yMat

        # 画图
        x_test = np.array([[30], [60]])
        y_test = ws[0] + x_test * ws[1]
        plt.plot(self.x_data, self.y_data, 'b.')
        plt.plot(x_test, y_test, 'r')
        plt.show()


if __name__ == '__main__':
    reg = standardEquationMethod()
    reg.load_data("data.csv")
    reg.standard_equation_method()