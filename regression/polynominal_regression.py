import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class polynomialRegression(object):
    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.num = 0

    def load_data(self, filename):
        '''
        加载数据
        :param filename:
        :return:
        '''
        data = np.genfromtxt(filename, delimiter=",")
        self.x_data = data[1:, 1]
        self.y_data = data[1:, 2]
        self.num = len(self.x_data)

    def regressionWithSklearn(self):
        '''
        用sklearn作回归
        :return:
        '''
        Xdata = self.x_data[:, np.newaxis]
        Ydata = self.y_data[:, np.newaxis]
        # 定义多项式回归,degree的值用来调节多项式的特征
        poly_reg = PolynomialFeatures(degree=5)
        poly_xdata = poly_reg.fit_transform(Xdata)
        # 创建并拟合模型
        model = LinearRegression()
        model.fit(poly_xdata, Ydata)  # 求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。
        # 画图
        plt.plot(Xdata, Ydata, 'b.')
        plt.plot(Xdata, model.predict(poly_xdata), 'r')
        plt.show()




if __name__ == '__main__':
    reg = polynomialRegression()
    reg.load_data("job.csv")
    reg.regressionWithSklearn()