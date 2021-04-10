import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import classification_report  # Prcision&Recall
from sklearn.preprocessing import PolynomialFeatures  # 定义多项式回归
from sklearn.datasets import make_gaussian_quantiles
from sklearn import linear_model
'''
梯度下降法-非线性逻辑回归
sklearn实现非线性逻辑回归
'''


class nonLinearLogisticRegression(object):
    def __init__(self):
        pass

    def load_data(self, filename):
        '''
        加载数据
        :param filename:
        :return:
        '''
        data = np.genfromtxt(filename, delimiter=",")
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1, np.newaxis]
        self.num = len(self.x_data)
        print(self.num)
        # 给样本添加偏置
        self.Xdata = np.concatenate((np.ones((118, 1)), self.x_data), axis=1)

    def draw(self):
        # 分别存放不同类别的数据
        cls0x = []
        cls0y = []
        cls1x = []
        cls1y = []
        for i in range(self.num):
            if self.y_data[i] == 0:
                cls0x.append(self.x_data[i, 0])
                cls0y.append(self.x_data[i, 1])
            else:
                cls1x.append(self.x_data[i, 0])
                cls1y.append(self.x_data[i, 1])
        # 画图以及图例
        scat0 = plt.scatter(cls0x, cls0y, marker='*')
        scat1 = plt.scatter(cls1x, cls1y, marker='o')
        plt.legend(handles=[scat0, scat1], labels=['tag-0', 'tag-1'])
        plt.show()

    def logistic_regression(self, scale=False):
        # 数据标准化
        if scale == True:
            self.Xdata = preprocessing.scale(self.Xdata)
        # 定义多项式回归
        poly_reg = PolynomialFeatures(degree=3)
        # 特征处理
        self.Xdata = poly_reg.fit_transform(self.Xdata)
        xMat = np.mat(self.Xdata)
        yMat = np.mat(self.y_data)
        # 学习率,迭代次数
        learn_rate = 0.03
        epochs = 50000
        costlist = []
        # 计算数据行列数，行代表数据个数，列代表权值个数
        rows, cols = np.shape(xMat)
        # 初始化权值
        ws = np.mat(np.ones((cols, 1)))

        for i in range(epochs+1):
            # xMat和wights相乘
            h = self.sigmoid(xMat*ws)
            # 计算误差
            ws_grad = xMat.T*(h-yMat)/rows
            ws = ws - learn_rate*ws_grad

            if i % 50 == 0:
                costlist.append(self.costfunc(xMat, yMat, ws))
        return ws, costlist

    # 逻辑回归函数：sigmoid函数
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    # 损失函数
    def costfunc(self, xMat, yMat, ws):
        left = np.multiply(yMat, np.log(self.sigmoid(xMat*ws)))
        right = np.multiply(1-yMat, np.log(1-self.sigmoid(xMat*ws)))
        return np.sum(left+right)/-(self.num)

    def predict(self, ws, scale=True):
        if scale==True:
            self.Xdata = preprocessing.scale(self.Xdata)
        xMat = np.mat(self.Xdata)
        ws = np.mat(ws)
        return [1 if x>= 0.5 else 0 for x in self.sigmoid(xMat*ws)]

    def logstic_regression_sklearn(self):
        x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
        plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
        plt.show()
        # 定义多项式回归
        poly_reg = PolynomialFeatures(degree=5)
        x_poly = poly_reg.fit_transform(x_data)
        logistic = linear_model.LogisticRegression()
        logistic.fit(x_poly, y_data)
        # 获取数据值所在的范围
        x_min, x_max = x_data[:, 0].min()-1, x_data[:, 0].max()+1
        y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
        # 生成网络矩阵
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        z = logistic.predict(poly_reg.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
        z = z.reshape(xx.shape)
        # 等高线图
        cs = plt.contour(xx, yy, z)
        plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
        plt.show()

        print('score：', logistic.score(x_poly, y_data))


if __name__ == '__main__':
    reg = nonLinearLogisticRegression()
    # reg.load_data("LRtestSet2.txt")
    # # reg.draw()
    # ws, cost = reg.logistic_regression()
    # # print(ws)
    # predict = reg.predict(ws)
    # print(classification_report(reg.y_data, predict))
    reg.logstic_regression_sklearn()

