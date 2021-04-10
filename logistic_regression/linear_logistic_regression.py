import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import classification_report  #Prcision&Recall
from  sklearn import linear_model


class linearLogisticRegression(object):
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

        # 给样本添加偏置
        self.Xdata = np.concatenate((np.ones((100, 1)), self.x_data), axis=1)

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

    def logistic_regression(self, scale=True):
        # 数据标准化
        if scale == True:
            self.Xdata = preprocessing.scale(self.Xdata)
        xMat = np.mat(self.Xdata)
        yMat = np.mat(self.y_data)
        # 学习率,迭代次数
        learn_rate = 0.001
        epochs = 10000
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
        logstic = linear_model.LogisticRegression()
        logstic.fit(self.x_data, self.y_data)
        print(logstic.coef_)
        x_test = np.array([[-4], [3]])
        y_test = (-logstic.intercept_ - x_test * logstic.coef_[0][0]) / logstic.coef_[0][1]
        plt.plot(x_test, y_test, 'k')
        self.draw()
        plt.show()


if __name__ == '__main__':
    reg = linearLogisticRegression()
    reg.load_data("LRtestSet.csv")
    ws, cost = reg.logistic_regression()
    # 测试
    # x_test = [[-4], [3]]
    # y_test = (-ws[0] - x_test * ws[1]) / ws[2]
    # plt.plot(x_test, y_test, 'k')
    # reg.draw()
    # 画出loss值
    # x = np.linspace(0, 1000, 201)
    # plt.plot(x, cost, c='r')
    # plt.show()

    # 预测
    # predict = reg.predict(ws)
    # print(classification_report(reg.y_data, predict))

    reg.logstic_regression_sklearn()

