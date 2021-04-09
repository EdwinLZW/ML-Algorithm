import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


class lassoAndElasticNet(object):
    def __init__(self):
        pass

    def load_data(self, filename):
        '''
        加载数据
        :param filename:
        :return:
        '''
        data = np.genfromtxt(filename, delimiter=",")
        self.x_data = data[1:, 2:]
        self.y_data = data[1:, 1]
        self.num = len(self.x_data)
        # plt.scatter(self.x_data, self.y_data)
        # plt.show()

    def Lasso(self):
        '''

        :return:
        '''
        model = linear_model.LassoCV()
        model.fit(self.x_data, self.y_data)
        # lasso系数
        print(model.alpha_)
        # 相关系数
        print(model.coef_)
        # 预测值
        print(model.predict(self.x_data[2, np.newaxis]))
        print(self.y_data[2])

    def elastic_net(self):
        '''

        :return:
        '''
        model = linear_model.ElasticNetCV()
        model.fit(self.x_data, self.y_data)
        # 弹性网系数
        print(model.alpha_)
        # 相关系数
        print(model.coef_)
        # 预测值
        print(model.predict(self.x_data[2, np.newaxis]))
        print(self.y_data[2])


if __name__ == '__main__':
    reg = lassoAndElasticNet()
    reg.load_data("longley.csv")
    reg.Lasso()
    reg.elastic_net()