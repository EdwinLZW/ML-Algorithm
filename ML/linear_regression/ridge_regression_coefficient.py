import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


class ridgeRegressionCoefficient(object):
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

    def ridge_regression_coefficient(self):
        '''

        :return:
        '''
        # 选取50个岭回归系数（𝜆 ）
        ridge_coef = np.linspace(0.001, 1)
        # 创建模型并训练
        model = linear_model.RidgeCV(alphas=ridge_coef, store_cv_values=True)
        model.fit(self.x_data, self.y_data)
        # 选取最优岭系数
        print(model.alpha_)
        # loss值
        print(model.cv_values_)
        # 岭系数与Loss平均值的关系
        plt.plot(ridge_coef, model.cv_values_.mean(axis=0))
        plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
        plt.show()

    # 标准方程法
    def ridge_regression_coef_with_standard_equation(self):
        pass


if __name__ == '__main__':
    reg = ridgeRegressionCoefficient()
    reg.load_data("longley.csv")
    reg.ridge_regression_coefficient()