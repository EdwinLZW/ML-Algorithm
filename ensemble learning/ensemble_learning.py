'''
集成学习：
    组合多个学习器，最后可以得到一个更好的学习器。
    集成学习算法：
        1、个体学习器之间不存在依赖关系，装袋算法（bagging）
        2、随机森林（RF）
        3、个体学习器之间存在强依赖关系，提升算法（boosting）
        4、Stacking
    bagging: bootstrap aggregating引导聚合， 一种有放回抽样
    RF = 决策树+Bagging+随机属性选择
        1、从样本集中用bagging的方式，随机选择n个样本
        2、从所以属性D中随机选择k个属性，然后从k个属性中选择最佳分割属性作为节点
        建立CART决策树
        3、重复以上两个步骤m次，建立m棵CART决策树
        4、这m棵CART决策树形成随机森林，通过投票表决结果。
    boosting:
        AdaBoost(Adaptive Boosting): 自适应增强，若前一个基本分类器被错误分类的
        样本的权值会增大，而正确分类的样本的权值会减小，并再次用来训练下一个基本分类
        器。同时，在每一轮迭代中，加入一个新的弱分类器，直到达到某个预定的足够小的错
        误率或达到预先指定的最大迭代次数才确定最终的强分类器。
    stacking:(voting)
        使用多个不同的分类器对训练集作预测，把预测得到的结果作为一个次级分类器的输入，
        次级分类器得到的输出是整个模型的预测结果。

'''

# bagging： bootstrap aggregating引导聚合
from sklearn import neighbors, tree, model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt


class ensembleLearning(object):
    KNN = neighbors.KNeighborsClassifier()
    Dtree = tree.DecisionTreeClassifier(max_depth=5)
    Logictic = LogisticRegression()

    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_datasets(self):
        iris = datasets.load_iris()
        self.x_data = iris.data[:, 1:3]
        self.y_data = iris.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data)

    def load_data(self):
        # 载入数据
        data = np.genfromtxt("LRtestSet2.txt", delimiter=",")
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data, test_size=0.25)

    def plot(self, model):
        # 获取数据值所在的范围
        x_min, x_max = self.x_data[:, 0].min() - 1, self.x_data[:, 0].max() + 1
        y_min, y_max = self.x_data[:, 1].min() - 1, self.x_data[:, 1].max() + 1
        # 生成网格矩阵
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        # ravel()将多维数组转换为一维数组的功能,np.c_是按行拼接两个矩阵。
        z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # ravel与flatten类似，多维数据转一维。flatten不会改变原始数据，ravel会改变原始数据
        z = z.reshape(xx.shape)
        # 等高线图
        plt.contourf(xx, yy, z)
        # 样本散点图
        plt.scatter(self.x_data[:, 0], self.x_data[:, 1], c=self.y_data)
        plt.show()

    def bagging(self, learner, n_estimators=100):
        self.load_datasets()
        bagging = BaggingClassifier(learner, n_estimators)
        bagging.fit(self.x_data, self.y_data)
        # 画图
        self.plot(bagging)
        # 样本散点图
        plt.scatter(self.x_data[:, 0], self.x_data[:, 1], c=self.y_data)
        plt.show()
        # 准确率
        print(bagging.score(self.x_data, self.y_data))

    def rf(self):
        self.load_data()
        RF = RandomForestClassifier(n_estimators=50)
        RF.fit(self.x_train, self.y_train)
        self.plot(RF)
        print(RF.score(self.x_test, self.y_test))

    def creat_data(self):
        # 生成2维正态分布，生成的数据按分位数分为两类，500个样本,2个样本特征
        x1, y1 = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
        # 生成2维正态分布，生成的数据按分位数分为两类，400个样本,2个样本特征均值都为3
        x2, y2 = make_gaussian_quantiles(mean=(3, 3), n_samples=500, n_features=2, n_classes=2)
        # 将两组数据合成一组数据
        self.x_data = np.concatenate((x1, x2))
        self.y_data = np.concatenate((y1, - y2 + 1))

    def boosting(self, learener):
        self.creat_data()
        adaboosting = AdaBoostClassifier(learener, n_estimators=10)
        adaboosting.fit(self.x_data, self.y_data)
        self.plot(adaboosting)
        print(adaboosting.score(self.x_data, self.y_data))

    def voting(self):
        self.load_datasets()
        voting = VotingClassifier([('knn', self.KNN), ('dtree', self.Dtree), ('lr', self.Logictic)])
        # 打包为元组的列表
        for clf, label in zip([self.KNN, self.Dtree, self.Logictic, voting],
                              ['KNN', 'Decision Tree', 'LogisticRegression', 'VotingClassifier']):
            scores = model_selection.cross_val_score(clf, self.x_data, self.y_data, cv=3, scoring='accuracy')
            print("Accuracy: %0.2f [%s]" % (scores.mean(), label))


if __name__ == '__main__':
    el = ensembleLearning()
    # el.load_data()
    el.voting()
