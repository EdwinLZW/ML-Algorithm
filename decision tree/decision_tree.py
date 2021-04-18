import graphviz
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
'''
决策树：
    （1）一种基本的分类和回归方法
    （2）它是一种定义在特征空间与类空间上的条件概率分布。
    （3）根据损失函数最小化的原则建立决策树模型。其损失函数通常是正则化的极大似然函数。
    （4）学习时的三个步骤：a、特征选择；b、决策树的生成；c、决策树的修剪。
    （5）三种基本算法：a、ID3算法(信息增益)；b、C4.5算法（信息增益比）；c、CART算法(基尼指数最小化准则
    )。
    （6）熵公式：H(X) = -∑P(xi)logP(xi)
信息增益算法：
     训练数据集：D；特征集A（A1，A2，A3，A4）
     （1）计算训练集的熵：H(D)
     （2）计算特征A对数据集D的经验条件熵：H(D|A)
     （3）计算信息增益：g(D,A)=H(D)-H(D|A)

信息增益比：gr(D,A)=g(D,A)/Ha(D),Ha(D)=-∑(|Di|/|D|)logP(|Di|/|D|)(Ha(D)表示训练数据集D关于特征值A的值的熵)
CART算法：
    1、决策树生成：基于训练集生成决策树，生成决策树要尽量大。
    2、决策树剪枝：用验证数据集对已生成的树进行剪枝并选择最优子树，这时
       用损失函数最小作为剪枝的标准。
'''


class decisionTree(object):
    def __init__(self):
        self.header = None
        self.reader = None
        self.feature = []
        self.label = []
        self.x_data = None
        self.y_data = None
        self.model = None
        # self.load_data(filename)

    def load_data(self, filename):
        dt = open(filename, "r")
        self.reader = csv.reader(dt)
        self.header = self.reader.__next__()
        for row in self.reader:
            rowDict = {}
            self.label.append(row[-1])
            for i in range(1, len(row)-1):
                rowDict.setdefault(self.header[i], row[i])
            self.feature.append(rowDict)
        self.data_transform()

    def data_transform(self):
        # 把数据做成01格式,特征向量化,迭代器适合字典
        self.vec = DictVectorizer()
        self.x_data = self.vec.fit_transform(self.feature).toarray()  # 根据字母递增顺序排列
        # 把标签转换成01表示
        self.lb = preprocessing.LabelBinarizer()
        self.y_data = self.lb.fit_transform(self.label)

    def create_tree_model(self, feature_name, label, criterion='entropy', create_tree=True):
        # 创建决策树模型，并训练
        self.model = tree.DecisionTreeClassifier(criterion=criterion)  # 支持的标准有"gini"代表的是Gini impurity(不纯度)与"entropy"代表的是information gain（信息增益）。
        self.model.fit(self.x_data, self.y_data)

        # 生成决策树
        if create_tree:
            self.create_tree(feature_name, label)

    def create_tree(self, feature_name, label):
        dot_data = tree.export_graphviz(self.model,
                                        out_file=None,
                                        # feature_names=self.vec.get_feature_names(),
                                        # class_names=self.lb.classes_,
                                        feature_names=feature_name,
                                        class_names=label,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render('computer')

    def predict(self, xdata):
        return self.model.predict(xdata.reshape(1, -1))

    def linear_dichotomy(self):
        # 载入数据
        data = np.genfromtxt("LRtestSet.csv", delimiter=",")
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1]
        self.create_tree_model(['x', 'y'], ['label0', 'label1'], criterion='gini')
        self.draw_contour_line()

    def non_linear_dichotomy(self):
        # 载入数据
        data = np.genfromtxt("LRtestSet2.txt", delimiter=",")
        self.x_data = data[:, :-1]
        self.y_data = data[:, -1]
        # 分割数据
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_data, self.y_data)
        # 创建决策树模型
        # max_depth:树的深度 min_samples_split:内部节点再划分所需最小样本数
        self.model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=4)
        # 输入数据建立模型
        self.model.fit(self.x_train, self.y_train)
        self.create_tree(['x', 'y'], ['label0', 'label1'])
        self.draw_contour_line()

    def draw_contour_line(self):
        # 生成网格矩阵
        x_min, x_max = self.x_data[:, 0].min() - 1, self.x_data[:, 0].max() + 1
        y_min, y_max = self.x_data[:, 1].min() - 1, self.x_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        # 轮廓的高度值
        z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        # 等高线图,contourf轮廓带填充
        plt.contourf(xx, yy, z)
        plt.scatter(self.x_data[:, 0], self.x_data[:, 1], c=self.y_data)
        plt.show()


if __name__ == '__main__':
    dtree = decisionTree()
    # dtree.load_data("AllElectronics.csv")
    # dtree.create_tree_model(dtree.vec.get_feature_names(), dtree.lb.classes_)
    # print(dtree.predict(dtree.x_data[1]))

    # dtree.linear_dichotomy()
    # print(dtree.predict(dtree.x_data[1].reshape(1, -1)))
    # print(dtree.y_data[1])
    # print(classification_report(dtree.model.predict(dtree.x_data), dtree.y_data))

    dtree.non_linear_dichotomy()
    print(classification_report(dtree.model.predict(dtree.x_train), dtree.y_train))
    print(classification_report(dtree.model.predict(dtree.x_test), dtree.y_test))
