import graphviz
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import numpy as np

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
    def __init__(self, filename):
        self.header = None
        self.reader = None
        self.feature = []
        self.label = []
        self.x_data = None
        self.y_data = None
        self.model = None
        self.load_data(filename)

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

    def create_tree_model(self, create_tree=True):
        # 创建决策树模型，并训练
        self.model = tree.DecisionTreeClassifier(criterion='entropy')  # 支持的标准有"gini"代表的是Gini impurity(不纯度)与"entropy"代表的是information gain（信息增益）。
        self.model.fit(self.x_data, self.y_data)

        # 生成决策树
        if create_tree:
            dot_data = tree.export_graphviz(self.model,
                                            out_file=None,
                                            feature_names=self.vec.get_feature_names(),
                                            class_names=self.lb.classes_,
                                            filled=True,
                                            rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render('computer')

    def predict(self, xdata):
        return self.model.predict(xdata.reshape(1, -1))


if __name__ == '__main__':
    dtree = decisionTree("AllElectronics.csv")
    dtree.create_tree_model()
    print(dtree.predict(dtree.x_data[0]))
