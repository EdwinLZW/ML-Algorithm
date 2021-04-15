import numpy as np
import random
import operator
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import neighbors


class kNearestNeighbor(object):
    def __init__(self):
        self.iris = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self, sp_size):
        self.iris = datasets.load_iris()
        data_size = self.iris.data.shape[0]
        index = [i for i in range(data_size)]
        random.shuffle(index)  # 打乱顺序
        self.iris.data = self.iris.data[index]
        self.iris.target = self.iris.target[index]

        self.split_data(sp_size)

    def split_data(self, size):
        # 切分数据集，测试集：前size个， 训练集：后150-size个
        self.x_train = self.iris.data[size:]
        self.y_train = self.iris.target[size:]
        self.x_test = self.iris.data[:size]
        self.y_test = self.iris.target[:size]

    def knn(self, x_test, k):
        # 复制x_test,np.tile(a,(2,1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数。
        diff = np.tile(x_test, (self.x_train.shape[0], 1)) -self.x_train
        # 距离度量
        eqdiff = diff**2
        disdiff = np.sqrt(eqdiff.sum(axis=1))
        sortdiff = disdiff.argsort()  # argsort函数返回的是数组值从小到大的索引值

        class_count = {}
        for i in range(k):
            votelabel = self.y_train[sortdiff[i]]
            class_count[votelabel] = class_count.get(votelabel, 0) + 1
        # print(class_count)
        sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        # 获取数量最多的标签
        return sort_class_count[0][0]

    def prediction(self):
        predictions = []
        for i in range(self.x_test.shape[0]):
            predictions.append(self.knn(self.x_test[i], 5))
        # print(predictions)
        print(classification_report(self.y_test, predictions))

    def knn_sklearn(self):
        # 模型
        model = neighbors.KNeighborsClassifier()
        model.fit(self.x_train, self.y_train)
        prediction = model.predict(self.x_test)
        print(classification_report(self.y_test, prediction))


if __name__ == '__main__':
    knn = kNearestNeighbor()
    knn.load_data(30)
    knn.prediction()
    knn.knn_sklearn()