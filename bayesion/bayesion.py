'''
贝叶斯定理：文本分析。
    P(H|X) = P(X|H)P(H)/P(X)（大数定理）
    P(H|X):给定观测数据样本X，假设H成立的概率，后验概率。
    P(H):H的先验概率。
    P(X):X的先验概率。
    P(X|H):样本X中，H的概率。
fetch_20newsgroups:20newsgroups数据集是用于文本分类、文本挖据和信息检索研究的国际标准数据集之一。
数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。
CountVectorizer: 此方法构建单词的字典，每个单词实例被转换为特征向量的一个数值特征，每个元素是特定单词在文本中出现的次数  
TfidfVectorizer:使用了一个高级的计算方法，称为Term Frequency Inverse Document Frequency (TF-IDF)。这是一个衡量一
个词在文本或语料中重要性的统计方法。直觉上讲，该方法通过比较在整个语料库的词的频率，寻求在当前文档中频率较高的词。这是一
种将结果进行标准化的方法，可以避免因为有些词出现太过频繁而对一个实例的特征化作用不大
的情况(我猜测比如a和and在英语中出现的频率比较高，但是它们对于表征一个文本的作用没有什么作用)
'''
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


class bayesion(object):
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.news = None

    def load_data(self):
        self.news = fetch_20newsgroups(subset='all')

    def split_data(self, data):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, self.news.target)

    def bayes(self):
        # 方法1：
        # cv = CountVectorizer()
        # cv_data = cv.fit_transform(self.x_train)
        # 方法2：
        # 创建变换函数
        vectorizer = TfidfVectorizer()
        tfidf_data = vectorizer.fit_transform(self.news.data)
        self.split_data(tfidf_data)
        # 词条化以及创建词汇表
        mul_nb = MultinomialNB(alpha=0.01)
        mul_nb.fit(self.x_train, self.y_train)
        print(mul_nb.score(self.x_train, self.y_train))
        print(mul_nb.score(self.x_test, self.y_test))
        # tfidf_train = vectorizer.fit_transform(self.x_train)
        # scores = model_selection.cross_val_score(mul_nb, tfidf_train, self.y_train, cv=3, scoring='accuracy')
        # print("Accuracy: %0.3f" % (scores.mean()))


if __name__ == '__main__':
    bayes = bayesion()
    bayes.load_data()
    bayes.bayes()
