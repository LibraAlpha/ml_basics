from pathlib import Path
import pandas as pd


class Mnist(object):
    """
    Mnist 数据集
    """

    def __init__(self):

        self.train_path = Path('../data/mnist_train.csv')
        self.test_path = Path('../data/mnist_test.csv')

        self.train_data, self.train_label = self.load(self.train_path)
        self.test_data, self.test_label = self.load(self.test_path)

    def get_data(self, train=True):
        if train:
            return self.train_data, self.train_label
        else:
            return self.test_data, self.test_label

    def load(self, file):
        """
        读取数据集，第一列为标签列，其余为数据列
        :param file: csv文件位置
        :return:
        """
        df = pd.read_csv(file, header=None)

        labels = df[df.columns[0]].apply(lambda x: 0 if x == 0 else -1).values.tolist()

        data = df[df.columns[1:]].values.tolist()

        return data, labels
