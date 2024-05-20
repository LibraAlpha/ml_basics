import pathlib
import pandas as pd
from configs.path_internal import project_root_path
import numpy as np

class Mnist(object):
    """
    Mnist 数据集
    """

    def __init__(self):

        self.train_data_path = f"{project_root_path}/data/MNIST/train_images.csv"
        self.train_label_path = f"{project_root_path}/data/MNIST/train_labels.csv"
        self.test_data_path = f'{project_root_path}/data/MNIST/test_images.csv'
        self.test_label_path = f"{project_root_path}/data/MNIST/test_labels.csv"

        self.train_data, self.train_label = self.__load__(train=True)
        self.test_data, self.test_label = self.__load__(train=False)

    def get_data(self, train=True):
        if train:
            return self.train_data, self.train_label
        else:
            return self.test_data, self.test_label

    def __load__(self, train=True):
        """
        读取数据集
        :param train: 确定获取训练数据/测试数据
        :return:
        """
        if train:
            data = pd.read_csv(self.train_data_path, header=None)
            labels = pd.read_csv(self.train_label_path, header=None)
        else:
            data = pd.read_csv(self.test_data_path, header=None)
            labels = pd.read_csv(self.test_label_path, header=None)

        return data, labels
